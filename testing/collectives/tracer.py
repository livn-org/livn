import contextlib
import sys
import threading
from contextlib import contextmanager
from typing import NamedTuple

COLLECTIVE_OPS = frozenset(
    {
        "Barrier",
        "Ibarrier",
        "bcast",
        "Bcast",
        "Ibcast",
        "gather",
        "Gather",
        "Gatherv",
        "gatherv",
        "allgather",
        "Allgather",
        "Allgatherv",
        "scatter",
        "Scatter",
        "Scatterv",
        "reduce",
        "Reduce",
        "allreduce",
        "Allreduce",
        "Reduce_scatter",
        "Scan",
        "Exscan",
        "alltoall",
        "Alltoall",
        "Alltoallv",
        "Alltoallw",
        "Split",
        "Split_type",
        "Dup",
        "Idup",
        "Create",
        "Create_group",
        "Cart_create",
        "Free",
        "Disconnect",
        "Clone",
    }
)

P2P_OPS = frozenset(
    {
        "send",
        "recv",
        "isend",
        "irecv",
        "Send",
        "Recv",
        "Isend",
        "Irecv",
        "Sendrecv",
        "sendrecv",
        "ssend",
        "bsend",
        "Probe",
        "Iprobe",
        "Waitall",
    }
)

LOCAL_OPS = frozenset(
    {
        "Get_rank",
        "Get_size",
        "Get_group",
        "Get_attr",
        "Set_attr",
        "Get_name",
        "Set_name",
        "py2f",
        "f2py",
        "Is_inter",
        "Compare",
        "Test",
        "Wait",
    }
)

OPAQUE_MODULES = ("neuroh5",)


_TOOL_IDS = (3, 4, 2)


class Record(NamedTuple):
    op: str
    site: str
    root: int | None
    shape: str

    def signature(self) -> tuple:
        return (self.op, self.root)


class _State:
    def __init__(self):
        self.tool_id: int | None = None
        self.nodeid: str | None = None
        self.by_comm: dict = {}
        self.p2p: list = []
        self.opaque: list = []
        self.paused = False
        self.roots: tuple = ()
        self.lock = threading.Lock()
        self.lines: dict = {}
        self.comm_keys: dict = {}
        self.comm_ordinals: dict = {}
        self.watched: set = set()


_state = _State()


def install(roots=()) -> bool:
    if sys.version_info < (3, 12):  # pragma: no cover - PEP 669 landed in 3.12
        return False
    if _state.tool_id is not None:
        return True

    mon = sys.monitoring
    for candidate in _TOOL_IDS:
        try:
            mon.use_tool_id(candidate, "livn-collectives")
        except ValueError:
            continue
        _state.tool_id = candidate
        break
    else:  # pragma: no cover - every id taken
        return False

    _state.roots = tuple(str(r) for r in roots)
    mon.register_callback(_state.tool_id, mon.events.CALL, _on_call)
    return True


def watch(prefixes=("livn", "testing")) -> int:
    if _state.tool_id is None:
        return 0

    mon = sys.monitoring
    armed = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefixes) or module is None:
            continue
        for code in _module_code(module):
            if code in _state.watched:
                continue
            try:
                mon.set_local_events(_state.tool_id, code, mon.events.CALL)
            except (ValueError, TypeError):  # pragma: no cover
                continue
            _state.watched.add(code)
            armed += 1
    return armed


def _module_code(module):
    import types

    seen = set()
    stack = []
    for value in vars(module).values():
        if isinstance(value, types.FunctionType):
            stack.append(value.__code__)
        elif isinstance(value, type):
            for attribute in vars(value).values():
                function = getattr(attribute, "__func__", attribute)
                if isinstance(function, types.FunctionType):
                    stack.append(function.__code__)

    while stack:
        code = stack.pop()
        if code in seen:
            continue
        seen.add(code)
        yield code
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))


def uninstall() -> None:
    if _state.tool_id is None:
        return
    mon = sys.monitoring
    for code in _state.watched:
        with contextlib.suppress(ValueError, TypeError):  # pragma: no cover
            mon.set_local_events(_state.tool_id, code, 0)
    _state.watched.clear()
    mon.register_callback(_state.tool_id, mon.events.CALL, None)
    mon.free_tool_id(_state.tool_id)
    _state.tool_id = None


def start_test(nodeid: str) -> None:
    _state.nodeid = nodeid
    _state.by_comm = {}
    _state.p2p = []
    _state.opaque = []


def finish_test() -> dict:
    return _state.by_comm


def p2p_trace() -> list:
    return _state.p2p


def opaque_trace() -> list:
    return _state.opaque


@contextmanager
def paused():
    before = _state.paused
    _state.paused = True
    try:
        yield
    finally:
        _state.paused = before


def record_synthetic(op: str, site: str, comm=None) -> None:
    from mpi4py import MPI

    _append(comm if comm is not None else MPI.COMM_WORLD, Record(op, site, None, "-"))


def _on_call(code, offset, callable_, arg0):
    if _state.paused:
        return

    name = getattr(callable_, "__name__", None)
    if name is None:
        return

    if _is_comm(arg0):
        if name in COLLECTIVE_OPS:
            _append(arg0, _record(name, code, offset))
            if name in ("Free", "Disconnect"):
                _state.comm_keys.pop(_handle(arg0), None)
            return
        if name in P2P_OPS:
            _state.p2p.append(_record(name, code, offset))
            return
        return

    module = getattr(callable_, "__module__", "") or ""
    if module.startswith(OPAQUE_MODULES):
        _state.opaque.append(_record(f"{module}.{name}", code, offset))

    return


def _is_comm(value) -> bool:
    try:
        from mpi4py import MPI
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, MPI.Comm)


def _record(op: str, code, offset) -> Record:
    return Record(op=op, site=_site(code, offset), root=None, shape="-")


def _site(code, offset) -> str:
    spans = _state.lines.get(code)
    if spans is None:
        spans = _state.lines[code] = [
            (start, end, line) for start, end, line in code.co_lines() if line
        ]
    line = next(
        (line for start, end, line in spans if start <= offset < end),
        code.co_firstlineno,
    )

    filename = code.co_filename
    for root in _state.roots:
        if filename.startswith(root):
            filename = filename[len(root) :].lstrip("/")
            break
    return f"{filename}:{line}"


def _append(comm, record: Record) -> None:
    key = comm_key(comm)
    if key is None:
        return
    _state.by_comm.setdefault(key, []).append(record)


def _handle(comm):
    try:
        return comm.py2f()
    except Exception:  # pragma: no cover
        return None


def comm_key(comm):
    from mpi4py import MPI

    try:
        handle = comm.py2f()
    except Exception:  # pragma: no cover - a freed communicator
        return None

    cached = _state.comm_keys.get(handle)
    if cached is not None:
        return cached

    try:
        group = comm.Get_group()
        world = MPI.COMM_WORLD.Get_group()
        members = tuple(
            MPI.Group.Translate_ranks(group, list(range(group.Get_size())), world)
        )
    except Exception:  # pragma: no cover
        return None

    ordinal = _state.comm_ordinals.get(members, 0)
    _state.comm_ordinals[members] = ordinal + 1

    key = (members, ordinal)
    _state.comm_keys[handle] = key
    return key
