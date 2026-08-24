import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from testing.capabilities import backend_supports, missing

MPI_SUBPROCESS_ENV = "TEST_MPI_SUBTEST"
TEST_REPORT_DIR_ENV = "TEST_MPI_REPORT_DIR"
MPI_MARKER_NAME = "mpiexec"
MPIEXEC = "mpiexec"


def pytest_addoption(parser):
    group = parser.getgroup("mpiexec")
    group.addoption(
        "--mpiexec",
        action="store",
        dest="mpiexec",
        default=MPIEXEC,
        help="mpiexec executable (default: mpiexec)",
    )
    group.addoption(
        "--mpi-batch",
        action="store",
        dest="mpi_batch",
        choices=["yes", "no"],
        default="yes",
        help=(
            "run all mpiexec tests of the same rank count in one subprocess "
            "instead of one each (default: yes). --mpi-batch=no restores one "
            "subprocess per test, which is the thing to try first if a test "
            "passes alone and fails in company"
        ),
    )
    group.addoption(
        "--mpi-localize",
        action="store",
        dest="mpi_localize",
        choices=["yes", "no"],
        default="yes",
        help=(
            "when a batch dies without saying which test hung, re-run its "
            "unresolved members one at a time to find out (default: yes)"
        ),
    )
    group.addoption(
        "--mpi-no-lock",
        action="store_true",
        dest="mpi_no_lock",
        default=False,
        help="do not serialize mpiexec subprocesses against other pytest runs",
    )


def pytest_configure(config):
    global MPIEXEC

    mpiexec = config.getoption("mpiexec", default=None)
    if mpiexec:
        MPIEXEC = mpiexec

    config.addinivalue_line(
        "markers",
        f"{MPI_MARKER_NAME}(n=, timeout=, isolated=, env=): run under mpiexec. "
        "isolated=True gives the test its own subprocess, for a test that "
        "mutates process-global state other tests depend on",
    )
    config.addinivalue_line(
        "markers",
        "needs(*capabilities): skip unless the backend declares them "
        "(see livn.types.Capability)",
    )
    config.addinivalue_line(
        "markers", "slow: a long reference comparison, deselected by default"
    )
    config.addinivalue_line(
        "markers",
        "traces: the traced path is what this test is about; the fast tier's "
        "no-tracing budget does not apply",
    )

    if os.getenv(MPI_SUBPROCESS_ENV):
        from testing import mpiexec_child

        config.pluginmanager.register(mpiexec_child)
        mpiexec_child.install_tracer(config)
        mpiexec_child.install_reportlog(config)


def pytest_unconfigure(config):
    reporter = getattr(config, "_mpiexec_reporter", None)
    if reporter:
        reporter.close()


def pytest_runtest_setup(item):
    for mark in item.iter_markers("needs"):
        absent = missing(*mark.args)
        if absent:
            from livn.backend import backend

            name = backend() or "no backend"
            pytest.skip(f"{name} does not support {', '.join(map(str, absent))}")


def pytest_collection_modifyitems(config, items):
    if os.getenv("LIVN_BACKEND") != "neuron":
        return

    try:
        from livn.backend.neuron.mechanisms import nrnivmodl
    except ImportError:
        return

    if nrnivmodl() is not None:
        return

    skip = pytest.mark.skip(
        reason="nrnivmodl not found: NEURON mechanisms cannot be compiled "
        "(install the 'neuron' extra, or put its bin directory on PATH)"
    )
    for item in items:
        item.add_marker(skip)


def _is_skipped(item) -> bool:
    for mark in item.iter_markers("needs"):
        if any(not backend_supports(c) for c in mark.args):
            return True

    try:
        from _pytest.skipping import evaluate_skip_marks
    except ImportError:  # pragma: no cover - pytest internals moved
        if any(True for _ in item.iter_markers("skip")):
            return True
        return any(mark.args and mark.args[0] for mark in item.iter_markers("skipif"))

    return evaluate_skip_marks(item) is not None


@dataclass(frozen=True)
class BatchKey:
    n: int
    """Rank count. A process runs on one, so this is the primary split."""

    isolated: str | None
    """Set to the nodeid when a test must own its process.

    ``test_recompile_and_smoke`` is the case: it wipes the compiled-mechanism
    cache and loads models whose NEURON ``SUFFIX``es overlap, so two of them in
    one interpreter is not a slow test, it is a broken one.
    """

    env: tuple[tuple[str, str], ...]
    """Environment the child needs. The backend is frozen at import, so two
    tests wanting different backends can never share a process."""


@dataclass
class Outcome:
    kind: str
    message: str = ""
    sections: list = field(default_factory=list)
    duration: float = 0.0


def _mpi_kwargs(item) -> dict:
    merged: dict = {}
    for mark in reversed(list(item.iter_markers(MPI_MARKER_NAME))):
        merged.update(mark.kwargs)
    return merged


def _batch_key(item, mpi_mark=None) -> BatchKey:
    kwargs = _mpi_kwargs(item)

    if getattr(item, "callspec", None) and "mpiexec_n" in item.callspec.params:
        n = int(item.callspec.params["mpiexec_n"])
    else:
        n = int(kwargs.get("n", 2))

    overlay = kwargs.get("env") or {}
    return BatchKey(
        n=n,
        isolated=item.nodeid if kwargs.get("isolated") else None,
        env=tuple(sorted((str(k), str(v)) for k, v in overlay.items())),
    )


def _timeout_of(item) -> float:
    return float(_mpi_kwargs(item).get("timeout", 30))


def pytest_runtest_protocol(item, nextitem):
    if os.getenv(MPI_SUBPROCESS_ENV):
        return None

    mpi_mark = item.get_closest_marker(MPI_MARKER_NAME)
    if not mpi_mark:
        return None

    if _is_skipped(item):
        return None

    try:
        import mpi4py  # noqa: F401
    except ImportError:
        _report_skip(item, "mpi4py not available")
        return True

    _run_mpi_test(item, mpi_mark)
    return True


def _cache(config) -> dict:
    cache = getattr(config, "_mpi_outcomes", None)
    if cache is None:
        cache = config._mpi_outcomes = {}
    return cache


def _outcome_for(item, mpi_mark) -> Outcome:
    config = item.config
    cache = _cache(config)

    if item.nodeid not in cache:
        key = _batch_key(item, mpi_mark)
        members = _group_members(item.session, key, cache)
        cache.update(_run_batch(config, key, members))

    if item.nodeid not in cache:  # pragma: no cover - _run_batch covers everyone
        cache[item.nodeid] = Outcome("failed", "the batch reported nothing for it")

    return cache[item.nodeid]


def _group_members(session, key: BatchKey, cache: dict) -> list:
    members = []
    for candidate in session.items:
        if candidate.nodeid in cache:
            continue
        if not candidate.get_closest_marker(MPI_MARKER_NAME):
            continue
        if _batch_key(candidate) != key:
            continue
        if _is_skipped(candidate):
            continue
        members.append(candidate)
    return members


def _report_skip(item, reason):
    hook = item.config.hook
    hook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    skip_exc = pytest.skip.Exception(reason)
    call = pytest.CallInfo.from_call(lambda: (_ for _ in ()).throw(skip_exc), "call")
    report = hook.pytest_runtest_makereport(item=item, call=call)
    hook.pytest_runtest_logreport(report=report)
    hook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)


def _run_mpi_test(item, mpi_mark):
    hook = item.config.hook
    hook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

    item.session._setupstate.setup(item)

    setup_call = pytest.CallInfo.from_call(lambda: None, "setup")
    setup_report = hook.pytest_runtest_makereport(item=item, call=setup_call)
    hook.pytest_runtest_logreport(report=setup_report)

    call = pytest.CallInfo.from_call(lambda: _mpi_subprocess(item, mpi_mark), "call")
    call_report = hook.pytest_runtest_makereport(item=item, call=call)
    hook.pytest_runtest_logreport(report=call_report)

    teardown_call = pytest.CallInfo.from_call(lambda: None, "teardown")
    teardown_report = hook.pytest_runtest_makereport(item=item, call=teardown_call)
    hook.pytest_runtest_logreport(report=teardown_report)

    item.session._setupstate.teardown_exact(None)

    hook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)


def _run_in_process_group(exe, env, timeout):
    proc = subprocess.Popen(
        exe,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGUSR1)
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        _signal_group(proc, signal.SIGKILL)
        try:
            extra_out, extra_err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
            extra_out, extra_err = "", ""
        raise subprocess.TimeoutExpired(
            exe, timeout, output=stdout + extra_out, stderr=stderr + extra_err
        ) from None

    return subprocess.CompletedProcess(exe, proc.returncode, stdout, stderr)


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf8", "replace")
    return str(value)


def _signal_group(proc, sig) -> None:
    # pragma: no cover - the group may already be gone
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), sig)


def _mpi_subprocess(item, mpi_mark):
    if item.config.getoption("mpi_batch", "yes") == "no":
        outcome = _run_batch(item.config, _batch_key(item, mpi_mark), [item])[
            item.nodeid
        ]
    else:
        outcome = _outcome_for(item, mpi_mark)

    for title, body in outcome.sections:
        item.add_report_section("call", title, body)

    if outcome.kind == "skipped":
        pytest.skip(outcome.message or "skipped in mpi subprocess")
    if outcome.kind == "failed":
        pytest.fail(outcome.message, pytrace=False)


def _run_batch(config, key: BatchKey, members: list) -> dict:
    if not members:
        return {}

    nodeids = [item.nodeid for item in members]
    timeout = sum(_timeout_of(m) for m in members)
    timeout += 15 + 3 * key.n

    exe = _command(config, key, members)
    env = dict(os.environ)
    env[MPI_SUBPROCESS_ENV] = "1"
    env["TQDM_DISABLE"] = "1"
    env.update(dict(key.env))

    command_section = (
        "mpiexec command",
        f"{MPI_SUBPROCESS_ENV}=1 {shlex.join(exe)}",
    )

    if len(members) > 1:
        _announce(config, key, members)

    with TemporaryDirectory() as report_dir:
        env[TEST_REPORT_DIR_ENV] = report_dir

        timed_out = None
        try:
            with _mpi_slot(config):
                completed = _run_in_process_group(exe, env, timeout)
        except subprocess.TimeoutExpired as e:
            timed_out, completed = e, None

        reports = _index_reports(report_dir, key.n)
        hung = _hung_ranks(report_dir)

    outcomes = {}
    for nodeid in nodeids:
        outcomes[nodeid] = _outcome_from(reports.get(nodeid, []), key.n)
        outcomes[nodeid].sections.append(command_section)

    unresolved = [n for n in nodeids if outcomes[n] is None or not reports.get(n)]
    if unresolved:
        _explain_unresolved(
            config, key, members, outcomes, unresolved, hung, timed_out, completed
        )

    output = _output_sections(timed_out or completed)
    for nodeid in nodeids:
        if outcomes[nodeid].kind == "failed":
            outcomes[nodeid].sections.extend(output)

    return outcomes


def _command(config, key: BatchKey, members: list) -> list[str]:
    launcher = [] if key.n == 1 else [*shlex.split(MPIEXEC), "-n", str(key.n)]
    return [
        *launcher,
        sys.executable,
        "-m",
        "pytest",
        "--quiet",
        "--no-header",
        "--no-summary",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(config.rootpath),
        "-m",
        "",
        *[f"{item.fspath}::{item.name}" for item in members],
    ]


def _announce(config, key: BatchKey, members: list) -> None:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    reporter.write_line(
        f"[mpiexec] batch of {len(members)} on {key.n} rank(s)", yellow=True
    )


def _outcome_from(reports: list[dict], n: int) -> Outcome:
    if not reports:
        return Outcome("failed", "no report from the mpi subprocess")

    kinds = {r["outcome"] for r in reports}
    duration = max((float(r.get("duration") or 0.0) for r in reports), default=0.0)

    if "failed" in kinds:
        return Outcome(
            "failed", "\n".join(_extract_failure_messages(reports)), duration=duration
        )

    reported = {r["_mpi_rank"] for r in reports}
    if "skipped" in kinds:
        return Outcome("skipped", _extract_skip_reason(reports), duration=duration)

    missing = sorted(set(range(n)) - reported)
    if missing:
        return Outcome(
            "failed",
            f"rank(s) {', '.join(map(str, missing))} of {n} never reported on this "
            "test while the others finished it -- the usual cause is a collective "
            "only some ranks reach",
            duration=duration,
        )

    return Outcome("passed", duration=duration)


def _explain_unresolved(
    config, key, members, outcomes, unresolved, hung, timed_out, completed
):
    culprit = _culprit(hung, [m.nodeid for m in members], outcomes)

    if (
        timed_out is not None
        and culprit is None
        and len(members) > 1
        and config.getoption("mpi_localize", "yes") == "yes"
    ):
        for item in members:
            if item.nodeid in unresolved:
                single = _run_batch(config, key, [item])
                outcomes.update(single)
        return

    for nodeid in unresolved:
        if nodeid == culprit:
            outcomes[nodeid] = Outcome(
                "failed",
                _hang_message(hung, nodeid, timed_out, key),
            )
        elif culprit is not None:
            outcomes[nodeid] = Outcome(
                "failed",
                f"not run: the batch on {key.n} rank(s) died in {culprit}",
            )
        elif timed_out is not None:
            outcomes[nodeid] = Outcome(
                "failed",
                f"the batch on {key.n} rank(s) did not complete in "
                f"{timed_out.timeout:.0f} seconds, and no rank said where",
            )
        else:
            code = getattr(completed, "returncode", "?")
            outcomes[nodeid] = Outcome(
                "failed",
                f"no report from the mpi subprocess (exit code {code})",
            )


def _hang_message(hung, nodeid, timed_out, key) -> str:
    ranks = sorted(r for r, info in hung.items() if info.get("nodeid") == nodeid)
    where = f"rank{'s' if len(ranks) > 1 else ''} {', '.join(map(str, ranks))}"
    seconds = next(iter(hung.values())).get("timeout") if hung else None
    limit = f"{seconds:.0f}s" if seconds else "its timeout"
    elsewhere = sorted(set(hung) - set(ranks))
    trailer = (
        f"; rank(s) {', '.join(map(str, elsewhere))} were somewhere else, which is "
        "itself the asymmetry"
        if elsewhere
        else ""
    )
    return f"hung: {where} of {key.n} did not finish within {limit}{trailer}"


def _culprit(hung: dict, nodeids: list[str], outcomes: dict) -> str | None:
    named = {info.get("nodeid") for info in hung.values() if info.get("nodeid")}
    if len(named) == 1:
        return named.pop()
    if named:
        return sorted(named)[0]

    for nodeid in nodeids:
        if (outcomes.get(nodeid) is None or outcomes[nodeid].kind == "failed") and (
            outcomes.get(nodeid) and outcomes[nodeid].message.startswith("no report")
        ):
            return nodeid
    return None


def _hung_ranks(report_dir: str) -> dict:
    out = {}
    for path in Path(report_dir).glob("hang-*.json"):
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):  # pragma: no cover
            continue
        out[int(info.get("rank", -1))] = info
    return out


def _output_sections(result) -> list:
    sections = []
    for name in ("stdout", "stderr"):
        value = getattr(result, name, None)
        if value:
            sections.append((f"mpiexec {name}", _as_text(value)))
    return sections


@contextlib.contextmanager
def mpi_slot(rootpath):
    digest = hashlib.sha1(str(rootpath).encode()).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / f"livn-mpiexec-{digest}.lock"
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def _mpi_slot(config):
    if config.getoption("mpi_no_lock", False):
        yield
        return

    with mpi_slot(config.rootpath):
        yield


def _index_reports(reportlog_dir: str, n: int) -> dict:
    reports: dict = {}
    for rank in range(n):
        path = os.path.join(reportlog_dir, f"reportlog-{rank}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                for report in _parse_jsonl(line):
                    if report.get("$report_type") != "TestReport":
                        continue
                    if report.get("when") != "call" and report["outcome"] == "passed":
                        continue
                    report["_mpi_rank"] = rank
                    reports.setdefault(report.get("nodeid", ""), []).append(report)
    return reports


def _parse_jsonl(line: str) -> list[dict]:
    results = []
    decoder = json.JSONDecoder()
    line = line.strip()
    pos = 0
    while pos < len(line):
        try:
            obj, end = decoder.raw_decode(line, pos)
        except json.JSONDecodeError:
            break
        results.append(obj)
        while end < len(line) and line[end] in " \t\r\n":
            end += 1
        pos = end
    return results


def _extract_skip_reason(reports: list[dict]) -> str:
    for r in reports:
        if r["outcome"] != "skipped":
            continue
        lr = r.get("longrepr")
        if isinstance(lr, (list, tuple)) and len(lr) >= 3:
            return str(lr[2])
        if isinstance(lr, str):
            return lr
    return "skipped in mpi subprocess"


def _extract_failure_messages(reports: list[dict]) -> list[str]:
    seen_ranks: set[int] = set()
    messages: list[str] = []
    for r in reports:
        if r["outcome"] != "failed":
            continue
        rank = r.get("_mpi_rank", -1)
        if rank in seen_ranks:
            continue
        seen_ranks.add(rank)

        lr = r.get("longrepr")
        if isinstance(lr, dict):
            crash = lr.get("reprcrash", {})
            msg = crash.get("message", "unknown failure")
            full = _full_repr(lr)
            if full and len(full) > len(msg):
                msg = full
        elif isinstance(lr, str):
            msg = lr
        elif isinstance(lr, (list, tuple)):
            msg = str(lr[-1]) if lr else "unknown failure"
        else:
            msg = str(lr)
        messages.append(f"[rank {rank}] {msg}")
    return messages or ["unknown mpi test failure"]


def _full_repr(longrepr: dict) -> str:
    chain = longrepr.get("chain")
    if not chain:
        return ""

    lines = []
    for entry in chain:
        reprtraceback = entry[0] if isinstance(entry, (list, tuple)) else None
        if not isinstance(reprtraceback, dict):
            continue
        for frame in reprtraceback.get("reprentries", []):
            lines.extend(frame.get("lines", []) or [])
    return "\n".join(line for line in lines if line.strip())
