import faulthandler
import json
import os
import sys
import threading
import warnings
from pathlib import Path

import pytest

from testing.collectives import tracer
from testing.paths import REPO_ROOT

MPI_SUBPROCESS_ENV = "TEST_MPI_SUBTEST"
TEST_REPORT_DIR_ENV = "TEST_MPI_REPORT_DIR"

HUNG_EXIT_CODE = 87

ABORT_GRACE_SECONDS = 5.0


def rank() -> int:
    from mpi4py import MPI

    return int(MPI.COMM_WORLD.rank)


def report_dir() -> Path:
    return Path(os.environ.get(TEST_REPORT_DIR_ENV, ""))


def install_tracer(config) -> None:
    if os.environ.get(SYMMETRY_ENV, SYMMETRY_DEFAULT) == "off":
        return
    tracer.install(roots=(str(REPO_ROOT),))
    tracer.watch(WATCHED_MODULES)


def install_reportlog(config) -> None:
    try:
        from pytest_reportlog.plugin import ReportLogPlugin
    except ImportError:
        return

    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError:
        return

    path = report_dir() / f"reportlog-{rank()}.jsonl"
    config._mpiexec_reporter = reporter = ReportLogPlugin(config, path)
    config.pluginmanager.register(reporter)


SYMMETRY_ENV = "TEST_MPI_SYMMETRY"

SYMMETRY_DEFAULT = "strict"

WATCHED_MODULES = (
    "livn.utils",
    "livn.system",
    "livn.cells",
    "livn.env",
    "livn.decoding",
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    timer = _arm(item.nodeid, _timeout_for(item))
    try:
        return (yield)
    finally:
        if timer is not None:
            timer.cancel()


@pytest.fixture(autouse=True)
def _collective_symmetry(request):
    mode = _symmetry_mode(request.node)
    if mode == "off":
        yield
        return

    tracer.watch(WATCHED_MODULES)
    tracer.start_test(request.node.nodeid)

    yield

    _check_symmetry(request.node, mode)


def _symmetry_mode(item) -> str:
    from mpi4py import MPI

    if MPI.COMM_WORLD.size < 2:
        return "off"

    for mark in item.iter_markers("mpiexec"):
        if "symmetry" in mark.kwargs:
            value = mark.kwargs["symmetry"]
            if value is False:
                return "off"
            if isinstance(value, str):
                return value
            break

    return os.environ.get(SYMMETRY_ENV, SYMMETRY_DEFAULT)


def _check_symmetry(node, mode: str) -> None:
    from testing.collectives import CollectiveAsymmetry, verify

    trace = tracer.finish_test()
    try:
        verify(node.nodeid, trace, report_dir=str(report_dir()))
    except CollectiveAsymmetry as e:
        if mode == "strict":
            raise
        warnings.warn(
            f"collective asymmetry (not failing, symmetry={mode}):\n{e}",
            CollectiveAsymmetryWarning,
            stacklevel=1,
        )


class CollectiveAsymmetryWarning(UserWarning):
    pass


def _timeout_for(item) -> float | None:
    mark = item.get_closest_marker("mpiexec")
    if mark is None:
        return None
    try:
        return float(mark.kwargs.get("timeout", 30))
    except (TypeError, ValueError):
        return None


def _arm(nodeid: str, timeout: float | None):
    if not timeout:
        return None

    def fire():
        try:
            from mpi4py import MPI
        except ImportError:  # pragma: no cover - the parent checked already
            os._exit(HUNG_EXIT_CODE)

        me = int(MPI.COMM_WORLD.rank)
        try:
            (report_dir() / f"hang-{me}.json").write_text(
                json.dumps(
                    {
                        "nodeid": nodeid,
                        "rank": me,
                        "timeout": timeout,
                        "pid": os.getpid(),
                    }
                )
            )
        except OSError:  # pragma: no cover - nothing better to do
            pass

        print(
            f"[rank {me}] watchdog: {nodeid} did not finish in {timeout}s",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

        threading.Timer(ABORT_GRACE_SECONDS, lambda: os._exit(HUNG_EXIT_CODE)).start()
        MPI.COMM_WORLD.Abort(HUNG_EXIT_CODE)

    timer = threading.Timer(timeout, fire)
    timer.daemon = True
    timer.start()
    return timer
