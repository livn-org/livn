import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

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


def pytest_configure(config):
    global MPIEXEC

    mpiexec = config.getoption("mpiexec", default=None)
    if mpiexec:
        MPIEXEC = mpiexec

    config.addinivalue_line("markers", f"{MPI_MARKER_NAME}: Run test with mpiexec")

    if os.getenv(MPI_SUBPROCESS_ENV):
        try:
            from pytest_reportlog.plugin import ReportLogPlugin
        except ImportError:
            return

        from mpi4py import MPI

        rank = MPI.COMM_WORLD.rank
        reportlog_dir = Path(os.getenv(TEST_REPORT_DIR_ENV, ""))
        report_path = reportlog_dir / f"reportlog-{rank}.jsonl"
        config._mpiexec_reporter = reporter = ReportLogPlugin(config, report_path)
        config.pluginmanager.register(reporter)


def pytest_unconfigure(config):
    reporter = getattr(config, "_mpiexec_reporter", None)
    if reporter:
        reporter.close()


def pytest_runtest_protocol(item, nextitem):
    if os.getenv(MPI_SUBPROCESS_ENV):
        return

    mpi_mark = item.get_closest_marker(MPI_MARKER_NAME)
    if not mpi_mark:
        return

    _run_mpi_test(item, mpi_mark)
    return True



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


def _mpi_subprocess(item, mpi_mark):
    if getattr(item, "callspec", None) and "mpiexec_n" in item.callspec.params:
        n = item.callspec.params["mpiexec_n"]
    else:
        n = mpi_mark.kwargs.get("n", 2)

    timeout = mpi_mark.kwargs.get("timeout", 30)

    exe = [
        MPIEXEC,
        "-n",
        str(n),
        sys.executable,
        "-m",
        "pytest",
        "--quiet",
        "--no-header",
        "--no-summary",
        f"{item.fspath}::{item.name}",
    ]

    env = dict(os.environ)
    env[MPI_SUBPROCESS_ENV] = "1"

    item.add_report_section(
        "call",
        "mpiexec command",
        f"{MPI_SUBPROCESS_ENV}=1 {shlex.join(exe)}",
    )

    with TemporaryDirectory() as reportlog_dir:
        env[TEST_REPORT_DIR_ENV] = reportlog_dir

        try:
            p = subprocess.run(
                exe,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            if e.stdout:
                item.add_report_section(
                    "call", "stdout", e.stdout.decode("utf8", "replace")
                )
            if e.stderr:
                item.add_report_section(
                    "call", "stderr", e.stderr.decode("utf8", "replace")
                )
            pytest.fail(
                f"mpi test did not complete in {timeout} seconds", pytrace=False
            )

        reports = _collect_reports(reportlog_dir, n)

    if not reports:
        _attach_output(item, p)
        pytest.fail("No test reports captured from mpi subprocess", pytrace=False)

    outcomes = {r["outcome"] for r in reports}

    if outcomes <= {"skipped", "passed"} and any(
        r["outcome"] == "skipped" for r in reports
    ):
        reason = _extract_skip_reason(reports)
        pytest.skip(reason)

    if "failed" in outcomes:
        _attach_output(item, p)
        messages = _extract_failure_messages(reports)
        pytest.fail("\n".join(messages), pytrace=False)

    if p.returncode:
        _attach_output(item, p)
        pytest.fail(
            f"mpiexec returned non-zero exit code {p.returncode}", pytrace=False
        )


def _collect_reports(reportlog_dir: str, n: int) -> list[dict]:
    reports: list[dict] = []
    for rank in range(n):
        path = os.path.join(reportlog_dir, f"reportlog-{rank}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                report = json.loads(line)
                if report.get("$report_type") != "TestReport":
                    continue
                report["_mpi_rank"] = rank
                reports.append(report)
    return reports


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
        elif isinstance(lr, str):
            msg = lr
        elif isinstance(lr, (list, tuple)):
            msg = str(lr[-1]) if lr else "unknown failure"
        else:
            msg = str(lr)
        messages.append(f"[rank {rank}] {msg}")
    return messages or ["unknown mpi test failure"]


def _attach_output(item, p) -> None:
    if p.stdout:
        item.add_report_section("call", "mpiexec stdout", p.stdout)
    if p.stderr:
        item.add_report_section("call", "mpiexec stderr", p.stderr)
