import sys

import pytest

MESSAGE = {
    "env": (
        "constructing an Env is not allowed in the fast tier: it reads the "
        "graph and instantiates cells. Move this to tests/contract (behaviour "
        "of a backend), tests/concurrency (needs ranks) or tests/gradients "
        "(needs a trace)."
    ),
    "mpiexec": (
        "an mpiexec test does not belong in the fast tier: it spawns a "
        "subprocess per rank. Move this to tests/concurrency."
    ),
    "jit": (
        "tracing a jax function is not allowed in the fast tier: compilation "
        "dominates the runtime. Move this to tests/gradients -- or, if the "
        "traced path is precisely what this test is about and the trace is "
        "small, mark it @pytest.mark.traces."
    ),
}


def _fail(kind: str, detail: str = "") -> None:
    pytest.fail(f"{MESSAGE[kind]}{detail}", pytrace=False)


def install(monkeypatch, allow_tracing: bool = False) -> None:
    _forbid_env(monkeypatch)
    if not allow_tracing:
        _forbid_jit(monkeypatch)


def _forbid_env(monkeypatch) -> None:
    try:
        from livn.backend import Env
    except Exception:  # pragma: no cover - no backend importable at all
        return

    original = Env.__init__

    def refuse(self, *args, **kwargs):
        _fail("env", f" ({type(self).__name__})")
        return original(self, *args, **kwargs)  # pragma: no cover

    monkeypatch.setattr(Env, "__init__", refuse, raising=False)


def _forbid_jit(monkeypatch) -> None:
    jax = sys.modules.get("jax")
    if jax is None:
        return

    original = jax.jit

    def refuse(*args, **kwargs):
        jitted = original(*args, **kwargs)

        def refuse_call(*call_args, **call_kwargs):
            _fail("jit")
            return jitted(*call_args, **call_kwargs)  # pragma: no cover

        return refuse_call

    monkeypatch.setattr(jax, "jit", refuse, raising=False)


def refuse_mpiexec_items(items) -> None:
    offenders = [item.nodeid for item in items if item.get_closest_marker("mpiexec")]
    if offenders:
        raise pytest.UsageError(MESSAGE["mpiexec"] + "\n  " + "\n  ".join(offenders))
