from livn.backend.common import *  # noqa: F403
from livn.backend.config import backend

_MISSING = {
    "neuron": (
        ("neuron", "mpi4py", "neuroh5", "h5py"),
        "the NEURON backend is not installed. It needs MPI, a parallel HDF5 "
        "and neuroh5:\n"
        "    git clone https://github.com/livn-org/livn.git\n"
        "    cd livn && uv sync --group neuron\n"
        "See https://livn-org.github.io/livn/installation/ for the system "
        "packages to install first.",
    ),
    "brian2": (
        ("brian2", "cleo", "cleosim"),
        "the brian2 backend is not installed: `pip install livn[brian2]`.",
    ),
    "diffrax": (
        ("jax", "jaxlib", "diffrax", "equinox", "optimistix", "lineax", "optax"),
        "the diffrax backend is not installed: `pip install livn[diffrax]`.",
    ),
    "native": (
        ("livn.backend.native",),
        "the native backend's library could not be loaded. Reinstall livn, or "
        "build it from a checkout with a C compiler on PATH.",
    ),
}


def _explain(name: str, error: ImportError) -> ImportError | None:
    found = _MISSING.get(name)
    if found is None:
        return None
    owned, detail = found
    missing = getattr(error, "name", None) or ""
    if not any(missing == o or missing.startswith(o + ".") for o in owned):
        return None
    return ImportError(f"LIVN_BACKEND={name}: {detail}")


if backend() == "":
    from livn.backend.default import *  # noqa: F403
elif backend() == "brian2":
    try:
        from livn.backend.brian2 import *  # noqa: F403
    except ImportError as e:
        raise (_explain("brian2", e) or e) from e
elif backend() == "neuron":
    try:
        from livn.backend.neuron import *  # noqa: F403
    except ImportError as e:
        raise (_explain("neuron", e) or e) from e
elif backend() == "diffrax":
    try:
        from livn.backend.diffrax import *  # noqa: F403
    except ImportError as e:
        raise (_explain("diffrax", e) or e) from e
elif backend() == "native":
    try:
        from livn.backend.native import *  # noqa: F403
    except ImportError as e:
        raise (_explain("native", e) or e) from e
else:
    try:
        import importlib

        _mod = importlib.import_module(backend())
        globals().update(
            {k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")}
        )
    except ImportError as e:
        raise ImportError(f"livn: backend not found: {backend()}") from e
