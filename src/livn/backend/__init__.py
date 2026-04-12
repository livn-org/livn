from livn.backend.common import *  # noqa: F403
from livn.backend.config import backend

if backend() == "":
    from livn.backend.default import *  # noqa: F403
elif backend() == "brian2":
    from livn.backend.brian2 import *  # noqa: F403
elif backend() == "neuron":
    from livn.backend.neuron import *  # noqa: F403
elif backend() == "diffrax":
    from livn.backend.diffrax import *  # noqa: F403
else:
    try:
        import importlib

        _mod = importlib.import_module(backend())
        globals().update(
            {k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")}
        )
    except ImportError:
        raise ImportError(f"livn: backend not found: {backend()}")
