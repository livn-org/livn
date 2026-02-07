"""livn"""

__doc__ = """A testbed for learning to interact with in vitro neural networks"""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()


def make(system_url: str = "hf://datasets/livn-org/livn/systems/data/S1"):
    """Initializes a default env from a system directory"""
    from livn.env import Env
    from livn.system import fetch

    env = Env(fetch(system_url)).init()

    if (params := env.system.default_params()) is not None:
        env.set_params(params)
    else:
        env.apply_model_defaults()

    return env
