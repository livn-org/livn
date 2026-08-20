"""livn"""

__doc__ = """A testbed for learning to interact with in vitro neural networks"""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()


def make(system: str = "EI"):
    """Initializes a default env from a predefined system, or from any URL"""
    from livn.env import Env
    from livn.system import PREDEFINED, fetch, predefined

    source = predefined(system) if system in PREDEFINED else fetch(system)

    return Env(source).init().apply_default_params()
