"""livn"""

__doc__ = """A testbed for learning to interact with in vitro neural networks"""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()


def make(env: str):
    """Initialize a ready-to-run env.

    env = livn.make("EI")
    env = livn.make("runs/bursting/env.json")
    """
    from livn.env import Env
    from livn.system import predefined_document

    return Env.from_json(env if env.endswith(".json") else predefined_document(env))
