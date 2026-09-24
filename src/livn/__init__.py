"""livn"""

__doc__ = """A testbed for learning to interact with in vitro neural networks"""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()


def make(source="EI", *, cls=None, **kwargs):
    """Build a ready-to-run env:

        env = livn.make("EI")                       # a predefined culture
        env = livn.make("runs/bursting/env.json")   # a promoted document
        env = livn.make(document)                   # the same, already loaded
        env = livn.make(spec, comm=comm)            # a bare system spec
        env = livn.make(2600)                       # cells, no graph

    Args:
        source: A predefined name, a path to an `env.json`, a loaded document,
            a described system (`{"cls": ..., "kwargs": ...}`), a `System`, a
            `{population: count}` mapping, or a number of cells.
        cls: The env class, e.g. `DistributedEnv`; `livn.env.Env` by default.
    """
    from collections.abc import Mapping

    from livn.env import Env
    from livn.system import predefined_document

    def as_document(source) -> dict:
        if isinstance(source, str):
            if not source.endswith(".json"):
                source = predefined_document(source)

            import json as _json

            try:
                with open(source) as _f:
                    loaded = _json.load(_f)
            except (OSError, ValueError):
                return {"system": source}
            if isinstance(loaded, Mapping) and "system" in loaded:
                return {
                    **{k: v for k, v in loaded.items() if k != "system"},
                    "system": source,
                }
            return {"system": source}

        if isinstance(source, Mapping):
            if "system" in source and "cls" not in source:
                return dict(source)
            return {"system": dict(source)}

        if hasattr(source, "serialize") and hasattr(source, "populations"):
            from livn.types import _describe

            return {"system": _describe(source)}

        return {"system": source}

    return (cls or Env).from_json(as_document(source), **kwargs)
