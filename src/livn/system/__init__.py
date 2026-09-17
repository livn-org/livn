from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy

from livn.system._common import Projection, projection_attribute
from livn.system.monolayer import Monolayer
from livn.system.neuroh5 import NeuroH5System
from livn.system.parallel import ParallelSystem
from livn.utils import P, download_directory

__all__ = [
    "Monolayer",
    "NeuroH5System",
    "ParallelSystem",
    "Projection",
    "fetch",
    "predefined",
    "predefined_document",
    "predefined_systems",
    "projection_attribute",
    "resolve",
]

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.types import System


def fetch(
    source: str,
    directory: str = ".",
    name: str | None = None,
    force: bool = False,
    comm: MPI.Intracomm | None = None,
) -> str:
    """Download a hosted system and return the target directory.

    ``source`` is a hosted system's name (``"CA1"``) or any URL fsspec can read.

    An existing directory is reused unless ``force``.
    """
    if "://" not in source:
        name, source = source, f"{HUB}/{source}"

    target = None
    comm = P.comm(comm)
    if comm is not None and comm.Get_rank() != 0:
        # await download on 0
        return comm.bcast(target)

    if name is None:
        import fsspec

        parsed = fsspec.utils.infer_storage_options(source).get("path", "")
        name = os.path.basename(parsed.rstrip("/"))
        if not name:
            raise ValueError("Could not infer system name from source")

    target = os.path.join(directory, "systems", "graphs", name)

    if force or not os.path.isdir(target):
        download_directory(source, target, force=force)

    if comm is not None:
        comm.bcast(target)

    return target


HUB = "hf://datasets/livn-org/livn/systems/graphs"

PREDEFINED_DIRECTORY = os.path.join(os.path.dirname(__file__), "predefined")


def predefined_systems() -> tuple[str, ...]:
    return tuple(
        sorted(
            name[: -len(".json")]
            for name in os.listdir(PREDEFINED_DIRECTORY)
            if name.endswith(".json")
        )
    )


def predefined_document(name: str = "EI") -> str:
    path = os.path.join(PREDEFINED_DIRECTORY, f"{name}.json")
    if os.path.isfile(path):
        return path

    raise ValueError(
        f"{name!r} not found; pick from {predefined_systems()}, "
        f"or fetch a graph using:\n"
        f"    from livn.system import fetch\n"
        f"    system = NeuroH5System(fetch({name!r}))"
    )


def predefined(name: str = "EI") -> Monolayer:
    import json

    with open(predefined_document(name)) as f:
        document = json.load(f)
    return Monolayer(**document["system"]["kwargs"])


def resolve(
    spec: System | str | int,
    comm: MPI.Intracomm | None = None,
) -> System:
    if isinstance(spec, bool):
        raise TypeError("system must be an int, a path or a System, not a bool")
    if isinstance(spec, (int, numpy.integer, Mapping)):
        return ParallelSystem(spec, comm=comm)
    if isinstance(spec, (str, os.PathLike)):
        path = os.fspath(spec)
        if path.endswith(".json"):
            from livn.env import Env
            from livn.types import _build

            document = Env.document(path)
            system = _build(document.get("system", document))
            if getattr(system, "uri", None) is None:
                with contextlib.suppress(AttributeError):
                    system.uri = path
            return system
        return NeuroH5System(path, comm=comm)
    if not hasattr(spec, "populations"):
        raise TypeError(
            f"cannot resolve {type(spec).__name__} into a system; expected a number "
            "of neurons, a system directory, or an object implementing the System "
            "protocol"
        )
    return spec
