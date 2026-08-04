from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from livn.run import Run
from livn.types import Env as EnvProtocol


if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.stimulus import Stimulus
    from livn.types import Model


class Env(EnvProtocol):
    def __init__(
        self,
        system: Union["System", str, int],
        model: Union["Model", None] = None,
        io: Union["IO"] = None,
        seed: int | None = 123,
        comm: Optional["MPI.Intracomm"] = None,
        subworld_size: int | None = None,
    ):
        from livn.system import resolve

        self.system = resolve(system, comm=comm)
        if model is None:
            model = self.system.default_model()
        self.model = model
        if io is None:
            io = self.system.default_io()
        self.io = io

        self.comm = comm
        self.subworld_size = subworld_size

        self.encoding = None
        self.decoding = None

        self.t = 0

    def run(
        self,
        duration,
        stimulus: Optional["Stimulus"] = None,
        dt: float = 0.025,
        **kwargs,
    ) -> Run:
        print("No LIVN_BACKEND selected, returning None")

        return Run(t0=self.t, duration=duration)
