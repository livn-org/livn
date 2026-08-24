from __future__ import annotations

from typing import TYPE_CHECKING

from livn.cells import CellRegistry
from livn.run import Run
from livn.types import Env as EnvProtocol

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.stimulus import Stimulus
    from livn.system import System
    from livn.types import Model


class Env(EnvProtocol):
    def __init__(
        self,
        system: System | str | int,
        model: Model | None = None,
        io: IO = None,
        seed: int | None = 123,
        comm: MPI.Intracomm | None = None,
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
        self.cells = CellRegistry(self)

        self.t = 0

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float = 0.025,
        **kwargs,
    ) -> Run:
        print("No LIVN_BACKEND selected, returning None")

        return Run(t0=self.t, duration=duration)
