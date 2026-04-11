from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple

from livn.types import Env as EnvProtocol


if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.stimulus import Stimulus
    from livn.types import Model, Int, Float, Array


class Env(EnvProtocol):
    def __init__(
        self,
        system: Union["System", str],
        model: Union["Model", None] = None,
        io: Union["IO"] = None,
        seed: int | None = 123,
        comm: Optional["MPI.Intracomm"] = None,
        subworld_size: int | None = None,
    ):
        if isinstance(system, str):
            from livn.system import System

            system = System(system, comm=comm)
        self.system = system
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
    ) -> Tuple[
        Int[Array, "n_spiking_neuron_ids"] | None,
        Float[Array, "n_spiking_neuron_times"] | None,
        Int[Array, "n_voltage_neuron_ids"] | None,
        Float[Array, "n_neurons timestep"] | None,
        Int[Array, "n_membrane_current_neuron_ids"] | None,
        Float[Array, "n_neurons timestep"] | None,
    ]:
        print("No LIVN_BACKEND selected, returning None")

        return None, None, None, None, None, None
