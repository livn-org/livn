from typing import Optional, Self, Tuple

import distwq

from mpi4py import MPI
from livn.env import Env
from livn.types import Env as EnvProtocol, Model, Float, Array, Int
from livn.io import IO

state = {}


def envcall(decoding, inputs, encoding, kwargs):
    if kwargs is None:
        kwargs = {}

    return state["env"](decoding, inputs, encoding, **kwargs)


def env_config_call(method_name, args):
    if state.get("env") is None:
        return None

    getattr(state["env"], method_name)(*args)


def worker_init(worker, distributed_env):
    if distributed_env.system is not None:
        worker_comm = getattr(worker, "merged_comm", worker.comm)

        env = Env(
            distributed_env.system,
            distributed_env.model,
            distributed_env.io,
            distributed_env.seed,
            comm=worker_comm,
            subworld_size=distributed_env.subworld_size,
        )
        env.init()

        state["env"] = env

    MPI.COMM_WORLD.Barrier()


def controller_init(controller, distributed_env):
    if distributed_env.system is not None:
        Env(
            distributed_env.system,
            distributed_env.model,
            distributed_env.io,
            distributed_env.seed,
            subworld_size=distributed_env.subworld_size,
        )
        state["env"] = None

    MPI.COMM_WORLD.Barrier()

    distributed_env.controller = controller

    controller.info()


class DistributedEnv(EnvProtocol):
    def __init__(
        self,
        system: str,
        model: Optional["Model"] = None,
        io: Optional["IO"] = None,
        seed: int | None = 123,
        comm: Optional["MPI.Intracomm"] = None,
        subworld_size: int | None = None,
    ):
        self.controller = None

        if not isinstance(system, str):
            raise ValueError(
                "System must be a directory to allow re-initialization on workers"
            )

        self.system = system
        self.model = model
        self.io = io
        self.seed = seed
        self.comm = comm
        self.subworld_size = subworld_size

    def init(self):
        # MPI boot so that user can __call__ from controller
        args = (self,)

        verbose = True
        if distwq.is_controller:
            distwq.run(
                fun_name="controller_init",
                args=args,
                module_name="livn.integrations.distwq",
                verbose=verbose,
                nprocs_per_worker=self.subworld_size,
                worker_grouping_method="split",
                broker_is_worker=True,
            )
        else:
            distwq.run(
                fun_name="worker_init",
                args=args,
                module_name="livn.integrations.distwq",
                verbose=verbose,
                nprocs_per_worker=self.subworld_size,
                worker_grouping_method="split",
                broker_is_worker=True,
            )

    def _broadcast_to_workers(self, method_name: str, args: tuple) -> None:
        if self.controller is None:
            return

        n_workers = self.controller.comm.size - 1
        task_ids = self.controller.submit_multiple(
            "env_config_call",
            args=[(method_name, args)] * n_workers,
            module_name="livn.integrations.distwq",
        )

        for _ in range(len(task_ids)):
            self.controller.get_next_result()

    def apply_model_defaults(self, weights: bool = True, noise: bool = True) -> Self:
        self._broadcast_to_workers("apply_model_defaults", (weights, noise))
        return self

    def set_weights(self, weights: dict) -> Self:
        self._broadcast_to_workers("set_weights", (weights,))
        return self

    def set_noise(self, exc: float = 1.0, inh: float = 1.0) -> Self:
        self._broadcast_to_workers("set_noise", (exc, inh))
        return self

    def record_spikes(self, population: str | list | tuple | None = None) -> Self:
        self._broadcast_to_workers("record_spikes", (population,))
        return self

    def record_voltage(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        self._broadcast_to_workers("record_voltage", (population, dt))
        return self

    def record_membrane_current(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        self._broadcast_to_workers("record_membrane_current", (population, dt))
        return self

    def cell_stimulus(
        self,
        channel_inputs: Float[Array, "batch timestep n_channels"],
    ) -> Float[Array, "batch timestep n_gids"]:
        raise NotImplementedError(
            "Please implement the cell stimulus as part of the encoding in __call__"
        )

    def channel_recording(
        self,
        ii: Float[Array, "i"],
        *recordings: Float[Array, "_"],
    ) -> tuple[dict[int, Array], ...]:
        raise NotImplementedError(
            "Please implement the channel recording as part of the decoding in __call__"
        )

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
        Float[Array, "neuron_ids voltages"] | None,
        Int[Array, "n_membrane_current_neuron_ids"] | None,
        Float[Array, "neuron_ids membrane_currents"] | None,
    ]:
        raise NotImplementedError("Please use __call__ instead")

    def __call__(self, decoding, inputs=None, encoding=None, **kwargs):
        if self.controller is None:
            return

        if inputs is None:
            inputs = [None]

        # evaluates the simulator using the MPI workers (only available on controller)
        for i in range(0, len(inputs)):
            self.submit_call(decoding, inputs[i], encoding, **kwargs)

        recordings = []
        for i in range(0, len(inputs)):
            response = self.receive_response()

            if len(response) == 1:
                # unpack root-response automatically
                recordings.append(response[0])
            else:
                recordings.append(response)

        return recordings

    def submit_call(self, decoding, inputs=None, encoding=None, **kwargs):
        self.controller.submit_call(
            "envcall",
            (decoding, inputs, encoding, kwargs),
            module_name="livn.integrations.distwq",
        )

    def receive_response(self):
        call_id, response = self.controller.get_next_result()

        return response

    def potential_recording(
        self, membrane_currents: Float[Array, "timestep n_neurons"] | None
    ) -> Float[Array, "timestep n_channels"]:
        if state["env"] is None:
            return
        state["env"].potential_recording(membrane_currents)

    def clear(self) -> Self:
        self._broadcast_to_workers("clear", ())
        return self

    def shutdown(self):
        if self.controller is not None:
            self.controller.exit()
