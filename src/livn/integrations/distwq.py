import os
import logging
from typing import Optional, Self

import distwq
from mpi4py import MPI

from livn.env import Env
from livn.types import Env as EnvProtocol, Model, Float, Array, Int
from livn.io import IO
from livn.utils import P

state = {}


def envcall(decoding, inputs, encoding, kwargs):
    if kwargs is None:
        kwargs = {}

    result = state["env"](decoding, inputs, encoding, **kwargs)

    return result


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
        self._result_buffer: dict[int, object] = {}

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

    def is_root(self):
        return P.is_root(os.getenv("DISTWQ_CONTROLLER_RANK", 0))

    def init(self):
        # MPI boot so that user can __call__ from controller
        args = (self,)

        logging.getLogger("distwq").setLevel(
            os.getenv("LIVN_DISTWQ_LOGGING", "WARNING")
        )
        verbose = False
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

        return self

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

    def clear_recordings(self) -> Self:
        self._broadcast_to_workers("clear_recordings", ())
        return self

    def apply_model_defaults(self, weights: bool = True, noise: bool = True) -> Self:
        self._broadcast_to_workers("apply_model_defaults", (weights, noise))
        return self

    def set_weights(self, weights: dict) -> Self:
        self._broadcast_to_workers("set_weights", (weights,))
        return self

    def set_noise(self, noise: dict) -> Self:
        self._broadcast_to_workers("set_noise", (noise,))
        return self

    def enable_plasticity(self, config: dict | None = None) -> Self:
        self._broadcast_to_workers("enable_plasticity", (config,))
        return self

    def disable_plasticity(self) -> Self:
        self._broadcast_to_workers("disable_plasticity", ())
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
    ) -> tuple[
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
        num_inputs = 0
        for i in inputs:
            self.submit_call(decoding, i, encoding, **kwargs)
            num_inputs += 1

        recordings = []
        for _ in range(num_inputs):
            response = self.receive_response()

            if len(response) == 1:
                # unpack root-response automatically
                recordings.append(response[0])
            else:
                recordings.append(response)

        return recordings

    def submit_call(self, decoding, inputs=None, encoding=None, **kwargs) -> int:
        return self.controller.submit_call(
            "envcall",
            (decoding, inputs, encoding, kwargs),
            module_name="livn.integrations.distwq",
        )

    def receive_response(self):
        call_id, response = self.controller.get_next_result()

        return response

    def probe_response(self, task_id: int):
        if self.controller is None:
            return None

        if task_id not in self._result_buffer:
            for tid, result in self.controller.probe_all_next_results():
                self._result_buffer[tid] = result

        if task_id not in self._result_buffer:
            return None

        raw = self._result_buffer.pop(task_id)
        return raw[0] if len(raw) == 1 else raw

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
