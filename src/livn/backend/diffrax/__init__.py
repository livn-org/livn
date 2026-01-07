from typing import TYPE_CHECKING, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr

from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol
from livn.utils import P

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.types import Model


class _ParallelSystem:
    """System to simulate a number of neurons independently in parallel"""

    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        self.name = "ParallelSystem"
        self.populations = ["parallelized"]

    def default_io(self):
        return None


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
        if isinstance(system, int):
            system = _ParallelSystem(system)
        elif isinstance(system, str):
            from livn.system import CachedSystem

            system = CachedSystem(system, comm=comm)

        if model is None:
            from livn.models.rcsd import ReducedCalciumSomaDendrite

            model = ReducedCalciumSomaDendrite()

        self.system = system
        if io is None:
            io = self.system.default_io()
        self.model = model
        self.io = io
        self.comm = comm
        self.subworld_size = subworld_size

        self.encoding = None
        self.decoding = None

        self._noise = {}
        self._weights = None
        self.module = None
        self.seed = seed
        self.key = jr.PRNGKey(seed)
        self.key, self.init_key, self.run_key = jr.split(self.key, 3)

        self.t = 0.0
        self.v0 = None

    def init(self):
        self.module = self.model.diffrax_module(
            self,
            key=self.init_key,
        )
        return self

    def set_weights(self, weights):
        self._weights = weights
        return self

    def set_noise(self, noise: dict):
        # noise will be handled later during run
        self._noise = dict(noise)
        return self

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float = 0.1,
        **kwargs,
    ):
        if kwargs.get("root_only", True):
            if not P.is_root():
                raise NotImplementedError(
                    "The diffrax backend does not yet support MPI distributed solving."
                )

        stimulus = Stimulus.from_arg(stimulus)

        if stimulus.array is not None:
            stimulus.array = jnp.array(stimulus.array)

            # adjust timesteps if necessary
            if dt > stimulus.dt:
                raise ValueError("stimulus_dt can not be smaller than simulation dt")

            if int(stimulus.dt / dt) != stimulus.dt / dt:
                raise ValueError("stimulus_dt must be a multiple of dt")

            # converting the mV potential into nA current via Ohm's law,
            # assuming a membrane resistance of 400 MΩ
            stimulus.array = stimulus.array / 400

            expected_steps = int(duration / dt) + 1
            original_ndim = stimulus.array.ndim
            if original_ndim == 1:
                stimulus.array = stimulus.array[:, None]

            if stimulus.array.shape[0] != expected_steps or stimulus.dt != dt:
                time_src = jnp.arange(
                    stimulus.array.shape[0], dtype=stimulus.array.dtype
                )
                time_src = time_src * stimulus.dt
                time_target = jnp.linspace(
                    0.0, duration, expected_steps, dtype=stimulus.array.dtype
                )

                def _interp_column(column):
                    return jnp.interp(time_target, time_src, column)

                stimulus.array = jax.vmap(_interp_column, in_axes=1, out_axes=1)(
                    stimulus.array
                )

            current_steps = stimulus.array.shape[0]
            if current_steps < expected_steps:
                pad_rows = expected_steps - current_steps
                pad = jnp.zeros(
                    (pad_rows, stimulus.array.shape[1]), dtype=stimulus.array.dtype
                )
                stimulus.array = jnp.concatenate([stimulus.array, pad], axis=0)
            elif current_steps > expected_steps:
                stimulus.array = stimulus.array[:expected_steps]

            if original_ndim == 1:
                stimulus.array = stimulus.array[:, 0]

        it, tt, iv, v, im, mp = self.module.run(
            input_current=stimulus.array,
            noise=self._noise,
            t0=self.t,
            t1=self.t + duration,
            dt=dt,
            v0=self.v0,
            dt_solver=kwargs.get("dt_solver", 0.01),
            key=self.run_key,
            **kwargs,
        )

        self.t += duration
        self.v0 = v[:, -1]

        return it, tt, iv, v, im, mp

    def clear(self):
        self.t = 0
        self.v0 = None

        return self
