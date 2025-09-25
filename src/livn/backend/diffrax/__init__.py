from typing import TYPE_CHECKING, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol
from livn.utils import P

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.types import Model


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
        from livn.system import CachedSystem

        if model is None:
            from livn.models.slif import SLIF

            model = SLIF()

        self.system = (
            system if not isinstance(system, str) else CachedSystem(system, comm=comm)
        )
        if io is None:
            io = self.system.default_io()
        self.model = model
        self.io = io
        self.comm = comm
        self.subworld_size = subworld_size

        self._noise = {"exc": 0.0, "inh": 0.0}
        self._weights = None
        self.module = None
        self.seed = seed
        self.key = jr.PRNGKey(seed)
        self.key, self.init_key, self.run_key = jr.split(self.key, 3)

        self.t = 0.0
        self.v0 = None

    def init(self):
        self.module = self.model.diffrax_module(
            self.system,
            key=self.init_key,
        )
        return self

    def set_weights(self, weights):
        self._weights = weights
        return self

    def set_noise(self, exc: float = 1.0, inh: float = 1.0):
        # noise will be handled later during run
        self._noise = dict(exc=exc, inh=inh)
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
            stimulus_array = jnp.array(stimulus.array)

            # adjust timesteps if necessary
            if dt > stimulus.dt:
                raise ValueError("stimulus_dt can not be smaller than simulation dt")

            if int(stimulus.dt / dt) != stimulus.dt / dt:
                raise ValueError("stimulus_dt must be a multiple of dt")

            # converting the mV potential into nA current via Ohm's law,
            # assuming a membrane resistance of 400 MΩ
            stimulus_array = stimulus_array / 400

            def input_current(t):
                idx = jnp.minimum(
                    jnp.floor(t / stimulus.dt).astype(jnp.int32),
                    stimulus_array.shape[0] - 1,
                )
                return jnp.asarray(stimulus_array)[idx]

        else:

            def input_current(t):
                return jnp.zeros([self.system.cells_meta_data.cell_count()])

        it, tt, iv, v = self.module.run(
            input_current=input_current,
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

        return it, tt, iv, v

    def clear(self):
        self.t = 0
        self.i0 = None
        self.v0 = None

        return self
