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
        self.i0 = None

    def init(self):
        self.module = self.model.diffrax_module(
            connectivity=self.system.connectivity_matrix(
                reduced=getattr(self.model, "reduced", False)
            ),
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

        num_samples = 1  # batch processing currently not supported
        if self.v0 is None:
            v0 = jnp.full((num_samples, self.module.num_neurons), 0)
        if self.i0 is None:
            i0 = jnp.full((num_samples, self.module.num_neurons), 0)

        w = jnp.asarray(
            self.system.connectivity_matrix(
                self._weights, reduced=getattr(self.model, "reduced", False)
            )
        )

        dt0 = kwargs.get("dt_solver", 0.01)
        sol, (ts, ys) = self.module.run(
            input_current=input_current,
            w=w,
            t0=self.t,
            t1=self.t + duration,
            # max_spikes=max_spikes,
            num_samples=num_samples,
            dt=dt,
            v0=v0,
            i0=i0,
            # num_save=num_save,
            key=self.run_key,
            dt0=dt0,
            max_steps=int(duration / dt0),
        )

        self.t += duration

        # save last value for each neuron to keep track of state
        # TODO: handle inf
        self.v0 = ys[0, :, -1, 0]
        self.i0 = ys[0, :, -1, 1]

        # voltages
        gids = jnp.arange(self.module.num_neurons)
        v = ys[0, :, :, 0]
        assert v.shape[1] == int(duration / dt)

        # spikes
        if kwargs.get("unroll", None) == "marcus":
            from livn.models.slif.snn import marcus_lift

            @jax.vmap
            def get_marcus_lifts(spike_times, spike_marks):
                return marcus_lift(self.t - duration, self.t, spike_times, spike_marks)

            spike_train = get_marcus_lifts(sol.spike_times, sol.spike_marks)

            return spike_train, None, gids, v

        diff_values = jnp.abs(jnp.diff(v, axis=1))
        mask = diff_values > 0.5
        mask = jnp.concatenate(
            [mask, jnp.zeros((mask.shape[0], 1), dtype=bool)], axis=1
        )

        if kwargs.get("unroll", None) == "mask":
            return mask, None, gids, v

        indices = jnp.nonzero(mask)
        it = indices[0]
        tt = indices[1] * dt

        return it, tt, gids, v

    def clear(self):
        self.t = 0
        self.i0 = None
        self.v0 = None

        return self
