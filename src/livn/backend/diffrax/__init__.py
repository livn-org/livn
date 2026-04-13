from typing import TYPE_CHECKING, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol

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
            from livn.system import System

            system = System(system, comm=comm)
        self.system = system
        if model is None:
            model = system.default_model()
        self.model = model
        if io is None:
            io = self.system.default_io()
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
        if stimulus is not None:
            if not isinstance(stimulus, Stimulus):
                stimulus = Stimulus.from_arg(stimulus)
            stimulus = self.model.prepare_stimulus(stimulus)

        input_current = None
        if stimulus is not None:
            arr = stimulus.to_array(duration, dt)
            if arr is not None:
                input_current = jnp.array(arr)

        dt_solver = kwargs.pop("dt_solver", 0.01)
        t0 = kwargs.pop("t0", 0.0)
        y0 = kwargs.pop("y0", None)

        run_kwargs = dict(
            input_current=input_current,
            noise=self._noise,
            t0=t0,
            t1=t0 + duration,
            dt=dt,
            y0=y0,
            dt_solver=dt_solver,
            key=self.run_key,
            **kwargs,
        )

        it, tt, iv, v, im, mp, yT = self.module.run(**run_kwargs)

        return it, tt, iv, v, im, mp

    def clear_recordings(self):
        return self

    def clear(self):
        return self


def _env_tree_flatten(env):
    if env.module is not None:
        module_params, module_static = eqx.partition(env.module, eqx.is_array)
    else:
        module_params, module_static = None, None

    children = (module_params, env.key, env._noise)
    aux = (
        module_static,
        env.system,
        env._weights,
        env.model,
        env.io,
        env.comm,
        env.subworld_size,
        env.seed,
        env.init_key,
        env.run_key,
        env.encoding,
        env.decoding,
    )
    return children, aux


def _env_tree_unflatten(aux, children):
    module_params, key, noise = children
    (
        module_static,
        system,
        weights,
        model,
        io,
        comm,
        subworld_size,
        seed,
        init_key,
        run_key,
        encoding,
        decoding,
    ) = aux

    if module_params is not None and module_static is not None:
        module = eqx.combine(module_params, module_static)
    else:
        module = None

    env = Env(system, model, io, seed, comm, subworld_size)
    env.module = module
    env.key = key
    env._noise = noise
    env._weights = weights

    env.init_key = init_key
    env.run_key = run_key
    env.encoding = encoding
    env.decoding = decoding
    return env


jax.tree_util.register_pytree_node(Env, _env_tree_flatten, _env_tree_unflatten)
