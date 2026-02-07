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
            from livn.system import CachedSystem

            system = CachedSystem(system, comm=comm)
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

        self.t = 0.0
        self.y0 = None

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
        stimulus = Stimulus.from_arg(stimulus)

        if stimulus.array is not None:
            stimulus.array = jnp.array(stimulus.array)

            # adjust timesteps if necessary
            if dt > stimulus.dt:
                raise ValueError("stimulus_dt can not be smaller than simulation dt")

            if int(stimulus.dt / dt) != stimulus.dt / dt:
                raise ValueError("stimulus_dt must be a multiple of dt")

            model_input_mode = getattr(self.module, "input_mode", "conductance")
            stimulus_input_mode = stimulus.meta_data.get("input_mode", "conductance")

            if model_input_mode != stimulus_input_mode:
                raise ValueError(
                    f"Stimulus input_mode '{stimulus_input_mode}' does not match "
                    f"model input_mode '{model_input_mode}'. "
                    f"Create stimulus with correct mode:\n"
                    f"  - Stimulus.from_conductance() for ±uS\n"
                    f"  - Stimulus.from_current_density() for mA/cm2\n"
                    f"  - Stimulus.from_current() for nA"
                )

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

        dt_solver = kwargs.pop("dt_solver", 0.01)
        it, tt, iv, v, im, mp, yT = self.module.run(
            input_current=stimulus.array,
            noise=self._noise,
            t0=self.t,
            t1=self.t + duration,
            dt=dt,
            y0=self.y0,
            dt_solver=dt_solver,
            key=self.run_key,
            **kwargs,
        )

        self.t += duration
        self.y0 = yT

        return it, tt, iv, v, im, mp

    def clear(self):
        self.t = 0
        self.y0 = None

        return self


def _env_tree_flatten(env):
    if env.module is not None:
        module_params, module_static = eqx.partition(env.module, eqx.is_array)
    else:
        module_params, module_static = None, None

    children = (module_params, env.key, env._noise, env.system)
    aux = (
        module_static,
        env.t,
        env.y0,
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
    module_params, key, noise, system = children
    (
        module_static,
        t,
        y0,
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
    env.y0 = y0
    env.t = t
    env.key = key
    env._noise = noise
    env._weights = weights

    env.init_key = init_key
    env.run_key = run_key
    env.encoding = encoding
    env.decoding = decoding
    return env


jax.tree_util.register_pytree_node(Env, _env_tree_flatten, _env_tree_unflatten)
