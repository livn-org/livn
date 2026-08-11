import copy
import dataclasses
import functools
from typing import TYPE_CHECKING, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np

from livn.cells import CellRegistry
from livn.run import Run
from livn.stimulus import Stimulus
from livn.types import Cell
from livn.types import Env as EnvProtocol

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.types import Model


def cell_param_paths(module, n_cells: int) -> tuple[str, ...]:
    """Names of the module fields that hold one value per cell.

    A field qualifies when it is a non-static array of inexact dtype whose
    leading axis is the number of cells; nested :class:`equinox.Module` fields
    are reached by a dotted path. A module can declare the paths itself by
    exposing ``cell_param_names``, which takes precedence.
    """
    declared = getattr(module, "cell_param_names", None)
    if declared is not None:
        return tuple(str(name) for name in declared)

    if not isinstance(module, eqx.Module):
        return ()

    paths: list[str] = []
    for field in dataclasses.fields(module):
        if field.metadata.get("static", False):
            continue
        value = getattr(module, field.name, None)
        if isinstance(value, eqx.Module):
            paths.extend(
                f"{field.name}.{path}" for path in cell_param_paths(value, n_cells)
            )
            continue
        if not eqx.is_array(value) or value.ndim < 1:
            continue
        if value.shape[0] != n_cells:
            continue
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            continue
        paths.append(field.name)

    return tuple(sorted(paths))


def _resolve(module, path: str):
    value = module
    for part in path.split("."):
        value = getattr(value, part)
    return value


def cell_params(module, n_cells: int) -> dict[str, jnp.ndarray]:
    """Per-cell parameter arrays of a diffrax module"""
    return {path: _resolve(module, path) for path in cell_param_paths(module, n_cells)}


def replace_cell_params(module, params: dict, n_cells: int):
    """Return a copy of the module with its per-cell parameters replaced"""
    current = cell_params(module, n_cells)

    replace = []
    paths = []
    for path, value in params.items():
        if path not in current:
            raise KeyError(
                f"{type(module).__name__} has no {path!r} cell parameter "
                f"(available: {sorted(current)})"
            )
        array = current[path]
        paths.append(path)
        replace.append(
            jnp.broadcast_to(jnp.asarray(value, dtype=array.dtype), array.shape)
        )

    if not paths:
        return module

    return eqx.tree_at(lambda m: [_resolve(m, p) for p in paths], module, replace)


class CellHandle(Cell):
    """Per-cell parameter handle.

    env = env.cells[3].set_params({"tau_m": 12.0})
    """

    def __init__(self, env: "Env", population: str, gid: int, index: int):
        super().__init__(env, population, gid)
        self._index = int(index)

    @property
    def index(self) -> int:
        return self._index

    def get_params(self) -> dict:
        n_cells = self._env.num_cells
        return {
            path: array[self._index]
            for path, array in cell_params(self._env.module, n_cells).items()
        }

    def set_params(self, params: dict) -> "Env":
        env = self._env
        n_cells = env.num_cells
        current = cell_params(env.module, n_cells)

        updates = {}
        for path, value in params.items():
            if path not in current:
                raise self.unknown_param(path, current)
            array = current[path]
            updates[path] = array.at[self._index].set(
                jnp.asarray(value, dtype=array.dtype)
            )

        return env._with_module(replace_cell_params(env.module, updates, n_cells))


class ModuleCellRegistry(CellRegistry):
    @property
    def gids(self) -> np.ndarray:
        return np.asarray(self._env.active_gids(), dtype=np.int64)

    def get_params(self) -> dict:
        env = self._env
        if env.module is None:
            return {}
        return cell_params(env.module, env.num_cells)

    def set_params(self, params: dict) -> "Env":
        env = self._env
        if env.module is None:
            raise RuntimeError("call init() before setting cell parameters")
        return env._with_module(replace_cell_params(env.module, params, env.num_cells))


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

        self._noise = {}
        self._weights = None
        self._recording = {}
        self.module = None
        self.seed = seed
        self.key = jr.PRNGKey(seed)
        self.key, self.init_key, self.run_key = jr.split(self.key, 3)

    def init(self):
        self.module = self.model.diffrax_module(
            self,
            key=self.init_key,
        )
        self.__dict__.pop("_cells", None)
        return self

    @property
    def num_cells(self) -> int:
        """Number of cells the module simulates"""
        return int(len(self.active_gids()))

    @property
    def cells(self) -> CellRegistry:
        registry = self.__dict__.get("_cells")
        if registry is None:
            registry = ModuleCellRegistry(self)
            index = 0
            for population in self.active_populations():
                coordinates = np.asarray(
                    self.system.coordinate_array(population, all=False)
                )
                gids = coordinates[:, 0].astype(np.int64)
                registry.add(
                    population,
                    {
                        int(gid): CellHandle(self, population, int(gid), index + offset)
                        for offset, gid in enumerate(gids)
                    },
                )
                index += len(gids)
            self.__dict__["_cells"] = registry
        return registry

    def _with_module(self, module) -> "Env":
        env = copy.copy(self)
        env.module = module
        env._recording = {name: dict(o) for name, o in self._recording.items()}
        env.__dict__.pop("_cells", None)
        return env

    def set_weights(self, weights):
        self._weights = weights
        return self

    def set_noise(self, noise: dict):
        # noise will be handled later during run
        self._noise = dict(noise)
        return self

    def _enable(self, signal: str, **options) -> "Env":
        self._recording[signal] = dict(options)
        return self

    def _record_spikes(self, population: str) -> "Env":
        return self._enable("spikes")

    def _record_voltage(self, population: str, dt: float = 0.1) -> "Env":
        return self._enable("voltage", dt=dt)

    def _record_membrane_current(self, population: str, dt: float = 0.1) -> "Env":
        return self._enable("membrane_current", dt=dt)

    def _enable_state(self, signal: str, population: str | None = None) -> "Env":
        return self._enable(signal)

    def _model_states(self) -> tuple[str, ...]:
        model = self.__dict__.get("model")
        return () if model is None else tuple(model.recordable_states())

    def __getattr__(self, name: str):
        if name.startswith("_record_"):
            signal = name[len("_record_") :]
            if signal in self._model_states():
                return functools.partial(self._enable_state, signal)

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __dir__(self):
        states = (f"_record_{signal}" for signal in self._model_states())
        return sorted({*super().__dir__(), *states})

    def recording(self) -> dict:
        return dict(self._recording)

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
            input_current = jnp.array(arr)

        dt_solver = kwargs.pop("dt_solver", 0.01)
        t0 = kwargs.pop("t0", 0.0)
        y0 = kwargs.pop("y0", None)
        key = kwargs.pop("key", self.run_key)

        record = self.recording()

        it, tt, iv, v, im, mp, yT, states = self.module.run(
            input_current=input_current,
            noise=self._noise,
            t0=t0,
            t1=t0 + duration,
            dt=dt,
            y0=y0,
            dt_solver=dt_solver,
            key=key,
            record=frozenset(record),
            **kwargs,
        )

        run = Run(t0=t0, duration=duration)
        if "spikes" in record:
            run = run.add_spikes(
                it, tt, padded=getattr(self.module, "padded_spikes", False)
            )
        if "voltage" in record:
            run = run.add_voltage(iv, v, dt=dt)
        if "membrane_current" in record:
            run = run.add_current(im, mp, dt=dt)

        for name, (ids, values) in states.items():
            run = run.add(name, ids, values, dt=dt, kind="series")

        return run

    def clear_recordings(self):
        return self

    def clear(self):
        return self


def _env_tree_flatten(env):
    if env.module is not None:
        module_params, module_static = eqx.partition(env.module, eqx.is_array)
    else:
        module_params, module_static = None, None

    # If system or io is a registered JAX pytree, put it in children so its arrays
    # are traced/differentiated
    flat_system = jax.tree_util.tree_leaves(env.system)
    system_is_pytree = not (len(flat_system) == 1 and flat_system[0] is env.system)
    flat_io = jax.tree_util.tree_leaves(env.io)
    io_is_pytree = not (len(flat_io) == 1 and flat_io[0] is env.io)

    children = (
        module_params,
        env.key,
        env._noise,
        env.system if system_is_pytree else None,
        env.io if io_is_pytree else None,
    )
    aux = (
        module_static,
        None if system_is_pytree else env.system,
        system_is_pytree,
        env._weights,
        env.model,
        None if io_is_pytree else env.io,
        io_is_pytree,
        env.comm,
        env.subworld_size,
        env.seed,
        env.init_key,
        env.run_key,
        env.encoding,
        env.decoding,
        tuple(
            sorted(
                (name, tuple(sorted(o.items()))) for name, o in env._recording.items()
            )
        ),
    )
    return children, aux


def _env_tree_unflatten(aux, children):
    module_params, key, noise, system_child, io_child = children
    (
        module_static,
        system_aux,
        system_is_trainable,
        weights,
        model,
        io_aux,
        io_is_pytree,
        comm,
        subworld_size,
        seed,
        init_key,
        run_key,
        encoding,
        decoding,
        recording,
    ) = aux

    system = system_child if system_is_trainable else system_aux
    io = io_child if io_is_pytree else io_aux

    if module_params is not None and module_static is not None:
        module = eqx.combine(module_params, module_static)
    else:
        module = None

    env = Env(system, model, io, seed, comm, subworld_size)
    env.module = module
    env.key = key
    env._noise = noise
    env._weights = weights
    env._recording = {name: dict(options) for name, options in recording}

    env.init_key = init_key
    env.run_key = run_key
    env.encoding = encoding
    env.decoding = decoding
    return env


jax.tree_util.register_pytree_node(Env, _env_tree_flatten, _env_tree_unflatten)
