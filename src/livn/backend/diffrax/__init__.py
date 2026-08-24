import copy
import dataclasses
import functools
from typing import TYPE_CHECKING, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from livn.cells import CellRegistry
from livn.run import Run
from livn.stimulus import Stimulus
from livn.types import Capability, Cell
from livn.types import Env as EnvProtocol
from livn.utils import lnp

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
        return np.asarray(self._env.module_gids, dtype=np.int64)

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
    capabilities = frozenset(
        {
            Capability.SIMULATION,
            Capability.DIFFERENTIABLE,
            Capability.IMMUTABLE,
        }
    )

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

        self._select_spec = None
        self._select_method = "first"
        self._select_bounds = None
        self._selection: dict[str, object] | None = None
        self._selected_gids: set[int] | None = None
        self._selected_rows: dict[str, np.ndarray] | None = None
        self._module_gids: np.ndarray | None = None

        self._noise = {}
        self._weights = None
        self._recording = {}
        self.module = None
        self.seed = seed
        self.key = jr.PRNGKey(seed)
        self.key, self.init_key, self.run_key = jr.split(self.key, 3)

    def selection(self, select, method: str = "first", bounds=None) -> "Env":
        if self.module is not None:
            raise RuntimeError("selection() must be called before init()")

        self._select_spec = select
        self._select_method = method
        self._select_bounds = bounds
        self._resolve_selection()
        self.__dict__.pop("_cells", None)
        return self

    @property
    def selection_name(self) -> str | None:
        spec = self._select_spec
        return spec if isinstance(spec, str) else None

    def _resolve_selection(self) -> None:
        self._selection = self.system.selection(
            self._select_spec,
            populations=self.active_populations(),
            seed=self.seed,
            method=self._select_method,
            bounds=self._select_bounds,
        )
        if self._selection is None:
            self._selected_gids = None
        else:
            self._selected_gids = {
                int(g) for gids in self._selection.values() for g in gids
            }
        self._materialize_geometry()

    def _materialize_geometry(self) -> None:
        rows: dict[str, np.ndarray] = {}
        gids: list[np.ndarray] = []

        selected = (
            None
            if self._selected_gids is None
            else np.fromiter(sorted(self._selected_gids), dtype=np.int64)
        )

        for population in self.active_populations():
            population_gids = np.asarray(
                self.system.coordinate_array(population)[:, 0]
            ).astype(np.int64)
            if selected is None:
                keep = np.arange(len(population_gids), dtype=np.int64)
            else:
                keep = np.flatnonzero(np.isin(population_gids, selected))
            if len(keep):
                rows[population] = keep
                gids.append(population_gids[keep])

        self._selected_rows = rows
        self._module_gids = (
            np.concatenate(gids) if gids else np.zeros(0, dtype=np.int64)
        )

    def _selected_coordinates(self):
        if self._selected_rows is None:
            self._materialize_geometry()

        for population, keep in self._selected_rows.items():
            yield population, self.system.coordinate_array(population)[keep]

    @property
    def module_gids(self) -> np.ndarray:
        """Gids the module simulates, in module-index order."""
        if self._module_gids is None:
            self._materialize_geometry()
        return self._module_gids

    def simulated_gids(self, everywhere: bool = False):
        return self.module_gids

    def stimulus_coordinates(self):
        rows = [
            self.model.stimulus_coordinates(coordinates, population=population)
            for population, coordinates in self._selected_coordinates()
        ]
        return lnp().vstack(rows) if rows else np.zeros((0, 4))

    def init(self):
        if self._select_spec is not None or self._select_bounds is not None:
            self._resolve_selection()
        else:
            self._materialize_geometry()
        self.module = self.model.diffrax_module(
            self,
            key=self.init_key,
        )
        self.__dict__.pop("_cells", None)
        return self

    @property
    def num_cells(self) -> int:
        """Number of cells the module simulates"""
        return len(self.module_gids)

    @property
    def cells(self) -> CellRegistry:
        registry = self.__dict__.get("_cells")
        if registry is None:
            registry = ModuleCellRegistry(self)
            if self._selected_rows is None:
                self._materialize_geometry()

            module_gids = self.module_gids
            index = 0
            for population, keep in self._selected_rows.items():
                gids = module_gids[index : index + len(keep)]
                registry.add(
                    population,
                    {
                        int(gid): CellHandle(self, population, int(gid), index + offset)
                        for offset, gid in enumerate(gids)
                    },
                )
                index += len(keep)
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

    def _record_voltage(
        self, population: str, dt: float = 0.1, gids=None, sections=None
    ) -> "Env":
        if sections is not None and "soma" not in {str(s) for s in sections}:
            raise NotImplementedError(
                f"{sections!r}: diffrax cells are single compartment recorded as 'soma'"
            )
        if gids is not None:
            raise NotImplementedError(
                "the diffrax backend records every cell's voltage as one dense "
                "array, so it cannot narrow the recording to particular gids. "
                "Select the rows you want from the result instead"
            )
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
            stimulus = Stimulus.from_arg(stimulus, env=self, duration=duration)
            stimulus = self.model.prepare_stimulus(stimulus)
            if stimulus.gids is not None:
                from livn.io import section_labels

                stimulus = stimulus.expand(*section_labels(self.stimulus_coordinates()))

        input_current = None
        if stimulus is not None:
            arr = stimulus.to_array(duration, dt)
            input_current = jnp.array(arr)

        dt_solver = kwargs.pop("dt_solver", 0.01)
        t0 = kwargs.pop("t0", 0.0)
        y0 = kwargs.pop("y0", None)
        key = kwargs.pop("key", self.run_key)

        record = self.recording()

        it, tt, iv, v, im, mp, _yT, states = self.module.run(
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

        as_gid = self._module_index_to_gid

        run = Run(t0=t0, duration=duration)
        if "spikes" in record:
            run = run.add_spikes(
                as_gid(it), tt, padded=getattr(self.module, "padded_spikes", False)
            )
        if "voltage" in record:
            run = run.add_voltage(as_gid(iv), v, dt=dt)
        if "membrane_current" in record:
            run = run.add_current(as_gid(im), mp, dt=dt)

        for name, (ids, values) in states.items():
            run = run.add(name, as_gid(ids), values, dt=dt, kind="series")

        return run

    def _module_index_to_gid(self, indices):
        if indices is None:
            return None
        gids = self.module_gids
        if len(gids) == 0:
            return indices
        if np.array_equal(gids, np.arange(len(gids), dtype=gids.dtype)):
            return indices
        return jnp.asarray(gids)[jnp.asarray(indices).astype(int)]

    def clear_recordings(self):
        return self

    def clear(self, reseed: bool = True):
        if reseed:
            return self.reseed_noise()
        return self

    def reseed_noise(self, stream: int | None = None):
        env = copy.copy(self)
        if stream is None:
            env.key, env.run_key = jr.split(self.key)
        else:
            base = jr.PRNGKey(self.seed if self.seed is not None else 0)
            env.run_key = jr.fold_in(base, int(stream))
        return env


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
        None
        if env._selected_gids is None
        else tuple(sorted(int(g) for g in env._selected_gids)),
        None
        if env._selected_rows is None
        else tuple(
            (name, tuple(int(r) for r in rows))
            for name, rows in env._selected_rows.items()
        ),
        None if env._module_gids is None else tuple(int(g) for g in env._module_gids),
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
        selected_gids,
        selected_rows,
        module_gids,
        recording,
    ) = aux

    system = system_child if system_is_trainable else system_aux
    io = io_child if io_is_pytree else io_aux

    if module_params is not None and module_static is not None:
        module = eqx.combine(module_params, module_static)
    else:
        module = None

    env = Env(system, model, io, seed, comm, subworld_size)
    env._selected_gids = None if selected_gids is None else set(selected_gids)
    env._selected_rows = (
        None
        if selected_rows is None
        else {name: np.asarray(rows, dtype=np.int64) for name, rows in selected_rows}
    )
    env._module_gids = (
        None if module_gids is None else np.asarray(module_gids, dtype=np.int64)
    )
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
