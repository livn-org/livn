from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Self,
    Tuple,
    Union,
    runtime_checkable,
)
import hashlib
import pickle

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    import gymnasium
    from jaxtyping import Array as JaxArray
    from mpi4py import MPI
    from numpy import ndarray
    from tensorflow import TfTensor
    from torch import TorchTensor

    from livn.cells import CellRegistry
    from livn.io import IO
    from livn.run import Run
    from livn.stimulus import Stimulus
    from livn.system import Projection
    from livn.types import Model

    Array = Union[TorchTensor, ndarray, JaxArray, TfTensor]

    from jaxtyping import Float, Int

PopulationName = str
PostSynapticPopulationName = PopulationName
PreSynapticPopulationName = PopulationName

# list | dict | tuple | Stimulus | Float[Array, "batch timestep n_channels"] | None
StimulusLike = Any


class SynapticParam(BaseModel):
    population: Optional[str] = None
    source: Optional[str] = None
    sec_type: Optional[str] = None
    syn_name: Optional[str] = None
    param_path: Optional[Union[str, Tuple[str, ...]]] = None
    param_range: Optional[str] = None
    phenotype: Optional[str] = None

    @field_validator("param_path")
    @classmethod
    def parse_path(
        cls, v: Optional[Union[str, Tuple[str, ...]]]
    ) -> Optional[Tuple[str, ...]]:
        if v is None:
            return None
        if isinstance(v, tuple):
            return v
        if isinstance(v, str):
            if "/" not in v:
                return v

            return tuple(v.split("/"))
        raise ValueError(f"Invalid param_path type: {type(v)}")

    @classmethod
    def from_string(cls, string: str) -> "SynapticParam":
        """`population_source-sec_type-syn_name-param_path-param_range-phenotype`

        The source may be omitted for parameters that are not per-connection.
        """
        try:
            has_source = "_" in string.split("-", 1)[0]
            if has_source:
                population, rest = string.split("_", 1)
                parts = rest.split("-")
                source = parts[0]
                parts = parts[1:]
            else:
                parts = string.split("-")
                population, source = parts[0], None
                parts = parts[1:]

            data = {"population": population}

            data["source"] = source

            optional_fields = [
                "sec_type",
                "syn_name",
                "param_path",
                "param_range",
                "phenotype",
            ]
            for i, field in enumerate(optional_fields):
                data[field] = parts[i] if len(parts) > i else None

            return cls(**data)

        except Exception as e:
            raise ValueError(f"Failed to parse string '{string}': {str(e)}")


@runtime_checkable
class System(Protocol):
    """Protocol defining the interface for livn systems."""

    name: str
    """Human-readable identifier that keys the model's per-system defaults"""

    num_neurons: int
    """Total number of cells across all populations"""

    populations: list[PopulationName]
    """Population names"""

    gids: Int[Array, "n_neurons"]
    """Global cell ids across all populations"""

    population_ranges: dict[PopulationName, Tuple[int, int]]
    """``{population: (start_gid, count)}`` for every population"""

    connections_config: dict
    """``{"synapses": {post: {pre: config}}}`` or empty when unconnected"""

    neuron_coordinates: Float[Array, "n_neurons ixyz=4"]
    """``[gid, x, y, z]`` rows for every cell"""

    def population_count(self, population: PopulationName) -> int:
        """Number of cells in one population"""
        ...

    def synapse_projections(self) -> list[tuple[str, str, str, str, str]]:
        """``(post, pre, section, mechanism, type)`` per synapse the graph declares"""
        ...

    def default_io(self, comm: Optional["MPI.Intracomm"] = None) -> "IO":
        """IO device to use when the environment is constructed without one"""
        ...

    def default_model(self, comm: Optional["MPI.Intracomm"] = None) -> "Model":
        """Model to use when the environment is constructed without one"""
        ...

    def coordinate_array(
        self, population: PopulationName
    ) -> Float[Array, "n_coords ixyz=4"]:
        """``[gid, x, y, z]`` rows for one population across every rank.

        For the cells a rank simulates, see `Env.simulated_coordinates`.
        """
        ...

    def transform_coordinates(
        self,
        transform: Any,
        populations: list[PopulationName] | None = None,
    ) -> Float[Array, "n_coords ixyz=4"]:
        """Apply a model coordinate transform per population and stack the result"""
        ...

    def projection_array(
        self,
        pre: PreSynapticPopulationName,
        post: PostSynapticPopulationName,
        all: bool = True,
    ) -> list[tuple[int, tuple[list[int], "Projection"]]]:
        """Edges onto ``post`` from ``pre`` as ``(post_gid, (pre_gids, projection))``"""
        ...

    def connectivity_matrix(
        self, weights: dict | None = None, seed: int = 123
    ) -> Float[Array, "num_neurons num_neurons"]:
        """Dense signed weight matrix or all-zero when unconnected"""
        ...

    def selection(
        self,
        spec,
        populations: list[PopulationName] | None = None,
        seed: int | None = 123,
        method: str = "first",
        bounds=None,
    ) -> dict[PopulationName, Any] | None:
        """Resolve a cell subselection into ``{population: gids}`` (``None`` for all)"""
        ...


@runtime_checkable
class Cell(Protocol):
    """Protocol defining the interface for a single simulated cell and its physical parameters."""

    def __init__(self, env: "Env", population: PopulationName, gid: int):
        self._env = env
        self._population = str(population)
        self._gid = int(gid)

    @property
    def env(self) -> "Env":
        return self._env

    @property
    def gid(self) -> int:
        """Global id of the cell"""
        return self._gid

    @property
    def population(self) -> PopulationName:
        """Population the cell belongs to"""
        return self._population

    def get_params(self) -> dict[str, float]:
        """Physical parameters of this cell."""
        ...

    def set_params(self, params: dict[str, float]) -> "Env":
        """Set physical parameters of this cell."""
        ...

    def unknown_param(self, name: str, available) -> KeyError:
        """The error to raise for a parameter this cell does not have"""
        return KeyError(
            f"cell {self._gid} has no {name!r} parameter "
            f"(available: {sorted(available)})"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._population}, gid={self._gid})"


@runtime_checkable
class Env(Protocol):
    """Protocol defining the interface for livn environments"""

    cells: "CellRegistry"
    """The simulated cells, addressable by population name or gid"""

    def __init__(
        self,
        system: Union["System", str, int],
        model: "Model",
        io: "IO",
        seed: int | None = 123,
        comm: Optional["MPI.Intracomm"] = None,
        subworld_size: int | None = None,
    ): ...

    def apply_model_defaults(self, weights: bool = True, noise: bool = True) -> Self:
        self.model.apply_defaults(self, weights=weights, noise=noise)

        return self

    def apply_default_params(self, group: str | None = None, strict: bool = False):
        import warnings

        system = getattr(self, "system", None)
        selection_name = getattr(self, "selection_name", None)

        source, document = (None, None)
        if system is not None and hasattr(system, "params_document"):
            source, document = system.params_document(
                selection_name, comm=getattr(self, "comm", None)
            )

        if document is None:
            return self.apply_model_defaults()

        key = self.model.params_key()
        if key not in document:
            return self.apply_model_defaults()

        groups = document[key]
        name = group or "default"
        if name not in groups:
            raise KeyError(
                f"{source} has no parameter group {name!r} for "
                f"{key}; available: {', '.join(sorted(groups)) or 'none'}"
            )
        entry = groups[name]
        params = entry.get("params", entry) if isinstance(entry, dict) else entry

        if hasattr(self, "admissible_params"):
            admissible = self.admissible_params()
            known = set().union(
                *(
                    set(admissible.get(g) or ())
                    for g in ("weights", "mechanisms", "noise")
                )
            )
            unknown = sorted(
                k for k in params if k not in known and not k.startswith("cells-")
            )
            if unknown:
                complaint = (
                    f"{source} names {len(unknown)} parameter(s) this "
                    f"network has nothing for, which `set_params` drops "
                    f"silently: {unknown}"
                )
                if strict:
                    raise RuntimeError(complaint)
                warnings.warn(complaint, stacklevel=2)

        return self.set_params(dict(params))

    def cell_stimulus(
        self,
        channel_inputs: Float[Array, "batch timestep n_channels"],
    ) -> "Stimulus":
        """Transforms channel inputs into neural inputs."""
        from livn.stimulus import Stimulus

        coordinates = self.system.transform_coordinates(
            self.model.stimulus_coordinates,
            populations=self.active_populations(),
        )
        array = self.io.cell_stimulus(coordinates, channel_inputs)
        return Stimulus(array, gids=coordinates[:, 0].astype(int))

    def channel_recording(
        self,
        ii: Float[Array, "i"] | None,
        *recordings: Float[Array, "_"],
    ) -> tuple[dict[int, Array], ...]:
        """Transforms neural recordings identified by their gids into per channel recordings"""
        return self.io.channel_recording(
            self.active_neuron_coordinates(), ii, *recordings
        )

    def init(self) -> Self:
        """Initialize the environment."""
        ...
        return self

    def selection(self, select, method: str = "first", bounds=None) -> Self:
        """Restrict which cells are instantiated before ``init()``.

        ``select`` may be an int (total cell count, allocated across populations
        in proportion to their size), a float (fraction of each population), or a
        dict mapping population names to a count, fraction, or explicit gid list.
        ``method`` is ``"first"`` (contiguous gid block), ``"random"``, or
        ``"patch"`` (a centred planar region, optionally given by ``bounds``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support cell subselection"
        )

    def destination_sections(self) -> dict[str, dict[str, str]]:
        return {}

    @property
    def weight_names(self) -> list[str]:
        return self.system.weight_names

    def set_weights(self, weights: dict) -> Self:
        """Set the synaptic weights"""
        ...
        return self

    def set_noise(self, noise: dict) -> Self:
        """Set noise"""
        ...
        return self

    def enable_plasticity(self, config: dict | None = None) -> Self:
        """Enable plasticity"""
        ...
        return self

    def disable_plasticity(self) -> Self:
        """Freeze synaptic weights"""
        ...
        return self

    def get_weights(self) -> dict:
        """Return current synaptic weights of all plastic synapses"""
        ...

    def normalize_weights(self, target: float | None = None) -> Self:
        """Normalize incoming excitatory weights per neuron"""
        ...
        return self

    def record_weights(self, dt: float = 0.1) -> Self:
        """Enable recording of weight evolution for plastic synapses"""
        ...
        return self

    def set_params(self, params: dict) -> "Env":
        weights = {}
        noise = {}
        cells = {}

        for k, v in params.items():
            if k.startswith("noise-"):
                noise[k.replace("noise-", "")] = v
            elif k.startswith("weight-"):
                weights[k.replace("weight-", "")] = v
            elif k.startswith("cells-"):
                cells[k.replace("cells-", "", 1)] = v
            else:
                weights[k] = v

        env = self
        if weights:
            env.set_weights(weights)
        if noise:
            env.set_noise(noise)
        if cells:
            env = env.cells.set_params(cells)

        return env

    def active_populations(self) -> list[str]:
        ignored: set[str] = set()
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "ignored_populations"):
            ignored = set(model.ignored_populations())
        return [p for p in self.system.populations if p not in ignored]

    def active_neuron_coordinates(self):
        active = self.active_populations()
        if list(active) == list(self.system.populations):
            return self.system.neuron_coordinates
        import numpy as _np

        return _np.vstack([self.system.coordinate_array(p) for p in active])

    def active_gids(self):
        coords = self.active_neuron_coordinates()
        return coords[:, 0].astype(int)

    def simulated_gids(self):
        import numpy as _np

        return _np.array(
            sorted(int(g) for cells in self.cells.values() for g in cells),
            dtype=int,
        )

    def simulated_coordinates(self, transform=None):
        import numpy as _np

        coordinates = self.active_neuron_coordinates()
        gids = coordinates[:, 0].astype(int)
        mine = self.simulated_gids()
        rows = _np.searchsorted(gids, mine)
        if rows.size and not _np.array_equal(gids[rows], mine):
            raise RuntimeError(
                "this rank simulates cells the coordinate table has no row "
                "for; the table is not the whole graph"
            )
        selected = coordinates[rows]
        if transform is None:
            return selected
        return _np.vstack(
            [
                transform(
                    selected[_np.isin(selected[:, 0].astype(int), list(cells))],
                    population=population,
                )
                for population, cells in self.cells.items()
                if cells
            ]
        )

    def record(
        self,
        what: str,
        population: str | list | tuple | None = None,
        **kwargs,
    ) -> Self:
        """Enable recording of the ``what`` signal for population

        A signal is recordable when the environment implements ``_record_<what>``;
        see :meth:`recordable`. Signal-specific options (``dt``, ...) are passed
        through to that implementation as keyword arguments.
        """
        if not isinstance(what, str) or not what.isidentifier():
            raise ValueError(f"invalid signal name: {what!r}")

        try:
            handler = getattr(self, f"_record_{what}")
        except AttributeError:
            raise AttributeError(
                f"cannot record {what!r}; available: {self.recordable()}"
            ) from None

        if population is None:
            population = self.active_populations()
        if isinstance(population, (list, tuple)):
            for p in population:
                handler(p, **kwargs)
            return self

        handler(population, **kwargs)

        return self

    def recordable(self) -> list[str]:
        """Signals that can be passed to :meth:`record`"""
        return sorted(
            name[len("_record_") :] for name in dir(self) if name.startswith("_record_")
        )

    def record_spikes(self, population: str | list | tuple | None = None) -> Self:
        """Enable spike recording for population"""
        return self.record("spikes", population)

    def _record_spikes(self, population: str) -> Self: ...

    def record_voltage(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        """Enable voltage recording for population"""
        return self.record("voltage", population, dt=dt)

    def _record_voltage(self, population: str, dt: float) -> Self: ...

    def record_membrane_current(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        """Enable membrane current recording for population"""
        return self.record("membrane_current", population, dt=dt)

    def _record_membrane_current(self, population: str, dt: float) -> Self: ...

    def run(
        self,
        duration,
        stimulus: Optional["Stimulus"] = None,
        dt: float = 0.025,
        **kwargs,
    ) -> "Run":
        """Run the simulation

        Returns:
            A :class:`~livn.run.Run` exposing:
            - ``spike_ids``: Spiking neuron ids
            - ``spike_times``: Spike times
            - ``voltage_ids``: Voltage recording neuron ids
            - ``voltage``: Voltage traces with shape [n_neurons, timestep]
            - ``current_ids``: Membrane current recording neuron ids
            - ``current``: Membrane current traces with shape [n_neurons, timestep]

            It also unpacks as a six-tuple in exactly that order. Any further
            signal the model exposes to :meth:`record` arrives as a channel of
            its own, reachable by name.
        """
        ...

    def __call__(
        self,
        decoding: Union["Decoding", int],
        inputs: StimulusLike = None,
        encoding: Optional["Encoding"] = None,
        **kwargs,
    ) -> Any:
        self.encoding = encoding
        self.decoding = decoding

        if isinstance(decoding, int):
            duration = decoding
        else:
            duration = decoding.duration
            decoding.setup(self)

        if duration <= 0:
            raise ValueError(f"Encoding duration must be > 0, not {duration}.")

        stimulus = inputs
        if encoding is not None:
            stimulus = encoding(self, duration, inputs)

        response = self.run(duration, stimulus, **kwargs)

        if isinstance(decoding, int):
            return response

        return decoding(response, self)

    @property
    def voltage_recording_dt(self) -> float:
        """Recording time step for voltage traces in ms"""
        return 0.1

    @property
    def membrane_current_recording_dt(self) -> float:
        """Recording time step for membrane current traces in ms"""
        return 0.1

    def recording_distances(self):
        """Distances for the coordinates the membrane currents are recorded at."""
        return self.io.distances(
            self.simulated_coordinates(self.model.recording_coordinates)
        )

    def source_gain(
        self,
    ) -> Float[Array, "n_channels n_recording_coords"]:
        return self.io.source_gain(self.recording_distances())

    def neuron_gain(
        self,
    ) -> Float[Array, "n_channels n_neurons"]:
        return self.model.reduce_source_gain(self.source_gain())

    def potential_recording(
        self,
        membrane_currents: Float[Array, "n_neurons timestep"] | None,
    ) -> Float[Array, "n_channels timestep"]:
        return self.io.potential_recording(
            self.recording_distances(), membrane_currents
        )

    def clear_recordings(self) -> Self:
        """Clear recording buffers

        Note: This preserves simulation state for continued run(); for a full reset use clear()
        """
        ...

        return self

    def clear(self) -> Self:
        """Discard the simulation and reset to t=0"""
        ...

        return self

    def close(self) -> Self:
        """Deconstructor to clean up resources"""
        return self


@runtime_checkable
class Model(Protocol):
    """Protocol defining the interface for livn models"""

    def stimulus_coordinates(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        population: str | None = None,
    ) -> Float[Array, "n_stim_coords ixyz=4"]:
        return neuron_coordinates

    def recording_coordinates(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        population: str | None = None,
    ) -> Float[Array, "n_stim_coords ixyz=4"]:
        return neuron_coordinates

    def reduce_source_gain(
        self,
        gain: Float[Array, "n_channels n_recording_coords"],
    ) -> Float[Array, "n_channels n_neurons"]:
        return gain

    def expand_stimulus_currents(
        self,
        currents: Float[Array, "batch timestep n_neurons"],
    ) -> Float[Array, "batch timestep n_stimulus_coords"]:
        """Expand per-neuron currents to per-stimulus-coordinate currents.

        Mirrors ``reduce_source_gain`` on the stimulus side.  The default
        implementation is identity (one stimulus coordinate per neuron).
        Override for multi-compartment models.
        """
        return currents

    def prepare_stimulus(self, stimulus: "Stimulus") -> "Stimulus":
        return stimulus

    def recordable_states(self) -> tuple[str, ...]:
        return ()

    def diffrax_module(self, env: "Env", key=None):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the diffrax backend"
        )

    def ignored_populations(self) -> set[str]:
        """Populations that backends should skip when instantiating cells/connections."""
        return set()

    def params_key(self) -> str:
        """Key under which this model's parameter sets are stored."""
        return type(self).__name__

    def apply_defaults(self, env, weights: bool = True, noise: bool = True):
        if weights:
            env.set_weights(self.default_weights(env.system.name, default={}))

        if noise:
            env.set_noise(self.default_noise(env.system.name, default={}))

    def default_noise(self, system: str, backend: str | None = None, default=None):
        from livn.backend import backend as current_backend

        if backend is None:
            backend = current_backend()

        try:
            return getattr(self, f"{backend}_default_noise")(system)
        except (AttributeError, KeyError):
            if default is None:
                raise
            return default

    def default_weights(self, system: str, backend: str | None = None, default=None):
        from livn.backend import backend as current_backend

        if backend is None:
            backend = current_backend()

        try:
            return getattr(self, f"{backend}_default_weights")(system)
        except (AttributeError, KeyError):
            if default is None:
                raise
            return default


class Encoding(BaseModel):
    def __call__(self, env: "Env", t_end: int, inputs: Any) -> StimulusLike: ...

    @property
    def input_space(self) -> "gymnasium.Space":
        raise NotImplementedError

    def __hash__(self):
        return int.from_bytes(hashlib.sha256(pickle.dumps(self)).digest()[:8], "little")

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return pickle.dumps(self) == pickle.dumps(other)


class Decoding(BaseModel):
    duration: int

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"duration must be > 0, not {v}.")
        return v

    def __hash__(self):
        return int.from_bytes(hashlib.sha256(pickle.dumps(self)).digest()[:8], "little")

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return pickle.dumps(self) == pickle.dumps(other)

    def setup(self, env: "Env"):
        """Optional setup"""

    def __call__(self, signal: "Run", env: Optional["Env"] = None) -> Any:
        return signal

    @property
    def output_space(self) -> "gymnasium.Space":
        raise NotImplementedError
