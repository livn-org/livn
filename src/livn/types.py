from __future__ import annotations

import hashlib
import pickle
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    Self,
    runtime_checkable,
)

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

    Array = TorchTensor | ndarray | JaxArray | TfTensor

    from jaxtyping import Float, Int

PopulationName = str
PostSynapticPopulationName = PopulationName
PreSynapticPopulationName = PopulationName

# list | dict | tuple | Stimulus | Float[Array, "batch timestep n_channels"] | None
StimulusLike = Any


class SynapticParam(BaseModel):
    population: str | None = None
    source: str | None = None
    sec_type: str | None = None
    syn_name: str | None = None
    param_path: str | tuple[str, ...] | None = None
    param_range: str | None = None
    phenotype: str | None = None

    @field_validator("param_path")
    @classmethod
    def parse_path(cls, v: str | tuple[str, ...] | None) -> tuple[str, ...] | None:
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
    def from_string(cls, string: str) -> SynapticParam:
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
            raise ValueError(f"Failed to parse string '{string}': {e!s}") from e


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

    population_ranges: dict[PopulationName, tuple[int, int]]
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

    def default_io(self, comm: MPI.Intracomm | None = None) -> IO:
        """IO device to use when the environment is constructed without one"""
        ...

    def default_model(self, comm: MPI.Intracomm | None = None) -> Model:
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
    ) -> list[tuple[int, tuple[list[int], Projection]]]:
        """Edges onto ``post`` from ``pre`` as ``(post_gid, (pre_gids, projection))``"""
        ...

    def connectivity_matrix(
        self, weights: dict | None = None, seed: int = 123, gids=None
    ) -> Float[Array, "num_neurons num_neurons"]:
        """Dense signed weight matrix or all-zero when unconnected.

        ``gids`` restricts it to a sub-network, in the order given -- what a cell
        selection induces.
        """
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

    def __init__(self, env: Env, population: PopulationName, gid: int):
        self._env = env
        self._population = str(population)
        self._gid = int(gid)

    @property
    def env(self) -> Env:
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

    def set_params(self, params: dict[str, float]) -> Env:
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


class Capability(StrEnum):
    """Declares what a backend can do."""

    SIMULATION = "simulation"
    """Actually integrates. The default (no ``LIVN_BACKEND``) env does not."""

    MPI = "mpi"
    """Runs on more than one rank, distributing cells between them."""

    PER_GID_VOLTAGE = "per_gid_voltage"
    """``record_voltage(gids=...)`` can narrow the recording to named cells."""

    NOISE = "noise"
    """``set_noise()`` drives the cells with a stochastic conductance."""

    REPLAYABLE_NOISE = "replayable_noise"
    """The noise stream restarts with the simulation, so a run can be replayed."""

    PLASTICITY = "plasticity"
    """``enable_plasticity()`` lets synaptic weights evolve during a run."""

    DIFFERENTIABLE = "differentiable"
    """Gradients flow through ``run()``."""

    IMMUTABLE = "immutable"
    """Env operations return a new env instead of mutating in place."""

    EXTRACELLULAR_STIMULUS = "extracellular_stimulus"
    """Delivers an ``extracellular`` (mV) stimulus."""


@runtime_checkable
class Env(Protocol):
    """Protocol defining the interface for livn environments"""

    capabilities: ClassVar[frozenset[Capability]] = frozenset()
    """What this backend supports, see :class:`Capability`."""

    cells: CellRegistry
    """The simulated cells, addressable by population name or gid"""

    def __init__(
        self,
        system: System | str | int,
        model: Model,
        io: IO,
        seed: int | None = 123,
        comm: MPI.Intracomm | None = None,
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
                k
                for k in params
                if k not in known
                and not k.startswith("cells-")
                and not k.startswith("io-")
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
        dt: float = 1.0,
    ) -> Stimulus:
        """Transforms channel inputs into neural inputs."""
        from livn.policy import Policy

        if isinstance(channel_inputs, Policy):
            channel_inputs = channel_inputs()

        coordinates = self.system.transform_coordinates(
            self.model.stimulus_coordinates,
            populations=self.active_populations(),
        )
        return self.io.cell_stimulus(coordinates, channel_inputs, dt=dt)

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

    def set_params(self, params: dict) -> Env:
        weights = {}
        noise = {}
        cells = {}
        io = {}

        for k, v in params.items():
            if k.startswith("noise-"):
                noise[k.replace("noise-", "")] = v
            elif k.startswith("weight-"):
                weights[k.replace("weight-", "")] = v
            elif k.startswith("cells-"):
                cells[k.replace("cells-", "", 1)] = v
            elif k.startswith("io-"):
                io[k.replace("io-", "", 1)] = v
            else:
                weights[k] = v

        env = self
        if weights:
            env.set_weights(weights)
        if noise:
            env.set_noise(noise)
        if cells:
            env = env.cells.set_params(cells)
            repin = getattr(env, "apply_init_ic", None)
            if callable(repin):
                repin()
        if io:
            if getattr(self, "io", None) is None:
                raise ValueError(
                    f"no io on this env, {sorted('io-' + k for k in io)} is invalid"
                )
            self.io.set_params(io)

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

    def simulated_gids(self, everywhere: bool = False):
        import numpy as _np

        if not everywhere:
            return _np.array(
                sorted(int(g) for cells in self.cells.values() for g in cells),
                dtype=int,
            )

        ranges = getattr(self.system, "population_ranges", None) or {}
        active = set(self.active_populations())
        gids = {
            gid
            for name, (start, count) in ranges.items()
            if name in active
            for gid in range(int(start), int(start) + int(count))
        }

        selected = getattr(self, "_selected_gids", None)
        if selected is not None:
            gids &= {int(g) for g in selected}
        return _np.array(sorted(gids), dtype=int)

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

        if kwargs.get("gids") is not None:
            kwargs["gids"] = self.resolve_recorded_gids(kwargs["gids"])

        if population is None:
            population = self.active_populations()
        if isinstance(population, (list, tuple)):
            for p in population:
                handler(p, **kwargs)
            return self

        handler(population, **kwargs)

        return self

    def resolve_recorded_gids(self, gids) -> set[int]:
        wanted = {int(g) for g in gids}
        if not wanted:
            raise ValueError("no gids to record. Pass `gids=None` to record every cell")

        simulated = {int(g) for g in self.simulated_gids(everywhere=True)}
        missing = sorted(wanted - simulated)
        if missing:
            shown = missing[:10]
            more = (
                ""
                if len(missing) == len(shown)
                else f" (+{len(missing) - len(shown)} more)"
            )
            selection = getattr(self, "selection_name", None)
            because = (
                f"; this env is restricted to the {selection!r} selection"
                if selection
                else ""
            )
            raise ValueError(
                f"{shown}{more} have no cell in this simulation{because}. "
                f"{len(simulated)} gids do"
            )
        return wanted

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
        self,
        population: str | list | tuple | None = None,
        dt: float = 0.1,
        gids: list | tuple | set | None = None,
        sections: str | list | tuple | None = None,
    ) -> Self:
        if isinstance(sections, str):
            sections = [sections]
        return self.record("voltage", population, dt=dt, gids=gids, sections=sections)

    def _record_voltage(
        self, population: str, dt: float, gids=None, sections=None
    ) -> Self: ...

    def record_membrane_current(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        """Enable membrane current recording for population"""
        return self.record("membrane_current", population, dt=dt)

    def _record_membrane_current(self, population: str, dt: float) -> Self: ...

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float = 0.025,
        **kwargs,
    ) -> Run:
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
        decoding: Decoding | int,
        inputs: StimulusLike = None,
        encoding: Encoding | None = None,
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

    @staticmethod
    def _at_gids(coordinates, gids):
        """Keep the coordinate rows belonging to ``gids``, in coordinate order."""
        import numpy as _np

        wanted = {int(g) for g in _np.asarray(gids).ravel()}
        coordinates = _np.asarray(coordinates)
        keep = _np.asarray([int(g) in wanted for g in coordinates[:, 0]], dtype=bool)
        return coordinates[keep]

    def _at_simulated_gids(self, coordinates, everywhere: bool = False):
        """Drop the coordinate rows whose gid is not actually built."""
        return self._at_gids(coordinates, self.simulated_gids(everywhere=everywhere))

    def stimulus_coordinates(self, simulated_only: bool = True):
        """The sections a command couples into, as `[gid, x, y, z]` rows."""
        coordinates = self.system.transform_coordinates(
            self.model.stimulus_coordinates,
            populations=self.active_populations(),
        )
        if not simulated_only:
            return coordinates

        return self._at_simulated_gids(coordinates, everywhere=True)

    def channel_reach(self, coordinates=None):
        """Field induced per unit command at each section, per channel."""
        if coordinates is None:
            coordinates = self.stimulus_coordinates()
        return self.io.reach(coordinates)

    def recording_coordinates(self, simulated_only: bool = False):
        """The sections membrane current is recorded at, as `[gid, x, y, z]` rows."""
        coordinates = self.system.transform_coordinates(
            self.model.recording_coordinates,
            populations=self.active_populations(),
        )
        if not simulated_only:
            return coordinates

        return self._at_simulated_gids(coordinates)

    def recording_sections_per_cell(self, population: str) -> int:
        """How many sections of a `population` cell carry a recording coordinate."""
        coordinates = self.system.coordinate_array(population)
        n = len(coordinates)
        if n == 0:
            return 0
        rows = self.model.recording_coordinates(coordinates, population=population)
        return max(1, len(rows) // n)

    def recording_distances(self, gids=None):
        """Distances for the coordinates the membrane currents are recorded at."""
        import numpy as _np

        coordinates = self.recording_coordinates()
        if gids is not None:
            gids = _np.asarray(gids).ravel()
            if len(gids) != len(coordinates):
                coordinates = self._at_gids(coordinates, gids)
        return self.io.distances(coordinates)

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
        gids=None,
    ) -> Float[Array, "n_channels timestep"]:
        return self.io.potential_recording(
            self.recording_distances(gids), membrane_currents
        )

    def clear_recordings(self) -> Self:
        """Clear recording buffers

        Note: This preserves simulation state for continued run(); for a full reset use clear()
        """
        ...

        return self

    def clear(self, reseed: bool = True) -> Self:
        """Discard the simulation and reset to t=0.

        ``reseed`` advances the stochastic streams so the next ``run()`` is an
        independent realisation rather than a repeat of the last one.
        """
        ...

        return self

    def reseed_noise(self, stream: int | None = None) -> Self:
        """Move every stochastic stream onto a fresh, reproducible realisation."""
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

    def prepare_stimulus(self, stimulus: Stimulus) -> Stimulus:
        return stimulus

    def stimulus_bounds(self, input_mode: str) -> tuple[float, float] | None:
        return None

    def recordable_states(self) -> tuple[str, ...]:
        return ()

    def diffrax_module(self, env: Env, key=None):
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
    def __call__(self, env: Env, t_end: int, inputs: Any) -> StimulusLike: ...

    @property
    def input_space(self) -> gymnasium.Space:
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

    def setup(self, env: Env):
        """Optional setup"""

    def __call__(self, signal: Run, env: Env | None = None) -> Any:
        return signal

    @property
    def output_space(self) -> gymnasium.Space:
        raise NotImplementedError
