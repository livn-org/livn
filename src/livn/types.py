from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
from collections.abc import Iterable, Iterator, Mapping, Sequence
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

from livn.utils import Jsonable

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

SECTION_VOCABULARY = ("apical", "axon", "basal", "dend", "soma")

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

    gids: Int[Array, " n_neurons"]
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
        """Edges onto ``post`` from ``pre`` as ``(post_gid, (pre_gids, projection))``

        Every rank gets every edge. Prefer :meth:`edges`, which is scoped to the
        cells a rank actually builds.
        """
        ...

    def placement(
        self, population: PopulationName, gids: Iterable[int]
    ) -> dict[int, tuple[Array, Array, Array]]:
        """Where each synapse sits on the cell that owns it.

        ``{gid: (syn_ids, swc_types, syn_locs)}`` for ``gids``, each triple sorted
        by ``syn_id`` with duplicates dropped (a repeated id keeps its last site).

        May be collective, so a caller iterating populations has to call it for
        every population on every rank, including where ``gids`` is empty.
        """
        ...

    def edges(
        self,
        pre: PreSynapticPopulationName,
        post: PostSynapticPopulationName,
        gids: Iterable[int],
    ) -> Iterator[tuple[int, tuple[Array, Projection]]]:
        """Edges of one projection onto ``gids``.

        Yields ``(post_gid, (pre_gids, projection))`` for the postsynaptic cells
        in ``gids`` that this projection reaches, with ``projection`` carrying the
        ``"Synapses"`` (``syn_id``) and ``"Connections"`` (``distance``) namespaces.

        May be collective, call it for every ``(pre, post)`` pair on every rank.
        """
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


def _describe(obj) -> dict | None:
    """``{"cls", "kwargs"}`` for a system, model or io."""
    if obj is None:
        return None
    return {
        "cls": f"{type(obj).__module__}.{type(obj).__qualname__}",
        "kwargs": obj.serialize(),
    }


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        return [_plain(v) for v in value]
    return value


def _build(described):
    if described is None:
        return None
    if isinstance(described, (list, tuple)):
        path, kwargs = [*list(described), {}][:2]
        described = {"cls": path, "kwargs": kwargs}
    if isinstance(described, str):
        from livn.utils import import_instance

        return import_instance(described)
    if not isinstance(described, Mapping):
        return described  # already built
    if "cls" not in described:
        raise ValueError(
            f"expected {{'cls': ..., 'kwargs': ...}} naming what to build, got "
            f"{described!r}"
        )

    from livn.utils import import_object_by_path

    return import_object_by_path(described["cls"])(
        **_plain(described.get("kwargs") or {})
    )


def _is_the_systems_own_io(system, io) -> bool:
    if io is None:
        return True

    import json

    from livn.utils import serialize as _default

    def rendered(value):
        return json.dumps(_describe(value), default=_default, sort_keys=True)

    with contextlib.suppress(Exception):
        return rendered(system.default_io()) == rendered(io)
    return False


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
        ii: Float[Array, " i"] | None,
        *recordings: Float[Array, " _"],
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
        """Config section name -> key section name, per population."""
        declared = set(SECTION_VOCABULARY)
        with contextlib.suppress(
            OSError, KeyError, TypeError, ValueError, AttributeError
        ):
            declared |= {s for _, _, s, _, _ in self.system.synapse_projections()}

        namer = getattr(self.model, "section_name", None)
        if not callable(namer):
            return {}
        return {
            population: {
                section: str(namer(population, section)) for section in sorted(declared)
            }
            for population in self.system.populations
        }

    @property
    def weight_names(self) -> list[str]:
        """The weight keys this network accepts."""
        names = []
        for post, pre, section, mechanism, _ in self.system.synapse_projections():
            resolved = self.model.section_name(post, section)
            name = f"{post}_{pre}-{resolved}-{mechanism}-weight"
            if name not in names:
                names.append(name)
        return names

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

    def unmatched_params(self, params: dict) -> dict[str, str]:
        """`{key: reason} of `params` that do not match the network."""
        declared = {
            (post, pre, self.model.section_name(post, section), mechanism)
            for post, pre, section, mechanism, _ in (self.system.synapse_projections())
        }
        populations = set(self.system.populations)
        sections = {row[2] for row in declared}
        mechanisms = {row[3] for row in declared}

        unmatched: dict[str, str] = {}
        for key in params:
            if key.startswith(("noise-", "cells-", "io-", "weight-")):
                continue
            try:
                p = SynapticParam.from_string(key)
            except ValueError:
                continue
            for value, known, what in (
                (p.population, populations, "population"),
                (p.source, populations, "source population"),
                (p.sec_type, sections, "section"),
                (p.syn_name, mechanisms, "mechanism"),
            ):
                if value is not None and value not in known:
                    unmatched[key] = (
                        f"no {what} {value!r} in this network (it has {sorted(known)})"
                    )
                    break
            else:
                if (
                    p.population,
                    p.source,
                    p.sec_type,
                    p.syn_name,
                ) not in declared and (p.source is not None):
                    unmatched[key] = (
                        f"{p.population!r} has no {p.syn_name} synapse from "
                        f"{p.source!r} on {p.sec_type!r}"
                    )
        return unmatched

    def set_params(self, params: dict, strict: bool = False) -> Env:
        if strict:
            unmatched = self.unmatched_params(params)
            if unmatched:
                lines = "\n".join(f"  {k}: {why}" for k, why in unmatched.items())
                raise ValueError(
                    f"{len(unmatched)} parameter(s) address nothing in this "
                    f"network, so they would be silently ignored:\n{lines}"
                )
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

        # remember what was applied
        applied = {**getattr(self, "_applied_params", {}), **params}
        self._applied_params = applied
        if env is not self:
            env._applied_params = applied

        return env

    @property
    def applied_params(self) -> dict:
        return dict(getattr(self, "_applied_params", {}))

    def serialize(self) -> dict:
        system = self.system
        io = getattr(self, "io", None)
        return {
            "system": _describe(system),
            "model": _describe(self.model),
            "io": None if _is_the_systems_own_io(system, io) else _describe(io),
            "selection": getattr(self, "selection_name", None),
            "params": self.applied_params,
            "meta": dict(getattr(self, "meta", {}) or {}),
        }

    as_json = Jsonable.as_json

    def save(self, path: str) -> str:
        if os.path.isdir(path) or not path.endswith(".json"):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, "env.json")

        document = self.serialize()

        system = document.get("system") or {}
        uri = (system.get("kwargs") or {}).get("uri")
        if isinstance(uri, str):
            here = os.path.dirname(os.path.abspath(path))
            absolute = os.path.abspath(uri)
            if os.path.commonpath([here, absolute]) == here:
                system["kwargs"]["uri"] = os.path.relpath(absolute, here)
            else:
                system["kwargs"]["uri"] = absolute

        import json

        from livn.utils import serialize as _default

        with open(path, "w") as f:
            json.dump(document, f, default=_default, indent=2)
        return path

    @staticmethod
    def document(source) -> dict:
        import json

        if not isinstance(source, (str, os.PathLike)):
            return dict(source)

        path = os.fspath(source)
        if os.path.isdir(path):
            path = os.path.join(path, "env.json")
        if not path.endswith(".json"):
            return json.loads(path)

        with open(path) as f:
            document = json.load(f)

        here = os.path.dirname(os.path.abspath(path))
        kwargs = (document.get("system") or {}).get("kwargs") or {}
        uri = kwargs.get("uri")
        if isinstance(uri, str) and not os.path.isabs(uri):
            kwargs["uri"] = os.path.normpath(os.path.join(here, uri))
        return document

    @staticmethod
    def stored_params(source) -> dict:
        if source is None:
            return {}
        try:
            return dict(Env.document(source).get("params") or {})
        except (OSError, ValueError):
            return {}

    @classmethod
    def from_json(
        cls,
        serialized,
        selection=None,
        method: str = "first",
        params: dict | None = None,
        strict: bool = True,
        **kwargs,
    ) -> Env:
        """Build the env a document describes.

        Args:
            serialized: A document, or a path to one.
            selection: Overrides the document's, with `method`.
            params: Applied after the document's, overriding them.
            strict: Refuse parameters this network has nothing to apply to.
        """
        from livn.system import resolve

        document = cls.document(serialized)

        described = document.get("system")
        if described is None:
            raise ValueError("an env document has to name a system")

        system = resolve(described, comm=kwargs.get("comm"))

        env = cls(
            system,
            model=_build(document.get("model")),
            io=_build(document.get("io")),
            **kwargs,
        )
        env.meta = dict(document.get("meta") or {})

        selection = selection if selection is not None else document.get("selection")
        if selection is not None:
            env = env.selection(selection, method=method) or env
        env = env.init() or env

        applied = {**(document.get("params") or {}), **(params or {})}
        return env.set_params(applied, strict=strict) if applied else env

    @classmethod
    def from_directory(cls, directory: str, **kwargs) -> Env:
        return cls.from_json(os.path.join(directory, "env.json"), **kwargs)

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

    def section_name(self, population: str, section: str) -> str:
        """Name a config section resolves to in weight and `cells-` keys."""
        return section

    def serialize(self) -> dict:
        import inspect

        kwargs = {}
        for name, parameter in inspect.signature(
            type(self).__init__
        ).parameters.items():
            if name == "self" or parameter.kind in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                continue
            if not hasattr(self, name):
                raise NotImplementedError(
                    f"{type(self).__name__} takes {name!r} but does not keep it "
                    f"under that name, so it cannot be serialized by reading its "
                    f"attributes back; give {type(self).__name__} a `serialize`"
                )
            kwargs[name] = getattr(self, name)
        return kwargs

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
