from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy

from livn import types
from livn.backend import backend
from livn.system._common import (
    CellsMetaData,
    Projection,
    resolve_selection,
)
from livn.utils import sentinel

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.types import Model

_USES_JAX = False

if "ax" in backend():
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


class ParallelSystem:
    """A number of unconnected neurons, simulated independently in parallel

    Implements the :class:`livn.types.System` protocol without an H5 graph.
    Use it to simulate a single cell, or N cells that never interact::

        env = Env(64).init()   # 64 independent cells

    Arguments:
        num_neurons: Either a total cell count, which puts every cell in
            ``"EXC"`` (``3`` is short for ``{"EXC": 3}``), or a
            ``{population: count}`` mapping for several populations, e.g.
            ``{"EXC": 3, "INH": 5}``. Gids are assigned contiguously in the
            order the populations are given. Models key their cell factories by
            population name, so each name has to be one the model defines.
        coordinates: Where the cells sit, as either

            - a ``float`` spacing in um, laying the cells out along x in gid
              order (the default ``0.0`` puts every cell at the origin),
            - an ``[n_neurons, 3]`` array of ``x, y, z`` (or ``[n_neurons, 4]``
              of ``gid, x, y, z``, whose gids may be arbitrary as long as they
              are whole and unique), or
            - a callable taking the total cell count and returning either of
              the above.

        name: Identifier that keys per-system model defaults.
    """

    def __init__(
        self,
        num_neurons: int | Mapping[types.PopulationName, int] = 1,
        coordinates: float | Callable | Any = 0.0,
        name: str = "ParallelSystem",
        comm: MPI.Intracomm | None = None,
    ):
        if isinstance(num_neurons, bool):
            raise TypeError("num_neurons must be an int or a mapping, not a bool")
        if isinstance(num_neurons, (int, numpy.integer)):
            counts = {"EXC": int(num_neurons)}
        elif isinstance(num_neurons, Mapping):
            counts = {str(p): int(c) for p, c in num_neurons.items()}
            if not counts:
                raise ValueError("num_neurons must name at least one population")
        else:
            raise TypeError(
                f"num_neurons must be an int or a {{population: count}} mapping, "
                f"not {type(num_neurons).__name__}"
            )
        for p, count in counts.items():
            if count < 0:
                raise ValueError(f"population {p!r} has a negative count ({count})")

        self.population_counts = counts
        self.num_neurons = sum(counts.values())
        if self.num_neurons < 1:
            raise ValueError(
                f"num_neurons must be >= 1 in total, not {self.num_neurons}"
            )

        self.name = name
        self.comm = comm
        self.uri = None

        populations = list(self.population_counts)

        self.files: dict[str, str] = {}
        self.connections_config = {"synapses": {p: {} for p in populations}}

        ranges: dict[types.PopulationName, tuple[int, int]] = {}
        start = 0
        for p, count in self.population_counts.items():
            ranges[p] = (start, count)
            start += count

        self.cells_meta_data = CellsMetaData(
            population_names=populations,
            population_ranges=ranges,
            cell_attribute_info={p: {} for p in populations},
        )

        # resolve the coordinate spec into [gid, x, y, z] rows
        if callable(coordinates):
            coordinates = coordinates(self.num_neurons)

        gids = np.arange(self.num_neurons, dtype=float).reshape(-1, 1)

        if isinstance(
            coordinates, (int, float, numpy.integer, numpy.floating)
        ) and not isinstance(coordinates, bool):
            zeros = np.zeros((self.num_neurons, 1))
            self._neuron_coordinates = np.concatenate(
                [gids, gids * float(coordinates), zeros, zeros], axis=1
            )
            return

        array = np.asarray(coordinates, dtype=float)
        if array.ndim != 2 or array.shape[1] not in (3, 4):
            raise ValueError(
                "coordinates must be a spacing, an [n_neurons, 3] array of x, y, z "
                "or an [n_neurons, 4] array of gid, x, y, z; got shape "
                f"{tuple(array.shape)}"
            )
        if array.shape[0] != self.num_neurons:
            raise ValueError(
                f"expected {self.num_neurons} coordinate rows, one per neuron, "
                f"got {array.shape[0]}"
            )

        if array.shape[1] == 3:
            self._neuron_coordinates = np.concatenate([gids, array], axis=1)
            return

        gid_column = array[:, 0]
        if not bool(np.array_equal(gid_column, np.floor(gid_column))):
            raise ValueError("GID column of a coordinates array must contain integers")
        if len(numpy.unique(numpy.asarray(gid_column))) != self.num_neurons:
            raise ValueError("GID column of a coordinates array must be unique")
        self._neuron_coordinates = array

    def serialize(self) -> dict:
        return {
            "num_neurons": dict(self.population_counts),
            "coordinates": self._neuron_coordinates[:, 1:4].tolist(),
            "name": self.name,
        }

    def __repr__(self):
        return f"ParallelSystem({self.population_counts!r})"

    def default_io(self, comm=None) -> IO:
        from livn.io import IO

        return IO()

    def default_model(self, comm=None) -> Model:
        from livn.models.rcsd import ReducedCalciumSomaDendrite

        return ReducedCalciumSomaDendrite()

    def load_file(self, filepath: str | list[str], default: Any = sentinel, **kwargs):
        if default is sentinel:
            raise FileNotFoundError(f"{self!r} has no files ({filepath})")
        return default

    @property
    def population_ranges(self) -> dict[types.PopulationName, tuple[int, int]]:
        return self.cells_meta_data.population_ranges

    def population_count(self, population: types.PopulationName) -> int:
        return self.cells_meta_data.population_count(population)

    @property
    def populations(self) -> list[types.PopulationName]:
        return self.cells_meta_data.population_names

    def synapse_projections(self) -> list[tuple[str, str, str, str, str]]:
        return []

    @property
    def weight_names(self) -> list[str]:
        return []

    @property
    def neuron_coordinates(self) -> types.Float[types.Array, "n_coords ixyz=4"]:
        return self._neuron_coordinates

    @property
    def gids(self) -> types.Int[types.Array, "n_neurons"]:
        if _USES_JAX:
            return np.asarray(self._neuron_coordinates[:, 0], dtype=int)

        return self._neuron_coordinates[:, 0].astype(int)

    @property
    def bounding_box(self) -> types.Float[types.Array, "2 xyz=3"]:
        coordinates = self._neuron_coordinates[:, 1:4]
        padding = 100.0
        return np.stack(
            [coordinates.min(axis=0) - padding, coordinates.max(axis=0) + padding]
        )

    @property
    def center_point(self) -> types.Float[types.Array, "xyz=3"]:
        bb = self.bounding_box
        return (bb[0] + bb[1]) / 2.0

    def _population_slice(self, population: types.PopulationName) -> slice:
        try:
            start, count = self.cells_meta_data.population_ranges[population]
        except KeyError:
            raise KeyError(
                f"{self!r} has no population {population!r}; "
                f"expected one of {list(self.population_counts)}"
            ) from None
        return slice(start, start + count)

    def coordinates(
        self, population: types.PopulationName
    ) -> Iterator[tuple[int, tuple[float, float, float]]]:
        for row in self._neuron_coordinates[self._population_slice(population)]:
            yield int(row[0]), (float(row[1]), float(row[2]), float(row[3]))

    def coordinate_array(
        self, population: types.PopulationName
    ) -> types.Float[types.Array, "n_coords cxyz=4"]:
        return self._neuron_coordinates[self._population_slice(population)]

    def transform_coordinates(
        self,
        transform: Callable,
        populations: list[str] | None = None,
    ) -> types.Float[types.Array, "n_coords ixyz=4"]:
        if populations is None:
            populations = self.populations
        return np.vstack(
            [transform(self.coordinate_array(p), population=p) for p in populations]
        )

    def selection(
        self,
        spec,
        populations: Sequence[str] | None = None,
        seed: int | None = 123,
        method: str = "first",
        bounds=None,
    ) -> dict[str, Any] | None:
        return resolve_selection(self, spec, populations, seed, method, bounds)

    def selections(self, comm=None) -> list[str]:
        return []

    def selection_document(self, name: str, comm=None) -> dict:
        raise FileNotFoundError(f"{self!r} stores no selections")

    def projections(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
    ) -> Iterator[tuple[int, tuple[list[int], Projection]]]:
        return iter(())

    def projection_array(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        all: bool = True,
    ) -> list[tuple[int, tuple[list[int], Projection]]]:
        return []

    def synapses(
        self,
        population: types.PostSynapticPopulationName,
        node_allocation: set[int] | None = None,
    ):
        return iter(())

    def placement(self, population: types.PopulationName, gids) -> dict[int, tuple]:
        """Unconnected, so no cell carries a synapse site."""
        return {}

    def edges(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        gids,
    ):
        return iter(())

    def connectivity_matrix(
        self, weights: dict | None = None, seed=123, gids=None
    ) -> types.Float[types.Array, "num_neurons num_neurons"]:
        """Unconnected, so the only thing ``gids`` changes is the size."""
        # use numpy, not jax
        import numpy as npn

        n = self.num_neurons if gids is None else len(npn.asarray(gids))
        return npn.zeros([n, n], dtype=npn.float32)

    def summary(self) -> dict[str, int | dict[str, int]]:
        return {
            "num_neurons": self.num_neurons,
            "num_projections": 0,
            "population_counts": dict(self.population_counts),
        }
