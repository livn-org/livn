import hashlib
import json
import os
import random
from functools import cached_property
from typing import TYPE_CHECKING, Iterator, Optional

import pandas as pd
from pydantic import BaseModel

from livn import types
from livn.backend import backend

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO

_USES_JAX = False

if "ax" in backend():
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


class CellsMetaData(BaseModel):
    """Cells metadata"""

    population_names: list[types.PopulationName]
    population_ranges: dict[types.PopulationName, tuple[int, int]]
    cell_attribute_info: dict[types.PopulationName, dict[str, list[str]]]

    def has(self, population: types.PopulationName, attribute: str) -> bool:
        return attribute in self.cell_attribute_info.get(population, {})

    def population_count(self, population: types.PopulationName) -> int:
        return self.population_ranges[population][1]

    def cell_count(self) -> int:
        """Return the total number of cells across all populations."""
        return sum(
            self.population_count(population) for population in self.population_names
        )


class Tree(BaseModel):
    """Tree"""


class Projection(BaseModel):
    """Projection"""


class Element(BaseModel):
    uuid: str = str | None
    kind: str = "Element"
    module: str | None = None
    version: list[str | dict] = []
    config: dict | None = None
    predicate: dict | None = None
    context: dict | None = None
    lineage: tuple[str, ...] = ()


def predefined(name: str = "S1", download_directory: str = ".", force: bool = False):
    target = None
    comm = None
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        if comm.Get_rank() != 0:
            # await download on 0
            return comm.bcast(target)
    except ImportError:
        pass

    from huggingface_hub import snapshot_download

    available = [f"S{s + 1}" for s in range(4)] + ["C5"]

    if name not in available:
        raise ValueError(f"'{name}' is invalid, pick one of ", available)

    path = os.path.join("systems/data", name)
    target = os.path.join(download_directory, path)

    if force or not os.path.isdir(target):
        snapshot_download(
            "livn-org/livn",
            repo_type="dataset",
            allow_patterns=os.path.join(path, "*"),
            local_dir=download_directory,
        )

    if comm is not None:
        comm.bcast(target)

    return target


def make(name: str = "S1", cached: bool = True) -> "System":
    system = predefined(name)

    if cached:
        return CachedSystem(system)

    return System(system)


def read_cells_meta_data(
    filepath: str, comm: Optional["MPI.Intracomm"] = None
) -> CellsMetaData:
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    from neuroh5.io import (
        read_cell_attribute_info,
        read_population_names,
        read_population_ranges,
    )

    rank = comm.Get_rank()
    comm0 = comm.Split(int(rank == 0), 0)
    cell_attribute_info = None
    population_ranges = None
    population_names = None
    if rank == 0:
        population_names = read_population_names(filepath, comm0)
        (population_ranges, _) = read_population_ranges(filepath, comm0)
        cell_attribute_info = read_cell_attribute_info(
            filepath, population_names, comm=comm0
        )
    population_ranges = comm.bcast(population_ranges, root=0)
    population_names = comm.bcast(population_names, root=0)
    cell_attribute_info = comm.bcast(cell_attribute_info, root=0)

    comm0.Free()

    return CellsMetaData(
        population_names=population_names,
        population_ranges=population_ranges,
        cell_attribute_info=cell_attribute_info,
    )


def read_coordinates(
    filepath: str,
    population: types.PopulationName,
    comm: Optional["MPI.Intracomm"] = None,
) -> Iterator[tuple[int, tuple[float, float, float]]]:
    from mpi4py import MPI
    from neuroh5.io import scatter_read_cell_attributes

    if comm is None:
        comm = MPI.COMM_WORLD

    cell_attr_dict = scatter_read_cell_attributes(
        filepath,
        population,
        namespaces=["Generated Coordinates"],
        return_type="tuple",
        comm=comm,
    )
    coords_iter, coords_attr_info = cell_attr_dict["Generated Coordinates"]
    x_index = coords_attr_info.get("X Coordinate", None)
    y_index = coords_attr_info.get("Y Coordinate", None)
    z_index = coords_attr_info.get("Z Coordinate", None)
    for gid, cell_coords in coords_iter:
        yield (
            gid,
            (
                cell_coords[x_index][0],
                cell_coords[y_index][0],
                cell_coords[z_index][0],
            ),
        )


def coordinate_array(
    filepath: str,
    population: types.PopulationName,
    comm: Optional["MPI.Intracomm"] = None,
    all: bool = True,
) -> types.Float[types.Array, "n_coords cxyz=4"]:
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    coordinates = []
    for gid, coordinate in read_coordinates(filepath, population, comm=comm):
        coordinates.append([gid] + list(coordinate))

    if all:
        all_coordinates = comm.allgather(coordinates)
        coordinates = np.array(
            [coord for sublist in all_coordinates for coord in sublist]
        )
    else:
        coordinates = np.array(coordinates)

    if coordinates.size == 0:
        return np.zeros((0, 4))

    return coordinates[coordinates[:, 0].argsort()]


def read_trees(
    filepath: str,
    population: types.PopulationName,
    comm: Optional["MPI.Intracomm"] = None,
) -> Iterator[tuple[int, Tree]]:
    from mpi4py import MPI
    from neuroh5.io import scatter_read_trees

    if comm is None:
        comm = MPI.COMM_WORLD

    (trees, forestSize) = scatter_read_trees(filepath, population, comm=comm)
    yield from trees


def read_synapses(
    filepath: str,
    population: types.PostSynapticPopulationName,
    comm: Optional["MPI.Intracomm"] = None,
    node_allocation: set[int] | None = None,
):
    from mpi4py import MPI
    from neuroh5.io import scatter_read_cell_attributes

    if comm is None:
        comm = MPI.COMM_WORLD

    cell_attributes_dict = scatter_read_cell_attributes(
        filepath,
        population,
        namespaces=["Synapse Attributes"],
        mask={
            "syn_ids",
            "syn_locs",
            "syn_secs",
            "syn_layers",
            "syn_types",
            "swc_types",
        },
        comm=comm,
        node_allocation=node_allocation,
        io_size=1,
        return_type="dict",
    )

    for gid, syn_attrs in cell_attributes_dict["Synapse Attributes"]:
        yield gid, syn_attrs


def read_projections(
    filepath: str,
    pre: types.PreSynapticPopulationName,
    post: types.PostSynapticPopulationName,
    comm: Optional["MPI.Intracomm"] = None,
) -> Iterator[tuple[int, tuple[list[int], Projection]]]:
    from mpi4py import MPI
    from neuroh5.io import scatter_read_graph

    if comm is None:
        comm = MPI.COMM_WORLD

    (graph, a) = scatter_read_graph(
        filepath,
        comm=comm,
        io_size=1,
        projections=[(pre, post)],
        namespaces=["Synapses", "Connections"],
    )

    yield from graph[post][pre]


def projection_array(
    filepath: str,
    pre: types.PreSynapticPopulationName,
    post: types.PostSynapticPopulationName,
    comm: Optional["MPI.Intracomm"] = None,
    all: bool = True,
) -> list[tuple[int, tuple[list[int], Projection]]]:
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    projections = []
    for post_gid, (pre_gids, projection) in read_projections(
        filepath, pre, post, comm=comm
    ):
        projections.append([post_gid, (pre_gids, projection)])

    if all:
        all_projections = comm.allgather(projections)
        projections = [projs for sublist in all_projections for projs in sublist]

    return projections


class NeuroH5Graph:
    def __init__(self, directory):
        self.directory = os.path.abspath(directory)

    def local_directory(self, *args):
        return os.path.join(self.directory, *args)

    @property
    def cells_filepath(self):
        if os.path.isfile(self.local_directory("graph.h5")):
            return self.local_directory("graph.h5")
        return self.local_directory("cells.h5")

    @property
    def connections_filepath(self):
        if os.path.isfile(self.local_directory("graph.h5")):
            return self.local_directory("graph.h5")
        return self.local_directory("connections.h5")

    @cached_property
    def elements(self):
        with open(self.local_directory("graph.json")) as f:
            graph = json.load(f)

        def _load_element(model):
            if "uuid" not in model:
                return {k: _load_element(v) for k, v in model.items()}

            return Element(**model)

        for k in graph:
            graph[k] = _load_element(graph[k])

        return graph

    @property
    def architecture(self):
        return self.elements["architecture"]

    @property
    def distances(self):
        return self.elements["distances"]

    @property
    def synapse_forest(self):
        return self.elements["synapse_forest"]

    @property
    def synapses(self):
        return self.elements["synapses"]

    @property
    def connections(self):
        return self.elements["connections"]

    def files(self) -> dict[str, str]:
        return {
            "cells": self.cells_filepath,
            "connections": self.connections_filepath,
        }

    @property
    def population_names(self):
        return list(self.architecture.config.cell_distributions.keys())

    @property
    def layer_names(self):
        return list(self.architecture.config.layer_extents.keys())


class System:
    """In vitro system"""

    def __init__(self, uri: str, comm: Optional["MPI.Intracomm"] = None):
        self.uri = uri
        self.comm = comm

        self._graph = NeuroH5Graph(uri)
        self._cells_meta_data = None
        self.connections_config = next(iter(self._graph.connections.values())).config
        self.synapses_config = next(iter(self._graph.synapses.values())).config
        self.files = self._graph.files()
        self._neuron_coordinates = None

    def default_io(self) -> "IO":
        from livn.io import MEA

        return MEA.from_directory(self.uri)

    @property
    def name(self):
        return self.uri.split("/")[-1]

    @property
    def cells_meta_data(self):
        if self._cells_meta_data is None:
            self._cells_meta_data = read_cells_meta_data(
                self._graph.cells_filepath, comm=self.comm
            )
        return self._cells_meta_data

    @property
    def populations(self):
        return self.cells_meta_data.population_names

    @property
    def neuron_coordinates(self) -> types.Float[types.Array, "n_coords ixyz=4"]:
        if self._neuron_coordinates is None:
            self._neuron_coordinates = np.vstack(
                [
                    self.coordinate_array(population_name, all=False)
                    for population_name in self.populations
                ]
            )
            self._neuron_coordinates[self._neuron_coordinates[:, 0].argsort()]

        return self._neuron_coordinates

    @property
    def gids(self) -> types.Int[types.Array, "n_neurons"]:
        if _USES_JAX:
            return np.asarray(self.neuron_coordinates.at[:, 0], dtype=int)

        return self.neuron_coordinates[:, 0].astype(int)

    def coordinates(
        self, population: types.PopulationName
    ) -> Iterator[tuple[int, tuple[float, float, float]]]:
        yield from read_coordinates(
            self._graph.cells_filepath, population, comm=self.comm
        )

    def coordinate_array(
        self, population: types.PopulationName, all: bool = True
    ) -> types.Float[types.Array, "n_coords cxyz=4"]:
        return coordinate_array(
            self._graph.cells_filepath, population, comm=self.comm, all=all
        )

    def projections(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
    ) -> Iterator[tuple[int, tuple[list[int], Projection]]]:
        yield from read_projections(
            self._graph.connections_filepath, pre, post, comm=self.comm
        )

    def synapses(
        self,
        population: types.PostSynapticPopulationName,
        node_allocation: set[int] | None = None,
    ):
        yield from read_synapses(
            self._graph.cells_filepath, population, self.comm, node_allocation
        )

    def projection_array(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        all: bool = True,
    ) -> list[tuple[int, tuple[list[int], Projection]]]:
        return projection_array(
            self._graph.connections_filepath, pre, post, comm=self.comm, all=all
        )

    def connectivity_matrix(
        self, weights: dict | None = None, reduced: bool = False, seed=123
    ) -> types.Float[types.Array, "num_neurons num_neurons"]:
        # use numpy, not jax
        import numpy as npn

        prng = random.Random(seed)

        if weights is None:
            weights = {}

        num_neurons = self.cells_meta_data.cell_count()
        w = npn.zeros([num_neurons, num_neurons], dtype=npn.float32)

        for post, v in self.connections_config["synapses"].items():
            for pre, synapse in v.items():
                kind = synapse["type"]
                prefix = -1.0 if kind == "inhibitory" else 1.0
                weight = weights.get(f"{pre}_{post}", 1.0)

                for post_gid, (pre_gids, projection) in self.projection_array(
                    pre, post
                ):
                    distances = projection
                    if isinstance(projection, dict):
                        distances = projection["Connections"][0]

                    # filter autapses
                    autapse = pre_gids == post_gid
                    distances = distances[~autapse]
                    pre_gids = pre_gids[~autapse]

                    if not reduced:
                        for pre_gid in pre_gids:
                            w[pre_gid, post_gid] = prefix * prng.random() * weight
                    else:
                        # multiplier connectivity
                        unique_q, inverse_indices = npn.unique(
                            pre_gids, return_inverse=True
                        )
                        multiplier = npn.bincount(inverse_indices) / 10000.0

                        for m, pre_gid in enumerate(unique_q):
                            w[pre_gid, post_gid] = (
                                prefix * prng.random() * weight * multiplier[m]
                            )

        return w

    def summary(self) -> dict[str, int | dict[str, int]]:
        import h5py

        num_neurons = 0
        num_projections = 0
        population_counts = {}

        for population in self.populations:
            count = self.cells_meta_data.population_count(population)
            population_counts[population] = count
            num_neurons += count

        with h5py.File(self._graph.connections_filepath, "r") as h5file:
            for post, v in self.connections_config["synapses"].items():
                for pre, _ in v.items():
                    path = f"Projections/{pre}/{post}/Connections"
                    if path in h5file:
                        distances = h5file[path]["distance"]
                        num_projections += len(distances)

        return {
            "num_neurons": num_neurons,
            "num_projections": num_projections,
            "population_counts": population_counts,
        }


class CachedSystem(System):
    def __init__(
        self, uri: str, comm: Optional["MPI.Intracomm"] = None, filter_projections=False
    ):
        super().__init__(uri=uri, comm=comm)

        rank = 0
        if self.comm:
            rank = self.comm.Get_rank()
        if rank == 0:
            try:
                parquet_file = self._graph.local_directory("system.parquet")
                if os.path.exists(parquet_file):
                    df = pd.read_parquet(parquet_file)

                    self._cache = self._deserialize_cache(df)
                else:
                    self._cache = {}
            except Exception:
                self._cache = {}
        else:
            self._cache = {}
        if self.comm:
            self._cache = self.comm.bcast(self._cache, root=0)

        if len(self._cache) == 0:
            # build cache
            from mpi4py import MPI

            if comm is None:
                comm = MPI.COMM_WORLD

            c = System(uri, comm)
            self._cache["cells_meta_data"] = c.cells_meta_data
            for p in c.cells_meta_data.population_names:
                self._cache[f"coordinates_{p}"] = c.coordinate_array(p, all=True)
            for post, v in c.connections_config["synapses"].items():
                for pre, _ in v.items():
                    if float(filter_projections) > 0:
                        projections = []
                        for post_gid, (pre_gids, projection) in c.projections(
                            pre, post
                        ):
                            distances = projection["Connections"][0]

                            mask = np.random.rand(len(pre_gids)) > filter_projections
                            if not np.any(mask):
                                mask[0] = True
                            projections.append(
                                [post_gid, (pre_gids[mask], distances[mask])]
                            )

                        projections = [
                            projs
                            for sublist in c.comm.allgather(projections)
                            for projs in sublist
                        ]
                    else:
                        projections = c.projection_array(pre, post, all=True)
                    self._cache[f"projection_{pre}_{post}"] = projections
            del c
            if rank == 0:
                serialized_cache = self._serialize_cache(self._cache)
                df = pd.DataFrame([serialized_cache])
                df.to_parquet(
                    self._graph.local_directory("system.parquet"), index=False
                )

    def _serialize_cache(self, cache):
        """Convert cache to parquet-compatible format"""
        serialized = {}

        if "cells_meta_data" in cache:
            cells_meta = cache["cells_meta_data"]
            serialized["cells_meta_data"] = {
                "population_names": cells_meta.population_names,
                "population_ranges": json.dumps(cells_meta.population_ranges),
                "cell_attribute_info": json.dumps(cells_meta.cell_attribute_info),
            }

        for key, value in cache.items():
            if key.startswith("coordinates_"):
                serialized[key] = value.tolist()

        for key, value in cache.items():
            if key.startswith("projection_"):
                serialized_projections = []
                for post_gid, (pre_gids, projection_data) in value:
                    serialized_projection = {
                        "post_gid": int(post_gid),
                        "pre_gids": pre_gids.tolist(),
                    }

                    if "Connections" in projection_data:
                        connections = [
                            conn.tolist() for conn in projection_data["Connections"]
                        ]
                        serialized_projection["connections"] = json.dumps(connections)

                    if "Synapses" in projection_data:
                        synapses = [syn.tolist() for syn in projection_data["Synapses"]]
                        serialized_projection["synapses"] = json.dumps(synapses)

                    serialized_projections.append(serialized_projection)

                serialized[key] = json.dumps(serialized_projections)

        return serialized

    def _deserialize_cache(self, df):
        """Convert parquet data back to cache format"""
        if len(df) == 0:
            return {}

        row = df.iloc[0]
        deserialized = {}

        if "cells_meta_data" in row:
            cells_meta = row["cells_meta_data"]
            deserialized["cells_meta_data"] = CellsMetaData(
                population_names=cells_meta["population_names"],
                population_ranges=json.loads(cells_meta["population_ranges"]),
                cell_attribute_info=json.loads(cells_meta["cell_attribute_info"]),
            )

        for key in row.index:
            if key.startswith("coordinates_"):
                deserialized[key] = np.array([np.array(d) for d in row[key]])

        for key in row.index:
            if key.startswith("projection_"):
                projections_data = json.loads(row[key])
                deserialized_projections = []

                for proj in projections_data:
                    post_gid = proj["post_gid"]
                    pre_gids = np.array(proj["pre_gids"], dtype=np.uint32)

                    projection_data = {}
                    if "connections" in proj:
                        connections = json.loads(proj["connections"])
                        projection_data["Connections"] = [
                            np.array(conn, dtype=np.float32) for conn in connections
                        ]

                    if "synapses" in proj:
                        synapses = json.loads(proj["synapses"])
                        projection_data["Synapses"] = [
                            np.array(syn, dtype=np.uint32) for syn in synapses
                        ]

                    deserialized_projections.append(
                        [post_gid, (pre_gids, projection_data)]
                    )

                deserialized[key] = deserialized_projections

        return deserialized

    @property
    def cells_meta_data(self):
        return self._cache["cells_meta_data"]

    def projection_array(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        all: bool = True,
    ) -> list[tuple[int, tuple[list[int], Projection]]]:
        return self._cache[f"projection_{pre}_{post}"]

    def coordinate_array(
        self,
        population: types.PopulationName,
        all: bool = True,
    ) -> types.Float[types.Array, "n_coords cxyz=4"]:
        return self._cache[f"coordinates_{population}"]

    def connectivity_matrix(
        self, weights: dict | None = None, reduced: bool = False, seed=123
    ) -> types.Float[types.Array, "num_neurons num_neurons"]:
        # use numpy, not jax
        import numpy as npn

        weights_str = json.dumps(weights or {}, sort_keys=True)
        hash_input = f"{weights_str}_{int(reduced)}_{seed}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()

        cache_file = self._graph.local_directory(
            f"connectivity_matrix_{hash_value}.parquet"
        )

        rank = 0
        if self.comm:
            rank = self.comm.Get_rank()

        matrix = None
        if rank == 0:
            try:
                if os.path.exists(cache_file):
                    df = pd.read_parquet(cache_file)
                    matrix = df["connectivity_matrix"][0]
            except Exception as e:
                print(f"Error loading cached connectivity matrix: {e}")
                matrix = None

            if matrix is None:
                # build cache
                matrix = super().connectivity_matrix(
                    weights=weights, reduced=reduced, seed=seed
                )

                try:
                    df = pd.DataFrame([{"connectivity_matrix": matrix.tolist()}])
                    df.to_parquet(cache_file, index=False)
                except Exception as e:
                    print(f"Error saving connectivity matrix to cache: {e}")

        if self.comm:
            matrix = self.comm.bcast(matrix, root=0)

        return npn.array(
            [npn.array(m, dtype=npn.float32) for m in matrix],
            dtype=npn.float32,
        )
