from __future__ import annotations

import json
import os
import pathlib
import random
import numpy
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Any

from pydantic import BaseModel
import pyfive

from livn import types
from livn.backend import backend
from livn.utils import download_directory, sentinel, load_file, import_object_by_path

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

_H5_BACKEND = "pyfive"  # default

try:
    import neuroh5.io  # noqa: F401
    from mpi4py import MPI as _MPI  # noqa: F401

    _H5_BACKEND = "neuroh5"
except ImportError:
    pass

_HSDS_CONFIG = None
if os.environ.get("LIVN_HSDS"):
    try:
        import h5pyd  # noqa: F401

        _HSDS_CONFIG = json.loads(os.environ["LIVN_HSDS"])
        _H5_BACKEND = "h5pyd"
    except (ImportError, json.JSONDecodeError):
        pass

_HAS_NEUROH5 = _H5_BACKEND == "neuroh5"


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


def fetch(
    source: str,
    directory: str = ".",
    name: str | None = None,
    force: bool = False,
    comm: Optional["MPI.Intracomm"] = None,
):
    target = None
    if comm is None:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        except ImportError:
            comm = None
    if comm is not None and comm.Get_rank() != 0:
        # await download on 0
        return comm.bcast(target)

    if name is None:
        import fsspec

        parsed = fsspec.utils.infer_storage_options(source).get("path", "")
        name = os.path.basename(parsed.rstrip("/"))
        if not name:
            raise ValueError("Could not infer system name from source")

    target = os.path.join(directory, "systems", "graphs", name)

    if force or not os.path.isdir(target):
        download_directory(source, target, force=force)

    if comm is not None:
        comm.bcast(target)

    return target


def predefined(name: str = "EI1", download_directory: str = ".", force: bool = False):
    available = (
        [f"EI{s + 1}" for s in range(4)]
        + [f"S{s + 1}" for s in range(4)]
        + ["CA1", "CA1d"]
    )

    if name not in available:
        raise ValueError(f"'{name}' is invalid, pick one of ", available)

    return fetch(
        source=f"hf://datasets/livn-org/livn/systems/graphs/{name}",
        directory=download_directory,
        name=name,
        force=force,
    )


def make(name: str = "EI1") -> "System":
    system = predefined(name)

    return System(system)


# --- Generic H5 readers (backend-agnostic) ---


def _h5_read_population_names(f):
    """Read population names from an open H5 file object."""
    return list(f["Populations"].keys())


def _h5_population_ranges(f, pop_names):
    pops_data = f["H5Types/Populations"][:]
    ranges = {}
    for name, row in zip(pop_names, pops_data):
        ranges[name] = (int(row[0]), int(row[1]))
    return ranges


def _h5_read_population_ranges(f):
    pop_names = _h5_read_population_names(f)
    return _h5_population_ranges(f, pop_names)


def _h5_read_cell_attribute_info(f, population_names):
    result = {}
    for pop_name in population_names:
        pop_group = f[f"Populations/{pop_name}"]
        namespaces = {}
        for ns_name in pop_group.keys():
            ns_group = pop_group[ns_name]
            if hasattr(ns_group, "keys"):
                namespaces[ns_name] = sorted(ns_group.keys())
        result[pop_name] = namespaces
    return result


def _h5_read_cell_attributes(f, pop_start, population, namespace, mask=None):
    ns_group = f[f"Populations/{population}/{namespace}"]
    attr_names = list(ns_group.keys())
    if mask is not None:
        attr_names = [a for a in attr_names if a in mask]

    attrs_data = {}
    cell_index = None
    for attr_name in attr_names:
        attr_group = ns_group[attr_name]
        if cell_index is None:
            cell_index = attr_group["Cell Index"][:]
        pointer = attr_group["Attribute Pointer"][:]
        value = attr_group["Attribute Value"][:]
        attrs_data[attr_name] = (pointer, value)

    if cell_index is None:
        return {}

    result = {}
    for i, rel_gid in enumerate(cell_index):
        abs_gid = int(rel_gid) + pop_start
        cell_attrs = {}
        for attr_name, (pointer, value) in attrs_data.items():
            start = int(pointer[i])
            end = int(pointer[i + 1])
            cell_attrs[attr_name] = value[start:end]
        result[abs_gid] = cell_attrs

    return result


def _h5_read_cell_attributes_tuple(f, pop_start, population, namespace):
    ns_group = f[f"Populations/{population}/{namespace}"]
    attr_names = sorted(ns_group.keys())
    attr_info = {name: idx for idx, name in enumerate(attr_names)}

    attrs_data = {}
    cell_index = None
    for attr_name in attr_names:
        attr_group = ns_group[attr_name]
        if cell_index is None:
            cell_index = attr_group["Cell Index"][:]
        pointer = attr_group["Attribute Pointer"][:]
        value = attr_group["Attribute Value"][:]
        attrs_data[attr_name] = (pointer, value)

    if cell_index is None:
        return [], {}

    items = []
    for i, rel_gid in enumerate(cell_index):
        abs_gid = int(rel_gid) + pop_start
        values = []
        for attr_name in attr_names:
            pointer, value = attrs_data[attr_name]
            start = int(pointer[i])
            end = int(pointer[i + 1])
            values.append(value[start:end])
        items.append((abs_gid, tuple(values)))

    return items, attr_info


def _h5_read_graph(f, pre_start, post_start, pre, post, namespaces=None):
    if namespaces is None:
        namespaces = []

    proj_group = f[f"Projections/{post}/{pre}"]
    edges = proj_group["Edges"]
    dest_block_index = edges["Destination Block Index"][:]
    dest_block_pointer = edges["Destination Block Pointer"][:]
    dest_pointer = edges["Destination Pointer"][:]
    source_index = edges["Source Index"][:]

    ns_data_arrays = {}
    for ns_name in namespaces:
        if ns_name in proj_group:
            ns_group = proj_group[ns_name]
            ns_data_arrays[ns_name] = {}
            for ds_name in ns_group.keys():
                ds = ns_group[ds_name]
                if hasattr(ds, "shape"):
                    ns_data_arrays[ns_name][ds_name] = ds[:]

    results = []
    for block_idx in range(len(dest_block_index)):
        block_start_gid = int(dest_block_index[block_idx])
        ptr_start = int(dest_block_pointer[block_idx])
        ptr_end = int(dest_block_pointer[block_idx + 1])
        n_dest = ptr_end - ptr_start - 1

        for d in range(n_dest):
            rel_dest_gid = block_start_gid + d
            abs_dest_gid = rel_dest_gid + post_start

            edge_start = int(dest_pointer[ptr_start + d])
            edge_end = int(dest_pointer[ptr_start + d + 1])

            pre_gids = (
                source_index[edge_start:edge_end].astype(numpy.uint32) + pre_start
            )

            ns_data = {}
            for ns_name in namespaces:
                if ns_name in ns_data_arrays:
                    ns_data[ns_name] = [
                        arr[edge_start:edge_end]
                        for arr in ns_data_arrays[ns_name].values()
                    ]

            results.append((abs_dest_gid, (pre_gids, ns_data)))

    return results


def _pyfive_open(filepath):
    return pyfive.File(filepath)


def _h5pyd_open(filepath):
    import h5pyd

    hsds_path = _to_hsds_domain(filepath)
    return h5pyd.File(
        hsds_path,
        mode="r",
        endpoint=_HSDS_CONFIG["endpoint"],
        bucket=_HSDS_CONFIG.get("bucket", "/data"),
    )


def _to_hsds_domain(filepath):
    """Map a local file path to an HSDS domain path.

    The server's root_dir is systems/graphs/, so a local path like
    .../systems/graphs/EI1/graph.h5 maps to the HSDS domain /EI1/graph.h5.
    Relative paths like EI1/graph.h5 are used directly.
    """
    parts = pathlib.PurePosixPath(filepath).parts
    try:
        idx = parts.index("graphs")
        return "/" + "/".join(parts[idx + 1 :])
    except ValueError:
        return "/" + filepath.lstrip("/")


def _open_h5(filepath):
    if _H5_BACKEND == "h5pyd":
        return _h5pyd_open(filepath)
    return _pyfive_open(filepath)


if _H5_BACKEND == "neuroh5":

    def read_cells_meta_data(
        filepath: str, comm: Optional["MPI.Intracomm"] = None
    ) -> CellsMetaData:
        from mpi4py import MPI
        from neuroh5.io import (
            read_cell_attribute_info,
            read_population_names,
            read_population_ranges,
        )

        if comm is None:
            comm = MPI.COMM_WORLD

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
        population_ranges: dict[str, tuple[int, int]] | None = None,
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
        population_ranges: dict[str, tuple[int, int]] | None = None,
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

else:  # h5pyd or pyfive — both use _open_h5 + generic readers

    def read_cells_meta_data(
        filepath: str, comm: Optional["MPI.Intracomm"] = None
    ) -> CellsMetaData:
        f = _open_h5(filepath)
        population_names = _h5_read_population_names(f)
        population_ranges = _h5_read_population_ranges(f)
        cell_attribute_info = _h5_read_cell_attribute_info(f, population_names)

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
        f = _open_h5(filepath)
        pop_ranges = _h5_read_population_ranges(f)
        pop_start = pop_ranges[population][0]
        items, attr_info = _h5_read_cell_attributes_tuple(
            f, pop_start, population, "Generated Coordinates"
        )
        x_index = attr_info.get("X Coordinate", None)
        y_index = attr_info.get("Y Coordinate", None)
        z_index = attr_info.get("Z Coordinate", None)
        for gid, cell_coords in items:
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
        coordinates = []
        for gid, coordinate in read_coordinates(filepath, population):
            coordinates.append([gid] + list(coordinate))
        coordinates = np.array(coordinates)
        if coordinates.size == 0:
            return np.zeros((0, 4))
        return coordinates[coordinates[:, 0].argsort()]

    def read_trees(
        filepath: str,
        population: types.PopulationName,
        comm: Optional["MPI.Intracomm"] = None,
    ) -> Iterator[tuple[int, Tree]]:
        raise NotImplementedError(
            "read_trees requires neuroh5; no pyfive fallback available"
        )

    def read_synapses(
        filepath: str,
        population: types.PostSynapticPopulationName,
        comm: Optional["MPI.Intracomm"] = None,
        node_allocation: set[int] | None = None,
    ):
        mask = {
            "syn_ids",
            "syn_locs",
            "syn_secs",
            "syn_layers",
            "syn_types",
            "swc_types",
        }
        f = _open_h5(filepath)
        pop_ranges = _h5_read_population_ranges(f)
        pop_start = pop_ranges[population][0]
        attrs = _h5_read_cell_attributes(
            f, pop_start, population, "Synapse Attributes", mask=mask
        )
        for gid in sorted(attrs.keys()):
            if node_allocation is not None and gid not in node_allocation:
                continue
            yield gid, attrs[gid]

    def read_projections(
        filepath: str,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        comm: Optional["MPI.Intracomm"] = None,
        population_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> Iterator[tuple[int, tuple[list[int], Projection]]]:
        f = _open_h5(filepath)
        if population_ranges is None:
            population_ranges = _h5_read_population_ranges(f)
        pre_start = population_ranges[pre][0]
        post_start = population_ranges[post][0]
        results = _h5_read_graph(
            f,
            pre_start,
            post_start,
            pre,
            post,
            namespaces=["Synapses", "Connections"],
        )
        yield from results

    def projection_array(
        filepath: str,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        comm: Optional["MPI.Intracomm"] = None,
        all: bool = True,
        population_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> list[tuple[int, tuple[list[int], Projection]]]:
        projections = []
        for post_gid, (pre_gids, projection) in read_projections(
            filepath, pre, post, population_ranges=population_ranges
        ):
            projections.append([post_gid, (pre_gids, projection)])
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
        if _H5_BACKEND == "h5pyd" and _HSDS_CONFIG:
            return self._load_elements_http()
        return self._load_elements_local()

    def _load_elements_local(self):
        with open(self.local_directory("graph.json")) as f:
            graph = json.load(f)
        return self._parse_elements(graph)

    def _load_elements_http(self):
        import urllib.request

        endpoint = _HSDS_CONFIG["endpoint"]
        port = int(endpoint.rsplit(":", 1)[1]) + 1
        host = endpoint.rsplit(":", 1)[0]
        system_name = os.path.basename(self.directory)
        url = f"{host}:{port}/files/{system_name}/graph.json"
        with urllib.request.urlopen(url) as resp:
            graph = json.loads(resp.read())
        return self._parse_elements(graph)

    @staticmethod
    def _parse_elements(graph):
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
        self._num_neurons = None
        self._bounding_box = None

    def default_io(self, comm=None) -> "IO":
        from livn.io import MEA, IO

        try:
            return MEA.from_directory(self.uri, comm=comm)
        except FileNotFoundError:
            return IO()

    def default_model(self, comm=None) -> "Model":
        model = self.load_file("model.json", None, comm=comm)
        if model is not None:
            model = import_object_by_path(model["cls"])(**model["kwargs"])
        else:
            from livn.models.rcsd import ReducedCalciumSomaDendrite

            model = ReducedCalciumSomaDendrite()

        return model

    def default_params(self, comm=None) -> dict | None:
        return self.load_file("params.json", None, comm=comm)

    def load_file(
        self,
        filepath: str | list[str],
        default: Any = sentinel,
        **kwargs,
    ):
        if isinstance(filepath, str):
            filepath = [filepath]
        return load_file(
            [self._graph.local_directory()] + list(filepath), default, **kwargs
        )

    @property
    def bounding_box(self) -> types.Float[types.Array, "2 xyz=3"]:
        if self._bounding_box is None:
            min_box = [1e10, 1e10, 1e10]
            max_box = [0.0, 0.0, 0.0]
            for box in self._graph.architecture.config["layer_extents"].values():
                for i in range(3):
                    if box[0][i] < min_box[i]:
                        min_box[i] = box[0][i]
                    if box[1][i] > max_box[i]:
                        max_box[i] = box[1][i]

            self._bounding_box = np.array([min_box, max_box])

        return self._bounding_box

    @property
    def center_point(self) -> types.Float[types.Array, "xyz=3"]:
        bb = self.bounding_box
        return np.array([(bb[1][i] - bb[0][i]) / 2.0 for i in range(3)])

    @property
    def name(self):
        return self.uri.split("/")[-1]

    @property
    def weight_names(self) -> list[str]:
        weight_names = []

        for post_pop, pre_connections in self.connections_config["synapses"].items():
            for pre_pop, synapse_config in pre_connections.items():
                if backend() != "neuron":
                    weight_names.append(f"{pre_pop}_{post_pop}")
                else:
                    sections = synapse_config.get("sections", [])
                    mechanisms = synapse_config.get("mechanisms", {}).get("default", {})
                    for section in sections:
                        for mech_name in mechanisms.keys():
                            weight_names.append(
                                f"{pre_pop}_{post_pop}-{section}-{mech_name}-weight"
                            )

        return weight_names

    @property
    def num_neurons(self):
        if self._num_neurons is None:
            self._num_neurons = sum(
                [
                    self.cells_meta_data.population_count(population)
                    for population in self.populations
                ]
            )

        return self._num_neurons

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
            return np.asarray(self.neuron_coordinates[:, 0], dtype=int)

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

    def transform_coordinates(
        self,
        transform: Callable,
        populations: list[str] | None = None,
        all: bool = True,
    ) -> types.Float[types.Array, "n_coords ixyz=4"]:
        if populations is None:
            populations = self.populations
        return np.vstack(
            [
                transform(self.coordinate_array(p, all=all), population=p)
                for p in populations
            ]
        )

    def projections(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
    ) -> Iterator[tuple[int, tuple[list[int], Projection]]]:
        yield from read_projections(
            self._graph.connections_filepath,
            pre,
            post,
            comm=self.comm,
            population_ranges=self.cells_meta_data.population_ranges,
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
            self._graph.connections_filepath,
            pre,
            post,
            comm=self.comm,
            all=all,
            population_ranges=self.cells_meta_data.population_ranges,
        )

    def connectivity_matrix(
        self, weights: dict | None = None, seed=123
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
                    # distances = projection
                    # if isinstance(projection, dict):
                    #     distances = projection["Connections"][0]

                    for pre_gid in pre_gids:
                        w[pre_gid, post_gid] = prefix * prng.random() * weight

        return w

    def summary(self) -> dict[str, int | dict[str, int]]:
        num_neurons = 0
        num_projections = 0
        population_counts = {}

        for population in self.populations:
            count = self.cells_meta_data.population_count(population)
            population_counts[population] = count
            num_neurons += count

        for post, v in self.connections_config["synapses"].items():
            for pre, _ in v.items():
                for _, (pre_gids, _) in self.projection_array(pre, post):
                    num_projections += len(pre_gids)

        return {
            "num_neurons": num_neurons,
            "num_projections": num_projections,
            "population_counts": population_counts,
        }


class TrainableSystem:
    """System with trainable neuron positions using offset-based
    parameterization where neuron coords = init_coords + params

    Arguments:
        n_neurons: Number of neurons per population
        n_populations: Number of populations
        params: Learnable offsets from origins [n_populations, n_neurons, xyz=3].
            If None or dict, initialized using physics-informed radial distribution
            around each origin. A dict is passed as kwargs to _init_radial_offsets,
            e.g. params={'max_dist': 13} to constrain the initial radius.
        origins: Reference origins [n_populations, xyz=3] for channel locations;
            defaults to vec{0} for all populations
        uri: Optional URI for loading IO configuration
        seed:
    """

    def __init__(
        self,
        n_neurons: int,
        n_populations: int = 1,
        params: types.Float[types.Array, "n_populations n_neurons xyz=3"]
        | dict
        | None = None,
        origins: types.Float[types.Array, "n_populations xyz=3"] | None = None,
        uri: str | None = None,
        seed: int | None = None,
    ):
        self.n_neurons = n_neurons
        self.n_populations = n_populations
        self.name = "TrainableSystem"
        self.populations = [str(i) for i in range(n_populations)]
        self.uri = uri

        if origins is None:
            origins = np.tile(np.array([0.0, 0.0, 0.0]), (n_populations, 1))
        self.origins = np.array(origins)

        if params is None:
            params = {}
        if isinstance(params, dict):
            params = self._init_radial_offsets(seed=seed, **params)
        self.set_params(params)

    def _init_radial_offsets(
        self,
        seed: int | None = None,
        sigma: float = 75.0,
        min_dist: float = 10.0,
        max_dist: float = 100.0,
    ) -> types.Float[types.Array, "n_populations n_neurons xyz=3"]:
        """Initialize neuron offsets with radial distribution from electrode origins.

        Neurons are placed around each electrode origin using a truncated Gaussian
        distribution, reflecting the typical recording radius of MEA electrodes

        Args:
            seed:
            sigma: Standard deviation of Gaussian
            min_dist:
            max_dist:
        """
        rng = numpy.random.default_rng(seed)

        offsets = numpy.zeros((self.n_populations, self.n_neurons, 3))

        for pop_idx in range(self.n_populations):
            for neuron_idx in range(self.n_neurons):
                while True:
                    r = abs(rng.normal(0, sigma))
                    if min_dist <= r <= max_dist:
                        break

                theta = rng.uniform(0, 2 * numpy.pi)
                phi = numpy.arccos(rng.uniform(-1, 1))

                x = r * numpy.sin(phi) * numpy.cos(theta)
                y = r * numpy.sin(phi) * numpy.sin(theta)
                z = r * numpy.cos(phi)

                offsets[pop_idx, neuron_idx] = numpy.array([x, y, z])

        return offsets

    def set_params(
        self,
        params: Optional[
            types.Float[types.Array, "n_populations n_neurons xyz=3"]
        ] = None,
    ):
        if params is None:
            params = np.zeros((self.n_populations, self.n_neurons, 3))

        self.params = np.array(params)
        return params

    def set_params_from_coordinates(
        self,
        neuron_coordinates: types.Float[types.Array, "n_coords ixyz=4"],
    ):
        coords = np.array(neuron_coordinates)

        total_neurons = len(coords)
        n_neurons_per_population = total_neurons // self.n_populations

        if total_neurons % self.n_populations != 0:
            total_neurons = n_neurons_per_population * self.n_populations
            coords = coords[:total_neurons]

        self.n_neurons = n_neurons_per_population

        # [n_populations, n_neurons, xyz=3]
        xyz = coords[:, 1:4]  # drop gids
        absolute_coords = xyz.reshape(self.n_populations, n_neurons_per_population, 3)

        # params = absolute_coords - origins (offsets from electrode)
        offsets = absolute_coords - self.origins[:, None, :]

        return self.set_params(offsets)

    def default_io(self) -> "IO":
        from livn.io import MEA

        if hasattr(self, "uri") and self.uri is not None:
            try:
                return MEA.from_directory(self.uri)
            except (FileNotFoundError, AttributeError):
                pass

        return MEA()

    @property
    def num_neurons(self):
        return self.n_populations * self.n_neurons

    @property
    def neuron_coordinates(self) -> types.Float[types.Array, "n_total_neurons ixyz=4"]:
        absolute_coords = (
            self.origins[:, None, :] + self.params
        )  # [n_pop, n_neurons, 3]

        xyz_flat = absolute_coords.reshape(-1, 3)  # [n_total_neurons, 3]

        # [n_total_neurons, 1]
        n_total = self.n_populations * self.n_neurons
        gids = np.arange(n_total, dtype=xyz_flat.dtype).reshape(-1, 1)

        # [n_total_neurons, ixyz=4]
        return np.concatenate([gids, xyz_flat], axis=1)

    def coordinate_array(
        self, population: str, all: bool = True
    ) -> types.Float[types.Array, "n_coords cxyz=4"]:
        pop_idx = self.populations.index(population)
        absolute_coords = self.origins[pop_idx] + self.params[pop_idx]  # [n_neurons, 3]
        gid_offset = pop_idx * self.n_neurons
        gids = np.arange(
            gid_offset, gid_offset + self.n_neurons, dtype=absolute_coords.dtype
        ).reshape(-1, 1)
        return np.concatenate([gids, absolute_coords], axis=1)

    def transform_coordinates(
        self,
        transform: Callable,
        populations: list[str] | None = None,
        all: bool = True,
    ) -> types.Float[types.Array, "n_coords ixyz=4"]:
        if populations is None:
            populations = self.populations
        return np.vstack(
            [
                transform(self.coordinate_array(p, all=all), population=p)
                for p in populations
            ]
        )

    @property
    def gids(self) -> types.Int[types.Array, "n_total_neurons"]:
        return np.arange(self.n_populations * self.n_neurons, dtype=np.int32)

    @property
    def bounding_box(self) -> types.Float[types.Array, "2 xyz=3"]:
        coords = self.neuron_coordinates[:, 1:4]
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        padding = 100.0
        min_coords = min_coords - padding
        max_coords = max_coords + padding

        return np.stack([min_coords, max_coords])

    @property
    def center_point(self) -> types.Float[types.Array, "xyz=3"]:
        bb = self.bounding_box
        return (bb[0] + bb[1]) / 2.0


if _USES_JAX:
    import jax

    def _trainable_system_flatten(system):
        children = (system.params, system.origins)
        aux = (
            system.n_neurons,
            system.n_populations,
            system.name,
            tuple(system.populations),
            system.uri,
        )
        return children, aux

    def _trainable_system_unflatten(aux, children):
        params, origins = children
        n_neurons, n_populations, name, populations, uri = aux

        system = object.__new__(TrainableSystem)
        system.n_neurons = n_neurons
        system.n_populations = n_populations
        system.origins = origins
        system.name = name
        system.populations = list(populations)  # Convert back to list
        system.uri = uri
        system.params = params
        return system

    jax.tree_util.register_pytree_node(
        TrainableSystem, _trainable_system_flatten, _trainable_system_unflatten
    )
