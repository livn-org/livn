from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import random
import warnings
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy
import pyfive

from livn import types
from livn.backend import backend
from livn.system._common import (
    CellsMetaData,
    Element,
    Projection,
    Tree,
    _placement_rows,
    resolve_selection,
)
from livn.utils import (
    P,
    import_object_by_path,
    load_file,
    sentinel,
)

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.types import Model

logger = logging.getLogger(__name__)

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


def _h5_read_population_names(f):
    """Read population names from an open H5 file object."""
    return list(f["Populations"].keys())


def _h5_population_ranges(f, pop_names):
    pops_data = f["H5Types/Populations"][:]
    ranges = {}
    for name, row in zip(pop_names, pops_data, strict=False):
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
        for ns_name in pop_group:
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


def stored_projections(f) -> dict[str, list[str]]:
    if "Projections" not in f:
        return {}
    return {post: list(f[f"Projections/{post}"]) for post in f["Projections"]}


def require_projection(f, filepath, pre, post) -> None:
    stored = stored_projections(f)
    if pre in stored.get(post, ()):
        return

    available = sorted(f"{p}->{q}" for q, sources in stored.items() for p in sources)
    raise KeyError(
        f"{filepath} has no {pre}->{post} projection; it has "
        f"{', '.join(available) if available else 'none'}"
    )


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
            for ds_name in ns_group:
                ds = ns_group[ds_name]
                if hasattr(ds, "shape"):
                    ns_data_arrays[ns_name][ds_name] = ds[:]

    results = []
    last_block = len(dest_block_index) - 1
    for block_idx in range(len(dest_block_index)):
        block_start_gid = int(dest_block_index[block_idx])
        ptr_start = int(dest_block_pointer[block_idx])
        ptr_end = int(dest_block_pointer[block_idx + 1])
        n_dest = ptr_end - ptr_start - (1 if block_idx == last_block else 0)

        for d in range(n_dest):
            rel_dest_gid = block_start_gid + d
            abs_dest_gid = rel_dest_gid + post_start

            edge_start = int(dest_pointer[ptr_start + d])
            edge_end = int(dest_pointer[ptr_start + d + 1])

            if edge_end == edge_start:
                continue

            pre_gids = (
                source_index[edge_start:edge_end].astype(numpy.uint32) + pre_start
            )

            ns_data = {}
            for ns_name in namespaces:
                if ns_name in ns_data_arrays:
                    ns_data[ns_name] = {
                        ds_name: arr[edge_start:edge_end]
                        for ds_name, arr in ns_data_arrays[ns_name].items()
                    }

            results.append((abs_dest_gid, (pre_gids, ns_data)))

    return results


def _pyfive_open(filepath):
    return pyfive.File(filepath)


def _h5pyd_open(filepath):
    import h5pyd

    hsds_config = None
    if os.environ.get("LIVN_HSDS"):
        with contextlib.suppress(json.JSONDecodeError):
            hsds_config = json.loads(os.environ["LIVN_HSDS"])
    if hsds_config is None:
        hsds_config = _HSDS_CONFIG
    if hsds_config is None:
        raise RuntimeError("HSDS not configured")

    hsds_path = _to_hsds_domain(filepath)
    kwargs = {
        "mode": "r",
        "endpoint": hsds_config["endpoint"],
    }
    if hsds_config.get("bucket"):
        kwargs["bucket"] = hsds_config["bucket"]
    if hsds_config.get("username"):
        kwargs["username"] = hsds_config["username"]
    if hsds_config.get("password"):
        kwargs["password"] = hsds_config["password"]
    return h5pyd.File(hsds_path, **kwargs)


def _to_hsds_domain(filepath):
    """Map a local file path to an HSDS domain path."""
    parts = pathlib.PurePosixPath(filepath).parts
    try:
        idx = parts.index("graphs")
        return "/" + "/".join(parts[idx + 1 :])
    except ValueError:
        # Extract system_name/filename from the end of the path
        # e.g. /home/pyodide/EI/cells.h5 -> /EI/cells.h5
        if len(parts) >= 2:
            return "/" + "/".join(parts[-2:])
        return "/" + filepath.lstrip("/")


def _open_h5(filepath):
    # Check dynamically for late-configured HSDS (e.g. Pyodide)
    if _H5_BACKEND == "h5pyd" or os.environ.get("LIVN_HSDS"):
        try:
            return _h5pyd_open(filepath)
        except Exception as e:
            import warnings

            warnings.warn(
                f"h5pyd open failed for {filepath}: {e}, falling back to pyfive",
                stacklevel=2,
            )
    return _pyfive_open(filepath)


if _H5_BACKEND == "neuroh5":

    def read_cells_meta_data(
        filepath: str, comm: MPI.Intracomm | None = None
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
        comm: MPI.Intracomm | None = None,
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
        comm: MPI.Intracomm | None = None,
        all: bool = True,
    ) -> types.Float[types.Array, "n_coords cxyz=4"]:
        from mpi4py import MPI

        if comm is None:
            comm = MPI.COMM_WORLD

        coordinates = []
        for gid, coordinate in read_coordinates(filepath, population, comm=comm):
            coordinates.append([gid, *list(coordinate)])

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
        comm: MPI.Intracomm | None = None,
    ) -> Iterator[tuple[int, Tree]]:
        from mpi4py import MPI
        from neuroh5.io import scatter_read_trees

        if comm is None:
            comm = MPI.COMM_WORLD

        (trees, _forestSize) = scatter_read_trees(filepath, population, comm=comm)
        yield from trees

    def read_synapses(
        filepath: str,
        population: types.PostSynapticPopulationName,
        comm: MPI.Intracomm | None = None,
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
            node_allocation=None,
            io_size=1,
            return_type="dict",
        )

        for gid, attributes in cell_attributes_dict["Synapse Attributes"]:
            if node_allocation is not None and int(gid) not in node_allocation:
                continue
            yield gid, attributes

    def read_projections(
        filepath: str,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        comm: MPI.Intracomm | None = None,
        population_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> Iterator[tuple[int, tuple[list[int], Projection]]]:
        from mpi4py import MPI
        from neuroh5.io import scatter_read_graph

        if comm is None:
            comm = MPI.COMM_WORLD

        require_projection(_open_h5(filepath), filepath, pre, post)

        (graph, _a) = scatter_read_graph(
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
        comm: MPI.Intracomm | None = None,
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

    def read_placement(
        filepath: str,
        population: types.PopulationName,
        gids,
        comm: MPI.Intracomm | None = None,
        io_size: int = 1,
    ) -> dict[int, tuple]:
        from mpi4py import MPI
        from neuroh5.io import scatter_read_cell_attribute_selection

        if comm is None:
            comm = MPI.COMM_WORLD

        # Collective, so issue the read even for an empty selection: a rank that
        # owns no cells of this population still has to participate or the ranks
        # that do will block waiting for it.
        out: dict[int, tuple] = {}
        it, info = scatter_read_cell_attribute_selection(
            filepath,
            population,
            sorted(int(g) for g in gids),
            namespace="Synapse Attributes",
            mask={"syn_ids", "swc_types", "syn_locs"},
            comm=comm,
            io_size=max(1, int(io_size)),
            return_type="tuple",
        )
        i_ids, i_swc, i_loc = (
            info.get("syn_ids"),
            info.get("swc_types"),
            info.get("syn_locs"),
        )
        if i_ids is None:
            return out
        for gid, data in it:
            out[int(gid)] = _placement_rows(data[i_ids], data[i_swc], data[i_loc])
        return out

    def read_edges(
        filepath: str,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        gids,
        comm: MPI.Intracomm | None = None,
        io_size: int = 1,
        destinations=None,
        population_ranges: dict[str, tuple[int, int]] | None = None,
    ):
        del population_ranges  # neuroh5 resolves gids itself
        from mpi4py import MPI
        from neuroh5.io import scatter_read_graph_selection

        if comm is None:
            comm = MPI.COMM_WORLD

        # neuroh5 wants only gids the projection actually has a destination for.
        # `destinations is None` means it stores no edges at all, which is not an
        # error -- the config may declare a projection the graph left empty -- so
        # ask for nothing rather than for gids that cannot be there.
        wanted: list[int] = []
        if destinations is not None:
            asked = sorted(int(g) for g in gids)
            if asked:
                import numpy as npn

                index = npn.fromiter(asked, dtype=npn.int64, count=len(asked))
                wanted = npn.sort(index[npn.isin(index, destinations)]).tolist()

        # Collective in the same way `read_placement` is: the selection may be
        # empty on this rank, but the call may not be skipped.
        graph, _ = scatter_read_graph_selection(
            filepath,
            comm=comm,
            io_size=max(1, int(io_size)),
            selection=wanted,
            projections=[(pre, post)],
            namespaces=["Synapses", "Connections"],
        )
        if post in graph and pre in graph[post]:
            yield from graph[post][pre]

else:  # h5pyd or pyfive — both use _open_h5 + generic readers

    def read_cells_meta_data(
        filepath: str, comm: MPI.Intracomm | None = None
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
        comm: MPI.Intracomm | None = None,
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
        comm: MPI.Intracomm | None = None,
        all: bool = True,
    ) -> types.Float[types.Array, "n_coords cxyz=4"]:
        coordinates = []
        for gid, coordinate in read_coordinates(filepath, population):
            coordinates.append([gid, *list(coordinate)])
        coordinates = np.array(coordinates)
        if coordinates.size == 0:
            return np.zeros((0, 4))
        return coordinates[coordinates[:, 0].argsort()]

    def read_trees(
        filepath: str,
        population: types.PopulationName,
        comm: MPI.Intracomm | None = None,
    ) -> Iterator[tuple[int, Tree]]:
        raise NotImplementedError(
            "read_trees requires neuroh5; no pyfive fallback available"
        )

    def read_synapses(
        filepath: str,
        population: types.PostSynapticPopulationName,
        comm: MPI.Intracomm | None = None,
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
        comm: MPI.Intracomm | None = None,
        population_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> Iterator[tuple[int, tuple[list[int], Projection]]]:
        f = _open_h5(filepath)
        require_projection(f, filepath, pre, post)
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
        comm: MPI.Intracomm | None = None,
        all: bool = True,
        population_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> list[tuple[int, tuple[list[int], Projection]]]:
        projections = []
        for post_gid, (pre_gids, projection) in read_projections(
            filepath, pre, post, population_ranges=population_ranges
        ):
            projections.append([post_gid, (pre_gids, projection)])
        return projections

    def read_placement(
        filepath: str,
        population: types.PopulationName,
        gids,
        comm: MPI.Intracomm | None = None,
        io_size: int = 1,
    ) -> dict[int, tuple]:
        # No scatter read here: every rank opens the file and keeps the rows it
        # was asked for. Correct, and fine up to the scale at which neuroh5 is
        # worth installing.
        wanted = {int(g) for g in gids}
        out: dict[int, tuple] = {}
        f = _open_h5(filepath)
        pop_start = _h5_read_population_ranges(f)[population][0]
        attrs = _h5_read_cell_attributes(
            f,
            pop_start,
            population,
            "Synapse Attributes",
            mask={"syn_ids", "swc_types", "syn_locs"},
        )
        for gid, data in attrs.items():
            if int(gid) not in wanted:
                continue
            out[int(gid)] = _placement_rows(
                data["syn_ids"], data["swc_types"], data["syn_locs"]
            )
        return out

    def read_edges(
        filepath: str,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        gids,
        comm: MPI.Intracomm | None = None,
        io_size: int = 1,
        destinations=None,
        population_ranges: dict[str, tuple[int, int]] | None = None,
    ):
        wanted = {int(g) for g in gids}
        for post_gid, payload in read_projections(
            filepath, pre, post, population_ranges=population_ranges
        ):
            if int(post_gid) in wanted:
                yield post_gid, payload


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

    @staticmethod
    def _get_hsds_config():
        """Re-read HSDS config from env (handles late configuration in Pyodide)"""
        if os.environ.get("LIVN_HSDS"):
            try:
                import h5pyd  # noqa: F401

                return json.loads(os.environ["LIVN_HSDS"])
            except (ImportError, json.JSONDecodeError):
                pass
        return None

    @cached_property
    def elements(self):
        hsds_config = self._get_hsds_config()
        if hsds_config:
            try:
                return self._load_elements_http(hsds_config)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"HSDS elements load failed: {e}, falling back to local",
                    stacklevel=2,
                )
        return self._load_elements_local()

    def _load_elements_local(self):
        with open(self.local_directory("graph.json")) as f:
            graph = json.load(f)
        return self._parse_elements(graph)

    def _load_elements_http(self, hsds_config=None):
        config = hsds_config or _HSDS_CONFIG
        # Use explicit files_endpoint if provided (e.g. Vite proxy)
        files_endpoint = config.get("files_endpoint")
        if files_endpoint:
            system_name = os.path.basename(self.directory)
            url = f"{files_endpoint}/{system_name}/graph.json"
        else:
            import urllib.parse

            endpoint = config["endpoint"]
            parsed = urllib.parse.urlparse(endpoint)
            if not parsed.port:
                raise ValueError(
                    f"Cannot derive file-server port from endpoint: {endpoint}"
                )
            file_port = parsed.port + 1
            file_host = f"{parsed.scheme}://{parsed.hostname}:{file_port}"
            system_name = os.path.basename(self.directory)
            url = f"{file_host}/files/{system_name}/graph.json"

        # In Pyodide, urllib doesn't work (no real sockets); use pyodide.http
        try:
            from pyodide.http import open_url

            graph = json.loads(open_url(url).read())
        except ImportError:
            import urllib.request

            with urllib.request.urlopen(url) as resp:
                graph = json.loads(resp.read())
        return self._parse_elements(graph)

    @staticmethod
    def _parse_elements(graph):
        def _load_element(model):
            if "uuid" not in model:
                return {k: _load_element(v) for k, v in model.items()}
            return Element(**model)

        for k in [k for k, v in graph.items() if isinstance(v, dict)]:
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
    def connections(self):
        return self.elements["connections"]

    @property
    def version(self) -> int:
        return int(self.elements.get("version", 0))

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


class NeuroH5System:
    """In vitro system"""

    GRAPH_FORMAT_VERSION = 1

    def __init__(self, uri: str, comm: MPI.Intracomm | None = None, io_size: int = 1):
        self.uri = uri
        self.comm = comm
        # How many ranks neuroh5 reads through. A backend that knows its rank
        # count raises this before building (the NEURON env sets `pc.nhost()`);
        # it changes throughput, never what is read.
        self.io_size = max(1, int(io_size))

        self._graph = NeuroH5Graph(uri)
        self._check_format_version()
        self._cells_meta_data = None
        self.connections_config = next(iter(self._graph.connections.values())).config
        self.files = self._graph.files()
        self._neuron_coordinates = None
        self._num_neurons = None
        self._bounding_box = None
        self._coordinate_arrays: dict[types.PopulationName, Any] = {}
        self._destination_indices: dict[tuple[str, str], Any] = {}

    def serialize(self) -> dict:
        """The directory to read this system back from.

        Relative when it sits below the env file that will carry it, so a
        directory can be moved or shared without rewriting what is inside it.
        """
        return {"uri": self.uri}

    def _check_format_version(self) -> None:
        found = self._graph.version
        if found > self.GRAPH_FORMAT_VERSION:
            raise ValueError(
                f"{self.uri!r} is a v{found} graph, but this version understands "
                f"v{self.GRAPH_FORMAT_VERSION}. Upgrade livn to read it"
            )
        if found < self.GRAPH_FORMAT_VERSION:
            warnings.warn(
                f"{self.uri!r} predates the v{self.GRAPH_FORMAT_VERSION} graph; re-download to get the latest version.",
                stacklevel=3,
            )

    def default_io(self, comm=None) -> IO:
        from livn.io import IO, MEA

        # Try local file first, then HTTP endpoint
        try:
            data = self.load_file("mea.json", comm=comm)
            if data is not None:
                return MEA.from_json(data, comm=comm)
        except Exception:
            pass
        try:
            data = self._load_json_file("mea.json")
            return MEA.from_json(data)
        except Exception:
            pass
        try:
            return MEA.from_directory(self.uri, comm=comm)
        except Exception:
            return IO()

    def default_model(self, comm=None) -> Model:
        model = self.load_file("model.json", None, comm=comm)
        if model is None:
            with contextlib.suppress(Exception):
                model = self._load_json_file("model.json")
        if model is not None and "cls" in model:
            model = import_object_by_path(model["cls"])(**model["kwargs"])
        else:
            from livn.models.rcsd import ReducedCalciumSomaDendrite

            model = ReducedCalciumSomaDendrite()

        return model

    def _load_json_file(self, filename):
        """Load a JSON file, trying HTTP (files_endpoint) first for Pyodide"""
        hsds_config = self._graph._get_hsds_config()
        if hsds_config and hsds_config.get("files_endpoint"):
            files_endpoint = hsds_config["files_endpoint"]
            system_name = os.path.basename(self._graph.directory)
            url = f"{files_endpoint}/{system_name}/{filename}"
            try:
                from pyodide.http import open_url

                resp = open_url(url)
                text = resp.read()
                data = json.loads(text)
                if isinstance(data, dict) and "error" in data:
                    raise FileNotFoundError(data["error"])
                return data
            except ImportError:
                import urllib.request

                with urllib.request.urlopen(url) as resp:
                    return json.loads(resp.read())
        raise FileNotFoundError(f"No HTTP endpoint for {filename}")

    def load_file(
        self,
        filepath: str | list[str],
        default: Any = sentinel,
        **kwargs,
    ):
        if isinstance(filepath, str):
            filepath = [filepath]
        return load_file(
            [self._graph.local_directory(), *list(filepath)], default, **kwargs
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

    def local_directory(self, *args) -> str:
        return self._graph.local_directory(*args)

    def selections(self, comm=None) -> list[str]:
        comm = self.comm if comm is None else comm
        found = None
        if comm is False or P.is_root(comm=comm):
            directory = self.local_directory("selection")
            found = (
                sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))
                if os.path.isdir(directory)
                else []
            )
        return found if comm is False else P.broadcast(found, comm=comm)

    def selection_document(self, name: str, comm=None) -> dict:
        comm = self.comm if comm is None else comm
        document = self.load_file(["selection", f"{name}.json"], None, comm=comm)
        if document is None:
            found = self.selections(comm=comm)
            raise FileNotFoundError(
                f"{self.name!r} has no stored selection {name!r}"
                + (f"; available: {', '.join(found)}" if found else "; it has none")
            )

        recorded = (document.get("meta") or {}).get("graph")
        built = getattr(self._graph.architecture, "uuid", None)
        if recorded is not None and built is not None and recorded != built:
            raise ValueError(
                f"selection {name!r} of {self.name!r} was cut from graph "
                f"{recorded}, but this graph is {built}."
            )
        return document

    def synapse_projections(self) -> list[tuple[str, str, str, str, str]]:
        found = []
        for post, sources in (self.connections_config.get("synapses") or {}).items():
            for pre, spec in (sources or {}).items():
                syn_type = (spec or {}).get("type", "excitatory")
                mechanisms = ((spec or {}).get("mechanisms") or {}).get("default") or {}
                for section in (spec or {}).get("sections") or []:
                    found.extend(
                        (post, pre, section, mechanism, syn_type)
                        for mechanism in mechanisms
                    )
        return sorted(set(found))

    @property
    def weight_names(self) -> list[str]:
        """The weight keys this graph's synapses answer to."""
        namer = self._weight_section_name()
        names = []
        for post, pre, section, mechanism, _ in self.synapse_projections():
            name = f"{post}_{pre}-{namer(post, section)}-{mechanism}-weight"
            if name not in names:
                names.append(name)
        return names

    def _weight_section_name(self):
        """`model.section_name`, or the identity when no model resolves."""
        cached = getattr(self, "_weight_section_namer", None)
        if cached is not None:
            return cached

        def identity(_population, section):
            return section

        namer = identity
        with contextlib.suppress(
            OSError, KeyError, TypeError, ValueError, AttributeError, ImportError
        ):
            namer = self.default_model().section_name
        self._weight_section_namer = namer
        return namer

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
    def population_ranges(self):
        return self.cells_meta_data.population_ranges

    def population_count(self, population: types.PopulationName) -> int:
        return self.cells_meta_data.population_count(population)

    @property
    def populations(self):
        return self.cells_meta_data.population_names

    def selection(
        self,
        spec,
        populations: Sequence[str] | None = None,
        seed: int | None = 123,
        method: str = "first",
        bounds=None,
    ) -> dict[str, np.ndarray] | None:
        """Resolve a cell subselection of this system's graph.

        The result is deterministic and identical on every MPI rank.

        Parameters
        ----------
        spec :
            - ``None`` -> no subselection (returns ``None``; build everything),
              unless ``bounds`` is given.
            - ``int N`` -> ``N`` cells total, allocated across populations in
              proportion to their size (preserves population ratios).
            - ``float f`` in (0, 1] -> fraction ``f`` of each population, or of
              the *area* under ``method="patch"``.
            - ``dict`` -> per-population override; each value may be an ``int``
              count, a ``float`` fraction, or an explicit sequence of gids.
              Under ``method="patch"`` only explicit sequences are accepted.
        populations :
            Populations eligible for selection; defaults to all of the system's.
        seed :
            Seed for ``method="random"`` (ignored by the other methods).
        method :
            - ``"first"`` (default) -> contiguous gid block.
            - ``"random"`` -> seeded sample.
            - ``"patch"`` -> a centred planar region
        bounds :
            ``[[x0, y0], [x1, y1]]`` explicit patch box, in the coordinate units
            of the graph. ``method="patch"`` only. Takes precedence over ``spec``.

        Returns
        -------
        ``{population: np.ndarray[gid]}`` or ``None``.
        """
        return resolve_selection(self, spec, populations, seed, method, bounds)

    @property
    def neuron_coordinates(self) -> types.Float[types.Array, "n_coords ixyz=4"]:
        if self._neuron_coordinates is None:
            coordinates = np.vstack(
                [
                    self.coordinate_array(population_name)
                    for population_name in self.populations
                ]
            )
            self._neuron_coordinates = coordinates[coordinates[:, 0].argsort()]

        return self._neuron_coordinates

    @property
    def gids(self) -> types.Int[types.Array, " n_neurons"]:
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
        self, population: types.PopulationName
    ) -> types.Float[types.Array, "n_coords cxyz=4"]:
        """Every cell's `[gid, x, y, z]` for a population, read once."""
        cached = self._coordinate_arrays.get(population)
        if cached is None:
            cached = coordinate_array(
                self._graph.cells_filepath, population, comm=self.comm, all=True
            )
            self._coordinate_arrays[population] = cached
        return cached.copy()

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

    def placement(
        self, population: types.PopulationName, gids
    ) -> dict[int, tuple[Any, Any, Any]]:
        placement = read_placement(
            self._graph.cells_filepath,
            population,
            gids,
            comm=self.comm,
            io_size=self.io_size,
        )

        endpoints = sum(
            int(((locs <= 0.0) | (locs >= 1.0)).sum())
            for _, _, locs in placement.values()
        )
        if endpoints:
            logger.warning(
                "%s: %d synapse site(s) are recorded at section position 0 or 1, "
                "which cannot hold an ion mechanism; they were moved to the "
                "nearest segment centre",
                population,
                endpoints,
            )

        return placement

    def _destination_index(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
    ):
        """The sorted destination gids a projection carries edges for.

        Read once per projection so that ``edges`` can hand neuroh5 only gids the
        projection actually has, which it requires.
        """
        key = (pre, post)
        if key in self._destination_indices:
            return self._destination_indices[key]

        import numpy as npn

        f = _open_h5(self._graph.connections_filepath)
        if pre not in stored_projections(f).get(post, ()):
            self._destination_indices[key] = None
            return None
        group = f[f"Projections/{post}/{pre}/Edges"]

        starts = npn.asarray(group["Destination Block Index"][:]).astype(npn.int64)
        block_ptr = npn.asarray(group["Destination Block Pointer"][:]).astype(npn.int64)
        n_dst = int(group["Destination Pointer"].shape[0]) - 1

        counts = npn.diff(block_ptr)
        within = npn.arange(int(counts.sum())) - npn.repeat(block_ptr[:-1], counts)
        gids = (npn.repeat(starts, counts) + within)[:n_dst]
        gids += int(self.population_ranges[post][0])

        self._destination_indices[key] = npn.sort(gids)
        return self._destination_indices[key]

    def edges(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        gids,
    ):
        yield from read_edges(
            self._graph.connections_filepath,
            pre,
            post,
            gids,
            comm=self.comm,
            io_size=self.io_size,
            # only the neuroh5 reader needs it, and building it costs a read
            destinations=self._destination_index(pre, post) if _HAS_NEUROH5 else None,
            population_ranges=self.cells_meta_data.population_ranges,
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
        self, weights: dict | None = None, seed=123, gids=None
    ) -> types.Float[types.Array, "num_neurons num_neurons"]:
        """The weighted adjacency matrix, optionally restricted to ``gids``."""
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
                weight = weights.get(f"{post}_{pre}", 1.0)

                for post_gid, (pre_gids, _projection) in self.projection_array(
                    pre, post
                ):
                    # distances = projection
                    # if isinstance(projection, dict):
                    #     distances = projection["Connections"][0]

                    for pre_gid in pre_gids:
                        w[pre_gid, post_gid] = prefix * prng.random() * weight

        if gids is None:
            return w

        index = npn.asarray(gids, dtype=int)
        return w[npn.ix_(index, index)]

    def summary(self) -> dict[str, int | dict[str, int]]:
        num_neurons = 0
        num_projections = 0
        population_counts = {}

        for population in self.populations:
            count = self.cells_meta_data.population_count(population)
            population_counts[population] = count
            num_neurons += count

        for post, v in self.connections_config["synapses"].items():
            for pre in v:
                for _, (pre_gids, _) in self.projection_array(pre, post):
                    num_projections += len(pre_gids)

        return {
            "num_neurons": num_neurons,
            "num_projections": num_projections,
            "population_counts": population_counts,
        }
