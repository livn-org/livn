import os

import numpy as np
import pytest

SYSTEM_DIR = os.environ.get("LIVN_TEST_SYSTEM", "./systems/graphs/EI1")


@pytest.fixture
def cells_filepath():
    cells = os.path.join(SYSTEM_DIR, "cells.h5")
    if not os.path.isfile(cells):
        graph = os.path.join(SYSTEM_DIR, "graph.h5")
        if os.path.isfile(graph):
            return graph
        pytest.skip("No test system H5 files found")
    return cells


@pytest.fixture
def connections_filepath():
    conns = os.path.join(SYSTEM_DIR, "connections.h5")
    if not os.path.isfile(conns):
        graph = os.path.join(SYSTEM_DIR, "graph.h5")
        if os.path.isfile(graph):
            return graph
        pytest.skip("No test system H5 files found")
    return conns


def _has_neuroh5():
    try:
        import neuroh5.io  # noqa: F401
        from mpi4py import MPI  # noqa: F401

        return True
    except ImportError:
        return False


neuroh5_required = pytest.mark.skipif(
    not _has_neuroh5(), reason="neuroh5/mpi4py not available"
)


class TestPyfiveReaders:
    def test_read_population_names(self, cells_filepath):
        from livn.system import _h5_read_population_names, _pyfive_open

        f = _pyfive_open(cells_filepath)
        names = _h5_read_population_names(f)
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_read_population_ranges(self, cells_filepath):
        from livn.system import (
            _h5_read_population_names,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        f = _pyfive_open(cells_filepath)
        ranges = _h5_read_population_ranges(f)
        names = _h5_read_population_names(f)
        assert set(ranges.keys()) == set(names)
        for name, (start, count) in ranges.items():
            assert isinstance(start, int)
            assert isinstance(count, int)
            assert count > 0

    def test_read_cell_attribute_info(self, cells_filepath):
        from livn.system import (
            _h5_read_cell_attribute_info,
            _h5_read_population_names,
            _pyfive_open,
        )

        f = _pyfive_open(cells_filepath)
        names = _h5_read_population_names(f)
        info = _h5_read_cell_attribute_info(f, names)
        assert set(info.keys()) == set(names)
        for pop, namespaces in info.items():
            assert "Generated Coordinates" in namespaces
            assert "X Coordinate" in namespaces["Generated Coordinates"]

    def test_read_cell_attributes_tuple(self, cells_filepath):
        from livn.system import (
            _h5_read_cell_attributes_tuple,
            _h5_read_population_names,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        f = _pyfive_open(cells_filepath)
        names = _h5_read_population_names(f)
        ranges = _h5_read_population_ranges(f)

        for pop in names:
            pop_start = ranges[pop][0]
            items, attr_info = _h5_read_cell_attributes_tuple(
                f, pop_start, pop, "Generated Coordinates"
            )
            pop_start, pop_count = ranges[pop]
            assert len(items) == pop_count
            assert "X Coordinate" in attr_info

            for gid, vals in items:
                assert gid >= pop_start
                assert gid < pop_start + pop_count

    def test_read_cell_attributes_dict(self, cells_filepath):
        from livn.system import (
            _h5_read_cell_attributes,
            _h5_read_population_names,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        f = _pyfive_open(cells_filepath)
        names = _h5_read_population_names(f)
        ranges = _h5_read_population_ranges(f)
        mask = {"syn_ids", "syn_types", "swc_types"}
        pop_start = ranges[names[0]][0]
        attrs = _h5_read_cell_attributes(
            f, pop_start, names[0], "Synapse Attributes", mask=mask
        )
        assert len(attrs) > 0
        for gid, cell_attrs in attrs.items():
            assert set(cell_attrs.keys()) == mask

    def test_read_graph(self, cells_filepath, connections_filepath):
        from livn.system import (
            _h5_read_graph,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        f_cells = _pyfive_open(cells_filepath)
        pop_ranges = _h5_read_population_ranges(f_cells)
        pop_names = list(pop_ranges.keys())

        f_conns = _pyfive_open(connections_filepath)
        for post in pop_names:
            for pre in pop_names:
                pre_start = pop_ranges[pre][0]
                post_start = pop_ranges[post][0]
                results = _h5_read_graph(
                    f_conns,
                    pre_start,
                    post_start,
                    pre,
                    post,
                    namespaces=["Connections", "Synapses"],
                )
                pre_end = pre_start + pop_ranges[pre][1]

                for post_gid, (pre_gids, ns_data) in results:
                    assert post_gid >= post_start
                    assert all(g >= pre_start and g < pre_end for g in pre_gids)
                    assert "Connections" in ns_data
                    assert len(ns_data["Connections"]) > 0


@neuroh5_required
class TestPyfiveVsNeuroh5:
    def test_population_names(self, cells_filepath):
        from mpi4py import MPI
        from neuroh5.io import read_population_names as neuroh5_read_pop_names

        from livn.system import _h5_read_population_names, _pyfive_open

        comm = MPI.COMM_WORLD
        neuroh5_names = neuroh5_read_pop_names(cells_filepath, comm)
        f = _pyfive_open(cells_filepath)
        pyfive_names = _h5_read_population_names(f)

        assert neuroh5_names == pyfive_names

    def test_population_ranges(self, cells_filepath):
        from mpi4py import MPI
        from neuroh5.io import read_population_ranges as neuroh5_read_pop_ranges

        from livn.system import _h5_read_population_ranges, _pyfive_open

        comm = MPI.COMM_WORLD
        neuroh5_ranges, _ = neuroh5_read_pop_ranges(cells_filepath, comm)
        f = _pyfive_open(cells_filepath)
        pyfive_ranges = _h5_read_population_ranges(f)

        assert neuroh5_ranges == pyfive_ranges

    def test_cell_attribute_info(self, cells_filepath):
        from mpi4py import MPI
        from neuroh5.io import (
            read_cell_attribute_info as neuroh5_read_attr_info,
            read_population_names,
        )

        from livn.system import _h5_read_cell_attribute_info, _pyfive_open

        comm = MPI.COMM_WORLD
        pop_names = read_population_names(cells_filepath, comm)
        neuroh5_info = neuroh5_read_attr_info(cells_filepath, pop_names, comm=comm)
        f = _pyfive_open(cells_filepath)
        pyfive_info = _h5_read_cell_attribute_info(f, pop_names)

        assert set(neuroh5_info.keys()) == set(pyfive_info.keys())
        for pop in neuroh5_info:
            assert set(neuroh5_info[pop].keys()) == set(pyfive_info[pop].keys())
            for ns in neuroh5_info[pop]:
                assert sorted(neuroh5_info[pop][ns]) == sorted(pyfive_info[pop][ns])

    def test_coordinates(self, cells_filepath):
        from mpi4py import MPI
        from neuroh5.io import scatter_read_cell_attributes

        from livn.system import (
            _h5_read_cell_attributes_tuple,
            _h5_read_population_names,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        comm = MPI.COMM_WORLD
        f = _pyfive_open(cells_filepath)
        pop_names = _h5_read_population_names(f)
        pop_ranges = _h5_read_population_ranges(f)

        for pop in pop_names:
            # neuroh5
            cell_attr_dict = scatter_read_cell_attributes(
                cells_filepath,
                pop,
                namespaces=["Generated Coordinates"],
                return_type="tuple",
                comm=comm,
            )
            neuroh5_iter, neuroh5_attr_info = cell_attr_dict["Generated Coordinates"]
            neuroh5_items = list(neuroh5_iter)

            # pyfive
            pop_start = pop_ranges[pop][0]
            pyfive_items, pyfive_attr_info = _h5_read_cell_attributes_tuple(
                f, pop_start, pop, "Generated Coordinates"
            )

            # Compare attr_info
            assert neuroh5_attr_info == pyfive_attr_info

            # Compare items
            assert len(neuroh5_items) == len(pyfive_items)
            for (n_gid, n_vals), (p_gid, p_vals) in zip(neuroh5_items, pyfive_items):
                assert n_gid == p_gid
                assert len(n_vals) == len(p_vals)
                for n_v, p_v in zip(n_vals, p_vals):
                    np.testing.assert_array_almost_equal(n_v, p_v)

    def test_synapse_attributes(self, cells_filepath):
        from mpi4py import MPI
        from neuroh5.io import scatter_read_cell_attributes

        from livn.system import (
            _h5_read_cell_attributes,
            _h5_read_population_names,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        comm = MPI.COMM_WORLD
        f = _pyfive_open(cells_filepath)
        pop_names = _h5_read_population_names(f)
        pop_ranges = _h5_read_population_ranges(f)
        mask = {
            "syn_ids",
            "syn_locs",
            "syn_secs",
            "syn_layers",
            "syn_types",
            "swc_types",
        }

        for pop in pop_names:
            # neuroh5
            cell_attr_dict = scatter_read_cell_attributes(
                cells_filepath,
                pop,
                namespaces=["Synapse Attributes"],
                mask=mask,
                comm=comm,
                io_size=1,
                return_type="dict",
            )
            neuroh5_items = {
                gid: attrs for gid, attrs in cell_attr_dict["Synapse Attributes"]
            }

            # pyfive
            pop_start = pop_ranges[pop][0]
            pyfive_items = _h5_read_cell_attributes(
                f, pop_start, pop, "Synapse Attributes", mask=mask
            )

            assert set(neuroh5_items.keys()) == set(pyfive_items.keys())
            for gid in neuroh5_items:
                n_attrs = neuroh5_items[gid]
                p_attrs = pyfive_items[gid]
                assert set(n_attrs.keys()) == set(p_attrs.keys())
                for attr_name in n_attrs:
                    np.testing.assert_array_equal(
                        n_attrs[attr_name], p_attrs[attr_name]
                    )

    def test_projections(self, cells_filepath, connections_filepath):
        from mpi4py import MPI
        from neuroh5.io import scatter_read_graph

        from livn.system import (
            _h5_read_graph,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        comm = MPI.COMM_WORLD
        f_cells = _pyfive_open(cells_filepath)
        pop_ranges = _h5_read_population_ranges(f_cells)
        pop_names = list(pop_ranges.keys())

        f_conns = _pyfive_open(connections_filepath)
        for post in pop_names:
            for pre in pop_names:
                # neuroh5
                (graph, _) = scatter_read_graph(
                    connections_filepath,
                    comm=comm,
                    io_size=1,
                    projections=[(pre, post)],
                    namespaces=["Synapses", "Connections"],
                )
                neuroh5_items = list(graph[post][pre])

                # pyfive
                pre_start = pop_ranges[pre][0]
                post_start = pop_ranges[post][0]
                pyfive_items = _h5_read_graph(
                    f_conns,
                    pre_start,
                    post_start,
                    pre,
                    post,
                    namespaces=["Connections", "Synapses"],
                )

                assert len(neuroh5_items) == len(pyfive_items), (
                    f"Projection {pre}->{post}: neuroh5 has {len(neuroh5_items)} "
                    f"entries, pyfive has {len(pyfive_items)}"
                )

                # Sort both by post_gid for stable comparison
                neuroh5_items.sort(key=lambda x: x[0])
                pyfive_items.sort(key=lambda x: x[0])

                for (n_gid, (n_pre, n_proj)), (p_gid, (p_pre, p_proj)) in zip(
                    neuroh5_items, pyfive_items
                ):
                    assert n_gid == p_gid, f"GID mismatch: {n_gid} vs {p_gid}"
                    np.testing.assert_array_equal(
                        np.sort(n_pre),
                        np.sort(p_pre),
                        err_msg=f"Pre-GIDs mismatch for post_gid={n_gid}",
                    )

                    # Compare namespace data
                    assert set(n_proj.keys()) == set(p_proj.keys())
                    for ns_name in n_proj:
                        for n_arr, p_arr in zip(n_proj[ns_name], p_proj[ns_name]):
                            np.testing.assert_array_almost_equal(
                                n_arr,
                                p_arr,
                                err_msg=f"Namespace {ns_name} mismatch for "
                                f"post_gid={n_gid}",
                            )


class TestSystemWithPyfive:
    def test_cells_meta_data(self):
        from livn.system import System

        system = System(SYSTEM_DIR)
        meta = system.cells_meta_data
        assert len(meta.population_names) > 0
        assert meta.cell_count() > 0

    def test_coordinate_array(self):
        from livn.system import System

        system = System(SYSTEM_DIR)
        for pop in system.populations:
            coords = system.coordinate_array(pop)
            assert coords.ndim == 2
            assert coords.shape[1] == 4  # gid, x, y, z
            assert coords.shape[0] == system.cells_meta_data.population_count(pop)

    def test_neuron_coordinates(self):
        from livn.system import System

        system = System(SYSTEM_DIR)
        coords = system.neuron_coordinates
        assert coords.shape[0] == system.num_neurons
        assert coords.shape[1] == 4

    def test_projection_array(self):
        from livn.system import System

        system = System(SYSTEM_DIR)
        for post, v in system.connections_config["synapses"].items():
            for pre in v:
                projs = system.projection_array(pre, post)
                assert len(projs) > 0
                for post_gid, (pre_gids, proj_data) in projs:
                    assert isinstance(post_gid, (int, np.integer))
                    assert len(pre_gids) > 0

    def test_connectivity_matrix(self):
        from livn.system import System

        system = System(SYSTEM_DIR)
        w = system.connectivity_matrix()
        n = system.num_neurons
        assert w.shape == (n, n)
        # Should have non-zero entries
        assert np.count_nonzero(w) > 0

    def test_summary(self):
        from livn.system import System

        system = System(SYSTEM_DIR)
        s = system.summary()
        assert s["num_neurons"] > 0
        assert s["num_projections"] > 0


@neuroh5_required
class TestSystemPyfiveVsNeuroh5:
    def test_coordinate_array_equivalence(self):
        from livn.system import (
            System,
            _h5_read_cell_attributes_tuple,
            _h5_read_population_names,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        system = System(SYSTEM_DIR)
        f = _pyfive_open(system._graph.cells_filepath)
        pop_names = _h5_read_population_names(f)
        pop_ranges = _h5_read_population_ranges(f)

        for pop in pop_names:
            # neuroh5 path (via System)
            coords_n = system.coordinate_array(pop)

            # pyfive path (manual)
            pop_start = pop_ranges[pop][0]
            items, attr_info = _h5_read_cell_attributes_tuple(
                f, pop_start, pop, "Generated Coordinates"
            )
            x_i = attr_info["X Coordinate"]
            y_i = attr_info["Y Coordinate"]
            z_i = attr_info["Z Coordinate"]
            coordinates = []
            for gid, vals in items:
                coordinates.append([gid, vals[x_i][0], vals[y_i][0], vals[z_i][0]])
            coords_p = np.array(coordinates)
            coords_p = coords_p[coords_p[:, 0].argsort()]

            np.testing.assert_array_almost_equal(coords_n, coords_p)

    def test_cells_meta_data_equivalence(self):
        from livn.system import (
            System,
            _h5_read_cell_attribute_info,
            _h5_read_population_names,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        system = System(SYSTEM_DIR)
        f = _pyfive_open(system._graph.cells_filepath)

        pop_names = _h5_read_population_names(f)
        pop_ranges = _h5_read_population_ranges(f)
        attr_info = _h5_read_cell_attribute_info(f, pop_names)

        meta_n = system.cells_meta_data
        assert meta_n.population_names == pop_names
        assert meta_n.population_ranges == pop_ranges
        assert meta_n.cell_attribute_info == attr_info

    def test_projection_array_equivalence(self):
        from livn.system import (
            System,
            _h5_read_graph,
            _h5_read_population_ranges,
            _pyfive_open,
        )

        system = System(SYSTEM_DIR)
        f_cells = _pyfive_open(system._graph.cells_filepath)
        pop_ranges = _h5_read_population_ranges(f_cells)

        f_conns = _pyfive_open(system._graph.connections_filepath)
        for post, v in system.connections_config["synapses"].items():
            for pre in v:
                projs_n = system.projection_array(pre, post)
                pre_start = pop_ranges[pre][0]
                post_start = pop_ranges[post][0]
                projs_p = _h5_read_graph(
                    f_conns,
                    pre_start,
                    post_start,
                    pre,
                    post,
                    namespaces=["Synapses", "Connections"],
                )

                assert len(projs_n) == len(projs_p)

                projs_n.sort(key=lambda x: x[0])
                projs_p.sort(key=lambda x: x[0])

                for (gid_n, (pre_n, proj_n)), (gid_p, (pre_p, proj_p)) in zip(
                    projs_n, projs_p
                ):
                    assert gid_n == gid_p
                    np.testing.assert_array_equal(np.sort(pre_n), np.sort(pre_p))

                    for ns in proj_n:
                        for arr_n, arr_p in zip(proj_n[ns], proj_p[ns]):
                            np.testing.assert_array_almost_equal(arr_n, arr_p)
