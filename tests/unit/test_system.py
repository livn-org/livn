import os

import numpy as np
import pytest


SYSTEM_DIR = os.environ.get("LIVN_TEST_SYSTEM", "./systems/graphs/EI")


@pytest.fixture
def system_dir():
    if not os.path.isfile(os.path.join(SYSTEM_DIR, "graph.json")):
        pytest.skip("No test system found")
    return SYSTEM_DIR


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
        declared = [
            (post, pre)
            for post in pop_names
            for pre in pop_names
            if f"Projections/{post}/{pre}" in f_conns
        ]
        assert declared, "the test system declares no projections at all"

        for post, pre in declared:
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


class TestSystemWithPyfive:
    def test_cells_meta_data(self, system_dir):
        from livn.system import System

        system = System(system_dir)
        meta = system.cells_meta_data
        assert len(meta.population_names) > 0
        assert meta.cell_count() > 0

    def test_coordinate_array(self, system_dir):
        from livn.system import System

        system = System(system_dir)
        for pop in system.populations:
            coords = system.coordinate_array(pop)
            assert coords.ndim == 2
            assert coords.shape[1] == 4
            assert coords.shape[0] == system.population_count(pop)

    def test_neuron_coordinates(self, system_dir):
        from livn.system import System

        system = System(system_dir)
        coords = system.neuron_coordinates
        assert coords.shape[0] == system.num_neurons
        assert coords.shape[1] == 4

    def test_projection_array(self, system_dir):
        from livn.system import System

        system = System(system_dir)
        for post, v in system.connections_config["synapses"].items():
            for pre in v:
                projs = system.projection_array(pre, post)
                assert len(projs) > 0
                for post_gid, (pre_gids, proj_data) in projs:
                    assert isinstance(post_gid, (int, np.integer))
                    assert len(pre_gids) > 0

    def test_connectivity_matrix(self, system_dir):
        from livn.system import System

        system = System(system_dir)
        w = system.connectivity_matrix()
        n = system.num_neurons
        assert w.shape == (n, n)
        assert np.count_nonzero(w) > 0

    def test_summary(self, system_dir):
        from livn.system import System

        system = System(system_dir)
        s = system.summary()
        assert s["num_neurons"] > 0
        assert s["num_projections"] > 0


class TestParallelSystem:
    def test_satisfies_the_system_protocol(self):
        from livn.system import ParallelSystem, System
        from livn.types import System as SystemProtocol

        assert isinstance(ParallelSystem(3), SystemProtocol)
        if os.path.isdir(SYSTEM_DIR):
            assert isinstance(System(SYSTEM_DIR), SystemProtocol)

    def test_resolve(self):
        from livn.system import ParallelSystem, resolve

        assert resolve(4).population_counts == {"EXC": 4}
        assert resolve({"EXC": 3, "INH": 5}).population_counts == {"EXC": 3, "INH": 5}

        existing = ParallelSystem(2)
        assert resolve(existing) is existing

        with pytest.raises(TypeError):
            resolve(True)
        with pytest.raises(TypeError):
            resolve(1.5)

    def test_populations(self):
        from livn.system import ParallelSystem

        assert ParallelSystem(4).population_counts == {"EXC": 4}

        system = ParallelSystem({"EXC": 3, "INH": 5})
        assert system.populations == ["EXC", "INH"]
        assert system.num_neurons == 8
        assert system.population_ranges == {
            "EXC": (0, 3),
            "INH": (3, 5),
        }
        assert system.summary()["population_counts"] == {"EXC": 3, "INH": 5}

        with pytest.raises(KeyError):
            system.coordinate_array("PYR")

    def test_empty_population(self):
        from livn.system import ParallelSystem

        system = ParallelSystem({"EXC": 3, "INH": 0})

        assert system.num_neurons == 3
        assert system.populations == ["EXC", "INH"]
        assert np.asarray(system.coordinate_array("INH")).shape == (0, 4)

    def test_rejects_an_empty_system(self):
        from livn.system import ParallelSystem

        for spec in (0, {}, {"EXC": 0}, {"EXC": -1}):
            with pytest.raises(ValueError):
                ParallelSystem(spec)

    def test_is_unconnected(self):
        from livn.system import ParallelSystem

        system = ParallelSystem(3)

        assert system.connections_config["synapses"] == {"EXC": {}}
        assert system.weight_names == []
        assert system.projection_array("EXC", "EXC") == []
        assert list(system.projections("EXC", "EXC")) == []
        assert system.connectivity_matrix().shape == (3, 3)
        assert not system.connectivity_matrix().any()

    def test_coordinates_from_spacing(self):
        from livn.system import ParallelSystem

        origin = np.asarray(ParallelSystem(3).neuron_coordinates)
        np.testing.assert_array_equal(origin[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(origin[:, 1:], np.zeros((3, 3)))

        system = ParallelSystem({"EXC": 2, "INH": 2}, coordinates=5.0)
        np.testing.assert_array_equal(
            np.asarray(system.coordinate_array("EXC"))[:, 1], [0.0, 5.0]
        )
        np.testing.assert_array_equal(
            np.asarray(system.coordinate_array("INH"))[:, 1], [10.0, 15.0]
        )

    def test_coordinates_from_array(self):
        from livn.system import ParallelSystem

        xyz = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        coords = np.asarray(ParallelSystem(2, coordinates=xyz).neuron_coordinates)
        np.testing.assert_array_equal(coords[:, 0], [0, 1])
        np.testing.assert_array_equal(coords[:, 1:], xyz)

        rows = [[100, 0.0, 0.0, 0.0], [7, 1.0, 0.0, 0.0], [42, 2.0, 0.0, 0.0]]
        system = ParallelSystem(3, coordinates=rows)
        np.testing.assert_array_equal(np.asarray(system.gids), [100, 7, 42])
        np.testing.assert_array_equal(np.asarray(system.neuron_coordinates), rows)

    def test_coordinates_from_callable(self):
        from livn.system import ParallelSystem

        def grid(n):
            return np.stack([np.arange(n) * 2.0, np.zeros(n), np.ones(n)], axis=1)

        coords = np.asarray(ParallelSystem(3, coordinates=grid).neuron_coordinates)
        np.testing.assert_array_equal(coords[:, 1], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(coords[:, 3], [1.0, 1.0, 1.0])

    def test_coordinates_validation(self):
        from livn.system import ParallelSystem

        with pytest.raises(ValueError, match="coordinate rows"):
            ParallelSystem(3, coordinates=[[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="coordinate rows"):
            ParallelSystem(3, coordinates=lambda n: np.zeros((n + 1, 3)))
        with pytest.raises(ValueError, match="coordinate rows"):
            ParallelSystem({"EXC": 2, "INH": 2}, coordinates=np.zeros((2, 3)))

        with pytest.raises(ValueError, match="n_neurons"):
            ParallelSystem(2, coordinates=[[0.0, 0.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="unique"):
            ParallelSystem(2, coordinates=[[7, 0.0, 0.0, 0.0], [7, 1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="must contain integers"):
            ParallelSystem(2, coordinates=[[0.5, 0.0, 0.0, 0.0], [1.5, 1.0, 0.0, 0.0]])

    def test_selection(self):
        from livn.system import ParallelSystem

        system = ParallelSystem(10)

        assert system.selection(None) is None
        np.testing.assert_array_equal(system.selection(4)["EXC"], [0, 1, 2, 3])
        np.testing.assert_array_equal(system.selection(0.2)["EXC"], [0, 1])
        np.testing.assert_array_equal(system.selection({"EXC": [7, 3]})["EXC"], [3, 7])
        assert len(system.selection(4, method="random")["EXC"]) == 4

    def _grid(self, side=10, spacing=100.0):
        from livn.system import ParallelSystem

        xy = [(i * spacing, j * spacing) for i in range(side) for j in range(side)]
        order = np.random.default_rng(7).permutation(len(xy))
        return ParallelSystem(
            len(xy),
            coordinates=[[gid, *xy[int(pos)], 0.0] for gid, pos in enumerate(order)],
        )

    def test_selection_patch_area_fraction(self):
        system = self._grid()
        coords = {int(r[0]): (r[1], r[2]) for r in system.coordinate_array("EXC")}

        gids = system.selection(0.25, method="patch")["EXC"]

        assert 16 <= len(gids) <= 36
        xs = [coords[int(g)][0] for g in gids]
        ys = [coords[int(g)][1] for g in gids]

        assert min(xs) >= 200.0 and max(xs) <= 700.0
        assert min(ys) >= 200.0 and max(ys) <= 700.0

    def test_selection_patch_beats_random_on_locality(self):
        system = self._grid()
        coords = {int(r[0]): (r[1], r[2]) for r in system.coordinate_array("EXC")}

        def spread(gids):
            xs = [coords[int(g)][0] for g in gids]
            ys = [coords[int(g)][1] for g in gids]
            return max(max(xs) - min(xs), max(ys) - min(ys))

        n = 25
        assert spread(system.selection(n, method="patch")["EXC"]) < spread(
            system.selection(n, method="random")["EXC"]
        )

    def test_selection_patch_cell_budget_is_exact(self):
        system = self._grid()
        assert len(system.selection(25, method="patch")["EXC"]) == 25
        assert len(system.selection(1, method="patch")["EXC"]) == 1

    def test_selection_patch_explicit_bounds(self):
        system = self._grid()
        coords = {int(r[0]): (r[1], r[2]) for r in system.coordinate_array("EXC")}

        gids = system.selection(
            None, method="patch", bounds=[[0.0, 0.0], [250.0, 150.0]]
        )["EXC"]

        assert len(gids) == 6
        assert all(
            coords[int(g)][0] <= 250.0 and coords[int(g)][1] <= 150.0 for g in gids
        )

    def test_selection_patch_matches_geometry_scale_convention(self):
        system = self._grid()
        coords = {int(r[0]): (r[1], r[2]) for r in system.coordinate_array("EXC")}

        s = 0.5
        lo, hi = 0.0, 900.0
        centre, half = (lo + hi) / 2.0, (hi - lo) * s / 2.0
        expected = {
            gid
            for gid, (x, y) in coords.items()
            if abs(x - centre) <= half and abs(y - centre) <= half
        }

        gids = set(int(g) for g in system.selection(s**2, method="patch")["EXC"])
        assert gids == expected

    def test_selection_patch_preserves_population_ratio(self):
        from livn.system import ParallelSystem

        xy = [(i * 100.0, j * 100.0) for i in range(10) for j in range(10)]
        system = ParallelSystem(
            {"EXC": 80, "INH": 20},
            coordinates=[[gid, *xy[gid], 0.0] for gid in range(100)],
        )
        sel = system.selection(0.25, method="patch")
        assert len(sel.get("EXC", [])) > len(sel.get("INH", []))

    def test_selection_patch_rejects_ambiguous_specs(self):
        system = self._grid()

        with pytest.raises(ValueError, match="per-population count is ambiguous"):
            system.selection({"EXC": 0.5}, method="patch")

        with pytest.raises(ValueError, match="only meaningful for method='patch'"):
            system.selection(0.5, method="random", bounds=[[0, 0], [1, 1]])

    def test_selection_patch_explicit_gids_bypass_geometry(self):
        system = self._grid()
        np.testing.assert_array_equal(
            system.selection({"EXC": [7, 3]}, method="patch")["EXC"], [3, 7]
        )

    def test_selection_none_still_means_everything(self):
        system = self._grid()
        assert system.selection(None, method="patch") is None
