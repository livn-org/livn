import os

import numpy as np
import pytest

from livn.backend import backend
from livn.run import Run
from livn.system import System
from livn.utils import P
from testing import livn_test_env, livn_test_system

try:
    import mpi4py  # noqa: F401

    _has_mpi4py = True
except ImportError:
    _has_mpi4py = False

TIMEOUT = int(os.environ.get("LIVN_TEST_TIMEOUT", 300))


def _get_rank():
    if backend() == "neuron":
        env = livn_test_env()
        rank = env.rank
        env.close()

        return rank

    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(not _has_mpi4py, reason="mpi4py not available")
@pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") == "diffrax",
    reason="MPI gather/merge not supported under diffrax backend",
)
@pytest.mark.mpiexec(timeout=10)
@pytest.mark.parametrize(
    "mpiexec_n",
    [
        1,
        2,
        4,
    ],
)
def test_utils_P_parallel(mpiexec_n):
    rank = _get_rank()

    assert P.is_root() is not bool(rank)

    a = [rank]
    b = {"X": [rank]}

    p = P.gather(a)
    if P.is_root():
        assert p == [[i] for i in range(mpiexec_n)]
    else:
        assert p is None

    if P.is_root():
        assert np.array_equal(P.merge(p), np.arange(mpiexec_n))

    gb, ga = P.gather(b, a)
    if P.is_root():
        ma, mb = P.merge(ga, gb)
        assert np.array_equal(ma, np.arange(mpiexec_n))
        assert np.array_equal(mb["X"], np.arange(mpiexec_n))
    else:
        assert ga is None
        assert gb is None

    p = P.gather(a, all=True)
    assert p == [[i] for i in range(mpiexec_n)]

    p = P.gather(b, all=True)
    assert p == [{"X": [i]} for i in range(mpiexec_n)]

    gb, ga = P.gather(b, a, all=True)
    assert ga == [[i] for i in range(mpiexec_n)]
    assert gb == [{"X": [i]} for i in range(mpiexec_n)]

    assert P.broadcast(a) == [0]
    bb, ba = P.broadcast(b, a)
    assert bb == {"X": [0]}
    assert ba == [0]


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(not _has_mpi4py, reason="mpi4py not available")
@pytest.mark.mpiexec(timeout=10)
@pytest.mark.parametrize(
    "mpiexec_n",
    [
        1,
        2,
        4,
    ],
)
def test_utils_reduce_sum_parallel(mpiexec_n):
    rank = _get_rank()

    arr = np.array([rank, 1], dtype=np.int64)
    reduced_root = P.reduce_sum(arr)
    if P.is_root():
        expected = np.array([sum(range(mpiexec_n)), mpiexec_n])
        assert np.array_equal(reduced_root, expected)
    else:
        assert reduced_root is None

    reduced_all = P.reduce_sum(arr, all=True)
    expected = np.array([sum(range(mpiexec_n)), mpiexec_n])
    assert np.array_equal(reduced_all, expected)

    scalar_all = P.reduce_sum(rank, all=True)
    assert np.asarray(scalar_all).item() == sum(range(mpiexec_n))

    scalar_root = P.reduce_sum(rank)
    if P.is_root():
        assert np.asarray(scalar_root).item() == sum(range(mpiexec_n))
    else:
        assert scalar_root is None

    vec_all, one_all = P.reduce_sum(arr, 1, all=True)
    assert np.array_equal(vec_all, expected)
    assert np.asarray(one_all).item() == mpiexec_n

    obj = np.array([float(rank), 2.0], dtype=object)
    obj_all = P.reduce_sum(obj, all=True)
    assert np.allclose(
        np.asarray(obj_all, dtype=float),
        np.array([sum(range(mpiexec_n)), 2.0 * mpiexec_n]),
    )


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(not _has_mpi4py, reason="mpi4py not available")
@pytest.mark.mpiexec(timeout=60)
@pytest.mark.parametrize("mpiexec_n", [1, 2])
def test_every_rank_sees_the_whole_coordinate_set(mpiexec_n):
    system = System(livn_test_system())

    cc = np.asarray(system.neuron_coordinates)

    from livn.system import (
        _h5_read_cell_attributes_tuple,
        _h5_read_population_names,
        _h5_read_population_ranges,
        _pyfive_open,
    )

    cells_fp = system._graph.cells_filepath
    with _pyfive_open(cells_fp) as f:
        pop_names = _h5_read_population_names(f)
        pop_ranges = _h5_read_population_ranges(f)
    ref_parts = []
    for pop in pop_names:
        pop_start = pop_ranges[pop][0]
        with _pyfive_open(cells_fp) as f:
            items, attr_info = _h5_read_cell_attributes_tuple(
                f, pop_start, pop, "Generated Coordinates"
            )
        x_i = attr_info["X Coordinate"]
        y_i = attr_info["Y Coordinate"]
        z_i = attr_info["Z Coordinate"]
        for gid, vals in items:
            ref_parts.append([gid, vals[x_i][0], vals[y_i][0], vals[z_i][0]])
    ref_coords = np.array(ref_parts)
    ref_coords = ref_coords[ref_coords[:, 0].argsort()]
    assert np.array_equal(cc[cc[:, 0].argsort()], ref_coords)


@pytest.mark.skipif(not _has_mpi4py, reason="mpi4py not available")
@pytest.mark.skipif(
    "ax" in backend(), reason="MPI gather not supported under the diffrax backend"
)
@pytest.mark.mpiexec(timeout=30)
@pytest.mark.parametrize("mpiexec_n", [1, 2, 4])
def test_gather_folds_the_per_rank_runs(mpiexec_n):
    from livn.utils import P

    rank = P.rank()
    assert P.size() == mpiexec_n

    run = (
        Run(duration=10.0)
        .add_spikes(np.array([rank]), np.array([float(rank)]))
        .add_voltage(np.array([rank]), np.full((1, 20), float(rank)), dt=0.5)
    )

    gathered = run.gather()

    if not P.is_root():
        assert gathered is None
        return

    np.testing.assert_array_equal(gathered.spike_ids, np.arange(mpiexec_n))
    np.testing.assert_allclose(gathered.spike_times, np.arange(mpiexec_n, dtype=float))
    np.testing.assert_array_equal(gathered.voltage_ids, np.arange(mpiexec_n))
    assert gathered.voltage.shape == (mpiexec_n, 20)
    np.testing.assert_allclose(gathered.voltage[:, 0], np.arange(mpiexec_n))
    assert gathered.duration == 10.0


@pytest.mark.skipif(
    backend() != "neuron", reason="only the neuron backend distributes cells over ranks"
)
def _a_param(params):
    assert params, "the backend's cells expose no parameters"
    return sorted(params)[0]


@pytest.mark.needs("mpi")
@pytest.mark.mpiexec(n=2, timeout=TIMEOUT)
def test_cell_params_are_global_across_ranks():
    from mpi4py import MPI

    from livn.env import Env

    comm = MPI.COMM_WORLD
    env = Env(4, comm=comm).init()

    np.testing.assert_array_equal(env.cells.gids, [0, 1, 2, 3])
    assert len(env.cells.local_gids) == 4 // comm.size

    name = _a_param(env.cells.get_params())
    env.cells.set_params({name: [1.0, 2.0, 3.0, 4.0]})

    np.testing.assert_allclose(env.cells.get_params()[name], [1.0, 2.0, 3.0, 4.0])

    env.close()
