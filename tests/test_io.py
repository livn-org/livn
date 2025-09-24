import os

import numpy as np
import pytest
from mpi4py import MPI

from livn import io
from livn.system import CachedSystem, System
from livn.utils import P


def test_calculate_distances():
    electrodes = [[0, 0.0, 0.0, 0.0], [1, 1.0, 0.0, 0.0], [2, 0.0, 1.0, 0.0]]
    coords = [[0, 0.0, 0.0, 0.0], [1, 1.0, 1.0, 1.0]]

    expected_output = np.array(
        [
            [0, 0, 0.0],
            [0, 1, np.sqrt(3)],
            [1, 0, 1.0],
            [1, 1, np.sqrt(2)],
            [2, 0, 1.0],
            [2, 1, np.sqrt(2)],
        ]
    )

    result = io.calculate_distances(electrodes, coords)

    assert np.allclose(result, expected_output)


def test_relative_distance():
    distances = np.array(
        [
            [0, 0, 100.0],
            [0, 1, 600.0],
            [1, 0, 200.0],
            [1, 1, 400.0],
            [2, 0, 500.0],
            [2, 1, 700.0],
        ]
    )

    boundary = 500

    expected_output = np.array([[0, 0, 0.2], [1, 0, 0.4], [1, 1, 0.8], [2, 0, 1.0]])
    result = io.relative_distance(distances, boundary, filter_out_of_bounds=True)
    assert np.allclose(result, expected_output)

    result = io.relative_distance(distances, boundary, filter_out_of_bounds=False)
    assert result[-1][-2] == 1
    assert result[-1][-1] > 1


def test_calculate_cell_stimulus():
    electrode_stimulus = np.array(
        [
            [
                # c0   c1   c2
                [0.1, 0.2, 0.3],  # t0
                [0.4, 0.5, 0.6],  # t1
            ],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ]
    )

    cell_induction = np.array(
        [
            # c  gid  amplitude
            [0, 0, 0.5],
            [1, 1, 0.6],
            [2, 0, 0.7],
        ]
    )

    expected_output = np.array(
        [
            [[0.1 * 0.5 + 0.3 * 0.7, 0.2 * 0.6], [0.4 * 0.5 + 0.6 * 0.7, 0.5 * 0.6]],
            [[0.7 * 0.5 + 0.9 * 0.7, 0.8 * 0.6], [1.0 * 0.5 + 1.2 * 0.7, 1.1 * 0.6]],
        ]
    )

    result = io.calculate_cell_stimulus(electrode_stimulus, cell_induction)

    assert np.allclose(result, expected_output)

    assert result[1, 1, 1] == 1.1 * 0.6


def test_channel_recording():
    mapping = np.array(
        [
            # c  gid
            [0, 1, -0.1],
            [1, 0, -0.2],
            [1, 2, -0.3],
            [2, 3, -0.4],
        ]
    )
    ii = np.array([0, 0, 1, 2])
    tt = np.array([0.1, 0.2, 0.3, 0.4])

    cii = io.channel_recording(mapping, ii)
    assert not isinstance(cii, tuple)
    cii, ctt = io.channel_recording(mapping, ii, tt)

    assert cii[0].tolist() == [1]
    assert np.allclose(ctt[0], [0.3])

    assert cii[1].tolist() == [0, 0, 2]
    assert np.allclose(ctt[1], [0.1, 0.2, 0.4])

    assert cii[2].tolist() == []
    assert ctt[2].tolist() == []


def test_mea():
    # test serialization
    mea = io.MEA(np.empty([3, 4]), 150, 300)
    clone = mea.clone()
    assert clone.input_radius == mea.input_radius


def test_potential_recording():
    e_coords = np.array([[0, 0.0, 0.0, 0.0]])
    n_coords = np.array([[0, 0.0, 0.0, 0.0], [1, 1000.0, 0.0, 0.0]])

    mea = io.MEA(e_coords, input_radius=250, output_radius=2000)

    sigma = 0.0003  # S/mm
    min_du = 5.0  # um
    r0 = (0.0 + min_du) / 1000.0
    r1 = (1000.0 + min_du) / 1000.0
    factor = 1.0 / (4.0 * np.pi * sigma)
    expected = factor * (1.0 / r0 + 1.0 / r1)

    d = mea.distances(n_coords)

    # 2D currents (timestep, n_neurons) -> returns (timestep, n_channels)
    i2d = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    v2 = mea.potential_recording(d, i2d)
    assert v2.shape == (2, 1)
    assert np.allclose(v2[:, 0], np.array([expected, 2.0 * expected]), rtol=1e-6)

    # Masking: only the neuron at origin contributes (tight radius)
    mea_masked = io.MEA(e_coords, input_radius=250, output_radius=10)
    expected_masked = factor * (1.0 / r0)
    v_masked = mea_masked.potential_recording(d, i2d[:1])
    assert np.allclose(v_masked[0], np.array([expected_masked]), rtol=1e-6)


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=60)
@pytest.mark.parametrize("mpiexec_n", [1, 2])
def test_mea_parallel(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

    system = System(os.environ["LIVN_TEST_SYSTEM"])
    cached_system = CachedSystem(os.environ["LIVN_TEST_SYSTEM"])

    mea = io.MEA.from_json(os.path.join(system.uri, "mea.json"))
    cached_mea = io.MEA.from_json(os.path.join(cached_system.uri, "mea.json"))

    q = P.gather(system.neuron_coordinates)

    if P.is_root():
        cc = np.vstack(q)
        assert np.array_equal(cc[cc[:, 0].argsort()], cached_system.neuron_coordinates)
