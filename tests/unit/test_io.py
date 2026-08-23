import numpy as np

from livn import io
from livn.utils import lnp


try:
    import mpi4py  # noqa: F401

    _has_mpi4py = True
except ImportError:
    _has_mpi4py = False


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
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ]
    )

    cell_induction = np.array(
        [
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


def test_cell_stimulus_keeps_the_command_precision():
    cell_induction = np.array([[0, 0, 0.5], [0, 1, 0.25]])

    for dtype in (np.float32, np.float64):
        command = np.ones((1, 4, 1), dtype=dtype)
        expected = np.asarray(lnp().asarray(command)).dtype

        result = io.calculate_cell_stimulus(command, cell_induction, n_gids=2)

        assert np.asarray(result).dtype == expected, (
            f"a {np.dtype(dtype).name} command came back as "
            f"{np.asarray(result).dtype}, not the backend's {expected}"
        )
        assert np.allclose(np.asarray(result)[0, 0], [0.5, 0.25])


def test_channel_recording():
    mapping = np.array(
        [
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
    mea = io.MEA(np.empty([3, 4]), 150, 300)
    clone = mea.clone()
    assert clone.input_radius == mea.input_radius


def test_potential_recording():
    e_coords = np.array([[0, 0.0, 0.0, 0.0]])
    n_coords = np.array([[0, 0.0, 0.0, 0.0], [1, 1000.0, 0.0, 0.0]])

    mea = io.MEA(e_coords, input_radius=250, output_radius=2000)

    sigma = 0.0003
    min_du = 5.0
    r0 = (0.0 + min_du) / 1000.0
    r1 = (1000.0 + min_du) / 1000.0
    factor = 1.0 / (4.0 * np.pi * sigma)
    expected = factor * (1.0 / r0 + 1.0 / r1)

    d = mea.distances(n_coords)

    i2d = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
        ]
    )
    v2 = mea.potential_recording(d, i2d)
    assert v2.shape == (1, 2)
    assert np.allclose(v2[0, :], np.array([expected, 2.0 * expected]), rtol=1e-6)

    mea_masked = io.MEA(e_coords, input_radius=250, output_radius=10)
    expected_masked = factor * (1.0 / r0)
    v_masked = mea_masked.potential_recording(d, i2d[:, :1])
    assert np.allclose(v_masked[0], np.array([expected_masked]), rtol=1e-6)
