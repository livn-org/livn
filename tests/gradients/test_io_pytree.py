import numpy as np
import pytest

pytest.importorskip("jax")

import jax

from livn.io import MEA, LightArray
from livn.io.mea import _USES_JAX

pytestmark = pytest.mark.skipif(not _USES_JAX, reason="the pytree needs a jax backend")


def test_the_mea_pytree_round_trip_keeps_the_detector():
    mea = MEA(
        [[0, 0.0, 0.0, 0.0], [1, 100.0, 0.0, 0.0]],
        detection_threshold=0.2,
        noise_uv=3.5,
        threshold_sigma=4.0,
        dead_time_ms=1.5,
        spike_profile={"distance_um": [0, 50, 100], "trough_uv": [-80, -20, -5]},
    )
    mea.attach_spike_profile({"distance_um": [0, 10], "trough_uv": [-50, -5]})
    assert mea.physical_detection

    out = jax.tree_util.tree_map(lambda x: x, mea)

    assert out.physical_detection
    for name in (
        "detection_threshold",
        "noise_uv",
        "threshold_sigma",
        "dead_time_ms",
        "spike_profile",
        "_attached_profile",
        "input_radius",
        "output_radius",
    ):
        assert getattr(out, name) == getattr(mea, name), name
    assert np.array_equal(out.electrode_coordinates, mea.electrode_coordinates)
    assert out.serialize() == mea.serialize()


def test_a_light_array_induction_builds_under_jax():
    fibers = np.array([[0, 0.0, 0.0, 0.0], [1, 200.0, 0.0, 50.0]])
    cells = np.array([[0, 10.0, 10.0, 100.0], [1, -50.0, 30.0, 300.0]])
    T = LightArray(fibers).cell_induction(cells)
    assert T.shape == (2, 2)
    assert np.all(np.asarray(T) > 0)
