"""record("extracellular") gives the channels the recorded currents project to."""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

from testing import livn_test_env

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVN_BACKEND"), reason="no simulation backend selected"
)

NOISE = {"g_e0": 0.5, "g_i0": 0.3, "std_e": 0.15, "std_i": 0.1}


@pytest.fixture(scope="module")
def recorded():
    env = livn_test_env()
    env.init()
    if env.io is None or "extracellular" not in env.recordable():
        with contextlib.suppress(Exception):
            env.close()
        pytest.skip("this env has no io or no extracellular recording")
    with contextlib.suppress(Exception):
        env.set_noise(NOISE)
    env = env.record_spikes()  # an env may record by copy
    env = env.record("extracellular", dt=0.1)
    projected_only = env.run(20, dt=0.025)
    env = env.record_membrane_current(dt=0.1)
    both = env.run(20, dt=0.025)
    yield env, projected_only, both
    with contextlib.suppress(Exception):
        env.close()


def test_the_channel_is_a_row_per_recording_channel(recorded):
    env, projected_only, both = recorded
    channel = projected_only.channel("extracellular")
    assert channel is not None
    assert channel.projection is not None
    # sampled as the backend samples currents (diffrax: its integration step)
    assert channel.dt == pytest.approx(both.channel("current").dt)
    ids = np.asarray(env.io.channel_ids)
    np.testing.assert_array_equal(np.asarray(channel.ids), ids)
    assert np.asarray(channel.values).shape == (
        len(ids),
        np.asarray(both.current).shape[1],
    )


def test_currents_are_not_kept_unless_asked_for(recorded):
    _, projected_only, both = recorded
    assert projected_only.current is None
    assert both.current is not None


def test_the_projection_equals_the_projected_currents(recorded):
    env, projected_only, both = recorded
    conditions = both.conditions()
    from_currents = np.asarray(
        env.potential_recording(conditions=conditions.replace(extracellular_uV=None))
    )
    projected = np.asarray(
        env.potential_recording(conditions=projected_only.conditions())
    )

    assert np.abs(from_currents).max() > 0, "a silent run tests nothing"
    np.testing.assert_allclose(
        projected, from_currents, rtol=1e-5, atol=1e-4 * np.abs(from_currents).max()
    )


def test_a_changed_medium_falls_back_to_the_currents(recorded):
    env, projected_only, both = recorded
    other = env.io.set_params({"medium-electrical-sigma_s_per_m": 0.6})

    conditions = env.fill_conditions(both.conditions())
    expected = np.asarray(other.observe(conditions.replace(extracellular_uV=None)))
    np.testing.assert_allclose(np.asarray(other.observe(conditions)), expected)

    with pytest.raises(ValueError, match="different gain"):
        other.observe(env.fill_conditions(projected_only.conditions()))
