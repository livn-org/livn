from __future__ import annotations

import contextlib
import gc

import numpy as np
import pytest

from livn.stimulus import Stimulus
from livn.types import Snapshot
from testing import livn_test_env

pytestmark = pytest.mark.needs("snapshot", "noise", "extracellular_stimulus")

NOISE = {
    "g_e0": 0.5,
    "g_i0": 0.3,
    "std_e": 0.15,
    "std_i": 0.1,
    "tau_e": 10.0,
    "tau_i": 10.0,
}
PLASTICITY = {"A_ltp": 0.01, "A_ltd": 0.005}
DT = 0.1
FIELD = 300.0  # mV at the most distant section
LEAD = 80.0
AFTER = 60.0

_OPEN_ENVS = []


@pytest.fixture(autouse=True)
def _close_envs():
    """One NEURON env per test, closed after it; leaked ones wedge later psolves."""
    yield
    while _OPEN_ENVS:
        with contextlib.suppress(Exception):
            _OPEN_ENVS.pop().close()


def _env(plastic: bool = False, model=None):
    gc.collect()
    env = livn_test_env(model=model).init()
    _OPEN_ENVS.append(env)
    env.apply_model_defaults()
    env.set_noise(NOISE)
    if plastic:
        env.enable_plasticity(PLASTICITY)
    return env.record_spikes().record_voltage(dt=DT)


def _current(env, duration: float, amplitude: float) -> Stimulus:
    gids = np.asarray(env.active_gids())
    values = np.zeros((round(duration / DT), len(gids)))
    values[values.shape[0] // 3 :] = amplitude
    return Stimulus.from_current(values, dt=DT, gids=gids)


def _field(env, duration: float, amplitude: float) -> Stimulus:
    """An extracellular field, graded across the sections so the junctions see it."""
    gids = np.asarray(env.stimulus_coordinates())[:, 0].astype(int)
    values = np.zeros((round(duration / DT), len(gids)))
    values[values.shape[0] // 4 :] = amplitude * np.linspace(0.0, 1.0, len(gids))
    return Stimulus(values, dt=DT, gids=gids)


def _assert_identical(a, b):
    np.testing.assert_array_equal(a.spike_ids, b.spike_ids)
    np.testing.assert_array_equal(a.spike_times, b.spike_times)
    np.testing.assert_array_equal(a.voltage_ids, b.voltage_ids)
    np.testing.assert_array_equal(a.voltage, b.voltage)


def _differs(a, b) -> bool:
    if a.spike_times.shape != b.spike_times.shape:
        return True
    return not (
        np.array_equal(a.spike_times, b.spike_times)
        and np.array_equal(a.voltage, b.voltage)
    )


def _in_flight(env, run) -> int:
    """Spikes of `run` whose events arrive after it ended."""
    table = env.connections()
    latest = {}
    for pre, delay in zip(table["pre_gid"], table["delay"], strict=True):
        latest[int(pre)] = max(latest.get(int(pre), 0.0), float(delay))
    arrival = [
        float(t) + latest.get(int(gid), 0.0)
        for gid, t in zip(run.spike_ids, run.spike_times, strict=True)
    ]
    return int(np.sum(np.asarray(arrival) > run.duration))


@pytest.mark.parametrize(
    "plastic",
    [False, pytest.param(True, marks=pytest.mark.needs("plasticity"))],
    ids=["static", "plastic"],
)
def test_a_restored_run_continues_the_uninterrupted_one_to_the_bit(plastic):
    env = _env(plastic)
    lead = env.run(LEAD, _current(env, LEAD, 0.05))
    assert _in_flight(env, lead) > 0, "no spike was in flight at the snapshot"
    snapshot = env.save_state()
    assert isinstance(snapshot, Snapshot)
    assert snapshot.t == LEAD

    stimulus = _field(env, AFTER, FIELD)
    straight = env.run(AFTER, stimulus)
    assert straight.spike_times.size > 0, "a silent window would prove nothing"
    weights = env.get_weights() if plastic else None

    # a different branch, a clear() that empties the event queue, a third branch
    env.run(25.0, _field(env, 25.0, 2 * FIELD))
    env.clear()
    env.run(15.0)

    env.restore_state(snapshot)
    assert env.t == LEAD
    forked = env.run(AFTER, stimulus)
    _assert_identical(straight, forked)
    if plastic:
        restored = env.get_weights()
        assert any(
            not np.array_equal(weights[k], np.ones_like(weights[k])) for k in weights
        ), "the weights did not learn, so the plastic case tests nothing"
        for key, value in weights.items():
            np.testing.assert_array_equal(value, restored[key])


def test_restoring_twice_gives_the_same_run():
    env = _env()
    env.run(LEAD)
    snapshot = env.save_state()
    quiet = env.run(AFTER)
    stimulus = _field(env, AFTER, FIELD)

    first = env.restore_state(snapshot).run(AFTER, stimulus)
    second = env.restore_state(snapshot).run(AFTER, stimulus)
    assert first.spike_times.size > 0
    _assert_identical(first, second)
    assert _differs(quiet, first)
    # the stimulus of a branch does not reach the next one
    _assert_identical(quiet, env.restore_state(snapshot).run(AFTER))


def test_another_noise_stream_forks_a_different_reproducible_run():
    env = _env()
    env.run(LEAD)
    snapshot = env.save_state()
    straight = env.run(AFTER)

    own = env.restore_state(snapshot, noise_stream=snapshot.noise_stream).run(AFTER)
    _assert_identical(straight, own)

    other = snapshot.noise_stream + 1
    forked = env.restore_state(snapshot, noise_stream=other).run(AFTER)
    assert env.noise_stream == other
    assert _differs(straight, forked)
    again = env.restore_state(snapshot, noise_stream=other).run(AFTER)
    _assert_identical(forked, again)


def test_a_stimulus_scheduled_past_the_snapshot_is_carried():
    env = _env()
    env.run(LEAD, _current(env, 2 * LEAD, 0.1))  # half of it still to come
    snapshot = env.save_state()
    straight = env.run(AFTER)
    env.restore_state(snapshot)
    _assert_identical(straight, env.run(AFTER))


def test_an_extracellular_field_continues_across_the_restore():
    env = _env()
    env.run(LEAD, _field(env, 2 * LEAD, FIELD))
    snapshot = env.save_state()
    straight = env.run(AFTER)
    env.clear().run(20.0)
    env.restore_state(snapshot)
    _assert_identical(straight, env.run(AFTER))


def test_a_snapshot_belongs_to_the_env_that_saved_it():
    env = _env()
    with pytest.raises(RuntimeError, match="first run"):
        env.save_state()
    env.run(10.0)
    snapshot = env.save_state()
    with pytest.raises(TypeError):
        env.restore_state(object())
    snapshot.release()
    with pytest.raises(ValueError, match="released"):
        env.restore_state(snapshot)
