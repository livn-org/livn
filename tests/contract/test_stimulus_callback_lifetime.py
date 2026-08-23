from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("livn")

from livn.backend import backend  # noqa: E402

pytestmark = pytest.mark.skipif(
    backend() != "neuron", reason="the callback is a NEURON cvode registration"
)


@pytest.fixture(scope="module")
def env():
    from testing import livn_test_env, livn_test_mea

    return livn_test_env(io=livn_test_mea()).init()


def _pulse(env, duration, onset, dt=0.5, amplitude=50.0, channel=0):
    n_steps = int(round(duration / dt))
    channel_inputs = np.zeros((n_steps, len(env.io.channel_ids)))
    lo = int(round(onset / dt))
    channel_inputs[lo, channel] = -amplitude
    channel_inputs[lo + 1, channel] = amplitude
    stimulus = env.cell_stimulus(channel_inputs)
    stimulus.dt = dt
    return stimulus


def test_the_callback_object_is_stable(env):
    assert env._update_extracellular is not env._update_extracellular
    assert env._stim_cb is env._stim_cb


def test_removal_is_passed_what_registration_used(env):
    import inspect

    register = inspect.getsource(type(env)._setup_extracellular)
    unregister = inspect.getsource(type(env)._unregister_stim_callback)

    assert "_register_callback(self._h.cvode, self._stim_cb)" in register
    assert "_unregister_callback(self._h.cvode, self._stim_cb)" in unregister


def test_the_pulse_arrives_at_the_same_time_on_every_run(env):
    duration, onset, bin_ms = 120.0, 60.0, 10.0
    bins = np.arange(0.0, duration + bin_ms, bin_ms)

    def run(stimulus):
        env.clear()
        env.record_spikes()
        result = env.run(duration, stimulus=stimulus)
        times = np.asarray(result.spike_times if result.spike_times is not None else [])
        return np.histogram(times, bins=bins)[0].astype(int)

    control = run(None)

    peaks = []
    for _ in range(4):
        evoked = run(_pulse(env, duration, onset)) - control
        if evoked.max() <= 0:
            pytest.skip("this system does not spike under the probe stimulus")
        peaks.append(float(bins[int(np.argmax(evoked))]))

    assert peaks == [onset] * 4, (
        f"the response moved across runs: {peaks} (pulse at {onset} ms). "
        "Stimulus callbacks are accumulating, so the array plays fast."
    )
