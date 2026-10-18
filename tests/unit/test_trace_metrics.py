import numpy as np
import pytest

from livn.optimization.metrics import reconstruction, spike_f1, threshold_spikes


def test_spike_f1_hand_built():
    out = spike_f1([10.0, 20.0, 30.0], [10.5, 29.0, 50.0], tol=1.0)
    assert (out["tp"], out["fp"], out["fn"]) == (2, 1, 1)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)

    far = spike_f1([10.0, 20.0, 30.0], [10.5, 40.0, 50.0], tol=1.0)
    assert (far["tp"], far["fp"], far["fn"]) == (1, 2, 2)
    assert far["f1"] == pytest.approx(2 / 6)

    perfect = spike_f1([1.0, 5.0], [5.0, 1.0], tol=0.1)
    assert perfect["f1"] == 1.0 and perfect["tp"] == 2


def test_spike_f1_is_one_to_one_and_inclusive():
    # two predictions near one recorded spike: only one can claim it
    out = spike_f1([9.5, 10.5], [10.0], tol=1.0)
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 0)
    # the nearer wins; the other stays unmatched (and can claim the next one)
    out = spike_f1([10.0, 12.0], [10.0, 11.0], tol=1.0)
    assert out["tp"] == 2
    # the boundary is inclusive
    assert spike_f1([0.0], [1.0], tol=1.0)["tp"] == 1
    assert spike_f1([0.0], [1.0 + 1e-6], tol=1.0)["tp"] == 0


def test_spike_f1_empty_trains():
    both = spike_f1([], [])
    assert np.isnan(both["f1"]) and both["tp"] == 0
    none_pred = spike_f1([], [1.0])
    assert none_pred["recall"] == 0.0 and np.isnan(none_pred["precision"])
    assert none_pred["f1"] == 0.0


def test_reconstruction_on_synthetic_traces():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(500, 4))
    perfect = reconstruction(obs, obs)
    assert perfect["mse"] == 0.0
    np.testing.assert_allclose(perfect["corr"], 1.0)
    assert perfect["corr_mean"] == pytest.approx(1.0)

    flipped = reconstruction(-obs, obs)
    np.testing.assert_allclose(flipped["corr"], -1.0)

    # correlation ignores gain and offset, the MSE does not
    scaled = reconstruction(2.0 * obs + 1.0, obs)
    assert scaled["corr_mean"] == pytest.approx(1.0)
    assert scaled["mse"] > 1.0

    noisy = reconstruction(obs + rng.normal(size=obs.shape), obs)
    assert noisy["mse"] == pytest.approx(1.0, rel=0.15)
    assert noisy["corr"].shape == (4,)
    assert noisy["corr_mean"] == pytest.approx(1 / np.sqrt(2), abs=0.05)

    # a flat channel has no defined correlation and counts as 0
    flat = obs.copy()
    flat[:, 0] = 0.0
    assert reconstruction(flat, obs)["corr"][0] == 0.0
    # the time axis is selectable
    assert reconstruction(obs.T, obs.T, axis=1)["corr"].shape == (4,)


def _noisy_trace(spike_samples, n=4000, depth=-12.0, seed=0):
    rng = np.random.default_rng(seed)
    trace = rng.normal(scale=1.0, size=n)
    for s in spike_samples:
        trace[s] += depth
    return trace


def test_threshold_spikes_finds_negative_spikes():
    dt = 0.1  # ms
    truth = [500, 1500, 2600, 3500]
    times = threshold_spikes(_noisy_trace(truth), dt=dt)
    np.testing.assert_allclose(times, np.asarray(truth) * dt)
    assert spike_f1(times, np.asarray(truth) * dt, 0.5)["f1"] == 1.0

    assert threshold_spikes(-_noisy_trace(truth), dt=dt).size == 0
    pos = threshold_spikes(-_noisy_trace(truth), dt=dt, polarity="positive")
    np.testing.assert_allclose(pos, np.asarray(truth) * dt)
    assert threshold_spikes(np.random.default_rng(1).normal(size=4000), dt=dt).size == 0


def test_threshold_spikes_refractory_and_channels():
    dt = 0.1
    trace = _noisy_trace([1000, 1050])
    assert threshold_spikes(trace, dt, refractory=10.0).size == 1
    assert threshold_spikes(trace, dt, refractory=2.0).size == 2

    both = np.stack([_noisy_trace([300]), _noisy_trace([700], seed=3)], axis=1)
    per_channel = threshold_spikes(both, dt)
    assert len(per_channel) == 2
    np.testing.assert_allclose(per_channel[0], [30.0])
    np.testing.assert_allclose(per_channel[1], [70.0])

    with pytest.raises(ValueError):
        threshold_spikes(trace, dt, polarity="up")


def test_explained_variance_is_in_ms():
    from livn.optimization.metrics import EV_DT, EV_SIGMA, explained_variance_ratio

    assert (EV_DT, EV_SIGMA) == (0.05, 10.0)
    trial = [20.0, 80.0, 150.0]
    # duration, spike times and sigma are all ms: identical trials explain each other
    same = explained_variance_ratio(trial, [trial, trial], duration=200.0)
    assert same == pytest.approx(1.0)
    # spikes 300 ms away from the data lie outside a 200 ms window and are dropped
    off = explained_variance_ratio([320.0, 380.0], [trial, trial], duration=200.0)
    assert off is None or off < 0.1
