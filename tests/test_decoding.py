import os

import numpy as np
import pytest

from livn.env import Env
from livn.decoding import (
    Slice,
    Pipe,
    MeanFiringRate,
    ActiveFraction,
    Stability,
    LFP,
    AvalancheAnalysis,
)
from livn.backend import backend


@pytest.fixture
def env_response(request):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])

    response = request.config.cache.get("livn/env/response-" + env.system.name, None)
    if response is None:
        env.init()

        env.record_spikes()
        env.record_voltage()
        env.record_membrane_current()

        t_end = 250
        inputs = np.zeros([t_end, env.io.num_channels])
        for r in range(20):
            for c in range(env.io.num_channels):
                inputs[50 + r, c] = 1.5

        stimulus = env.cell_stimulus(inputs)
        response = env.run(t_end, stimulus=stimulus)

        request.config.cache.set(
            "livn/env/response-" + env.system.name, [r.tolist() for r in response]
        )

    return tuple([np.array(r) for r in response])


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_slice_decoding(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()
    env.record_spikes()
    env.record_voltage()
    env.record_membrane_current()

    start = 100
    duration = 50
    recording_dt = 0.1

    ii, tt, iv, v, im, mp = Slice(start=start, stop=start + duration)(
        env, *env_response
    )
    orig_ii, orig_tt, orig_iv, orig_v, orig_im, orig_m = env_response

    # spike times
    assert tt[tt < start].shape[0] == 0
    assert tt[tt >= start + duration].shape[0] == 0

    expected_time_steps = int(duration / recording_dt)
    assert v.shape[0] == orig_v.shape[0]
    assert v.shape[1] == expected_time_steps

    assert mp.shape[0] == orig_m.shape[0]
    assert mp.shape[1] == expected_time_steps


class MockEnv:
    def __init__(self, n_units=100, n_channels=16):
        self.n_units = n_units
        self.n_channels = n_channels
        self.comm = None

        class MockSystem:
            def __init__(self, n_units):
                self.gids = list(range(n_units))

        self.system = MockSystem(n_units)

    def potential_recording(self, m):
        if m is None:
            return None
        n_samples = m.shape[1]
        return np.random.randn(self.n_channels, n_samples).astype(np.float32)


def make_mock_spikes(n_spikes, n_units, duration_ms, seed=42):
    np.random.seed(seed)
    it = np.random.randint(0, n_units, n_spikes)
    tt = np.random.uniform(0, duration_ms, n_spikes)
    tt = np.sort(tt)
    return it.astype(np.int32), tt.astype(np.float64)


def make_mock_membrane_current(n_units, duration_ms, dt=0.1, seed=42):
    np.random.seed(seed)
    n_samples = int(duration_ms / dt)
    im = np.arange(n_units)
    m = np.random.randn(n_units, n_samples).astype(np.float32)
    return im, m


class TestMeanFiringRate:
    def test_basic_computation(self):
        env = MockEnv(n_units=10)
        duration = 1000  # 1 second

        # 20 spikes over 1 second with 10 units = 2 Hz
        it = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        tt = np.linspace(0, 999, 20)

        mfr = MeanFiringRate(duration=duration)
        result = mfr(env, it, tt, None, None, None, None)

        assert result is not None
        assert "rate_hz" in result
        assert result["total_spikes"] == 20
        assert result["n_units"] == 10
        # 20 spikes / (10 units * 1 second) = 2 Hz
        assert abs(result["rate_hz"] - 2.0) < 0.01

    def test_empty_spikes(self):
        env = MockEnv(n_units=10)
        it = np.array([])
        tt = np.array([])

        mfr = MeanFiringRate(duration=1000)
        result = mfr(env, it, tt, None, None, None, None)

        assert result is not None
        assert result["total_spikes"] == 0
        assert result["rate_hz"] < 1e-6

class TestActiveFraction:
    def test_full_activation(self):
        env = MockEnv(n_units=10)

        it = np.arange(10)
        tt = np.linspace(0, 999, 10)

        af = ActiveFraction(duration=1000)
        result = af(env, it, tt, None, None, None, None)

        assert result is not None
        assert result["active_fraction"] == 1.0
        assert result["active_units"] == 10
        assert len(result["silent_units"]) == 0

    def test_partial_activation(self):
        env = MockEnv(n_units=10)

        it = np.array([0, 1, 2, 3, 4])
        tt = np.linspace(0, 999, 5)

        af = ActiveFraction(duration=1000)
        result = af(env, it, tt, None, None, None, None)

        assert result["active_fraction"] == 0.5
        assert result["active_units"] == 5

    def test_min_spikes_threshold(self):
        env = MockEnv(n_units=5)

        it = np.array([0, 0, 1, 2, 3, 4])
        tt = np.linspace(0, 999, 6)

        af = ActiveFraction(duration=1000, min_spikes=2)
        result = af(env, it, tt, None, None, None, None)

        assert result["active_units"] == 1


class TestStability:
    def test_stable_activity(self):
        env = MockEnv(n_units=10)
        duration = 5000

        np.random.seed(42)
        n_spikes = 50  # 1 Hz per unit over 5 seconds
        it, tt = make_mock_spikes(n_spikes, 10, duration)

        stability = Stability(duration=duration, max_rate_hz=20, min_rate_hz=0.01)
        result = stability(env, it, tt, None, None, None, None)

        assert result is not None
        assert "is_stable" in result
        assert "tail_mean_hz" in result
        assert "global_mean_hz" in result

    def test_runaway_detection(self):
        env = MockEnv(n_units=10)
        duration = 2000

        it = np.repeat(np.arange(10), 50)  # 500 spikes
        tt = np.random.uniform(1500, 2000, 500)  # all in last 500ms

        stability = Stability(duration=duration, tail_window=500, max_rate_hz=10)
        result = stability(env, it, tt, None, None, None, None)

        assert result["is_runaway"] is True
        assert result["is_stable"] is False

    def test_quiescence_detection(self):
        env = MockEnv(n_units=10)
        duration = 5000

        it = np.array([], dtype=np.int32)
        tt = np.array([], dtype=np.float64)

        stability = Stability(duration=duration, min_rate_hz=0.05)
        result = stability(env, it, tt, None, None, None, None)

        assert result["global_mean_hz"] == 0.0
        assert result["is_quiescent"] is True
        assert result["is_stable"] is False


class TestLFP:
    def test_basic_extraction(self):
        env = MockEnv(n_units=50, n_channels=8)
        duration = 1000

        im, m = make_mock_membrane_current(50, duration)

        lfp = LFP(duration=duration, downsample_hz=1000)
        result = lfp(env, None, None, None, None, im, m)

        assert result is not None
        assert "lfp" in result
        assert result["n_channels"] == 8
        assert result["sample_rate_hz"] > 0

    def test_downsampling(self):
        env = MockEnv(n_units=50, n_channels=4)
        duration = 1000
        env.membrane_current_recording_dt = 0.1  # 10kHz

        im, m = make_mock_membrane_current(50, duration, dt=0.1)

        # 10kHz to 1kHz
        lfp = LFP(duration=duration, downsample_hz=1000)
        result = lfp(env, None, None, None, None, im, m)

        assert result["sample_rate_hz"] == 1000.0

    def test_channel_selection(self):
        env = MockEnv(n_units=50, n_channels=16)
        duration = 500

        im, m = make_mock_membrane_current(50, duration)

        lfp = LFP(duration=duration, channels=[0, 1, 2, 3])
        result = lfp(env, None, None, None, None, im, m)

        assert result["n_channels"] == 4

    def test_no_membrane_current(self):
        env = MockEnv()

        lfp = LFP(duration=1000)
        result = lfp(env, None, None, None, None, None, None)

        assert result is not None
        assert result["n_channels"] == 0


class TestLFPBandPower:
    def test_basic_computation(self):
        env = MockEnv(n_units=50, n_channels=1)
        duration_ms = 4000

        im, m = make_mock_membrane_current(50, duration_ms, dt=1.0)  # 1ms dt for 1000Hz

        lfp_decoder = LFP(
            duration=duration_ms,
            compute_band_power={
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "beta": (13.0, 30.0),
            },
            nperseg=512,
        )
        result = lfp_decoder(env, None, None, None, None, im, m)

        assert result is not None
        assert "theta" in result
        assert "delta" in result
        assert "beta" in result
        assert "broadband" in result
        assert "lfp" in result

        assert result["theta"] >= 0
        assert result["delta"] >= 0
        assert result["beta"] >= 0

        assert "theta_relative" in result
        assert "delta_relative" in result

    def test_no_band_power(self):
        env = MockEnv(n_units=50, n_channels=4)
        duration = 1000
        im, m = make_mock_membrane_current(50, duration)

        lfp = LFP(duration=duration, compute_band_power=False)
        result = lfp(env, None, None, None, None, im, m)

        assert result is not None
        assert "lfp" in result
        assert "delta" not in result
        assert "theta" not in result
        assert "broadband" not in result

    def test_default_bands(self):
        env = MockEnv(n_units=50, n_channels=4)
        duration = 2000
        im, m = make_mock_membrane_current(50, duration)

        lfp = LFP(duration=duration, compute_band_power=True)
        result = lfp(env, None, None, None, None, im, m)

        assert result is not None
        assert "delta" in result
        assert "theta" in result
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result
        assert "broadband" in result

    def test_relative_power(self):
        env = MockEnv(n_units=50, n_channels=1)
        duration = 2000
        im, m = make_mock_membrane_current(50, duration, dt=2.0)  # 500Hz

        lfp = LFP(
            duration=duration,
            compute_band_power={"delta": (1.0, 4.0), "theta": (4.0, 8.0)},
        )
        result = lfp(env, None, None, None, None, im, m)

        assert "delta_relative" in result
        assert "theta_relative" in result
        total_relative = result["delta_relative"] + result["theta_relative"]
        assert total_relative <= 1.0


class TestPipe:
    def test_basic_chaining(self):
        env = MockEnv(n_units=10)
        duration = 1000

        it, tt = make_mock_spikes(20, 10, duration)

        # Create pipeline: compute mean firing rate
        pipeline = Pipe(
            duration=duration,
            stages=[MeanFiringRate(duration=duration)],
        )

        result = pipeline(env, it, tt, None, None, None, None)

        assert result is not None
        assert "rate_hz" in result

    def test_repr(self):
        pipeline = Pipe(
            duration=1000,
            stages=[
                Slice(start=100, stop=500),
                MeanFiringRate(duration=400),
            ],
        )
        repr_str = repr(pipeline)
        assert "Pipe" in repr_str
        assert "Slice" in repr_str
        assert "MeanFiringRate" in repr_str

    def test_invalid_stage(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Pipe(duration=1000, stages=["not_callable"])


class TestAvalancheAnalysis:
    def test_basic_computation(self):
        env = MockEnv(n_units=10)
        duration = 1000

        tt = np.concatenate(
            [
                # three avalanches
                np.random.uniform(10, 20, 5),
                np.random.uniform(60, 70, 8),
                np.random.uniform(120, 130, 3),
            ]
        )
        it = np.random.randint(0, 10, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=5.0)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert "n_avalanches" in result
        assert "mean_size" in result
        assert "mean_duration" in result
        assert "branching_ratio" in result
        assert "size_power_law_r2" in result

        assert result["n_avalanches"] > 0
        assert result["mean_size"] > 0

    def test_critical_branching_pattern(self):
        env = MockEnv(n_units=50)
        duration = 2000

        # Generate cascading pattern with branching ratio ~ 1
        # Each bin generates approximately the same number of spikes as previous
        tt_list = []
        t_current = 0
        bin_width = 4.0
        n_spikes_current = 5

        for _ in range(20):
            # add spikes to this bin
            spikes = np.random.uniform(
                t_current, t_current + bin_width, n_spikes_current
            )
            tt_list.extend(spikes.tolist())

            # next bin has approximately same number
            n_spikes_current = max(1, int(n_spikes_current + np.random.randn()))
            t_current += bin_width

        tt = np.array(tt_list)
        it = np.random.randint(0, 50, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=bin_width)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert 0.5 < result["branching_ratio"] < 1.5

    def test_subcritical_pattern(self):
        env = MockEnv(n_units=30)
        duration = 1000

        tt_list = []
        t_current = 0
        bin_width = 4.0
        n_spikes_current = 20

        while n_spikes_current > 0 and t_current < duration:
            spikes = np.random.uniform(
                t_current, t_current + bin_width, n_spikes_current
            )
            tt_list.extend(spikes.tolist())
            # decay
            n_spikes_current = max(0, int(n_spikes_current * 0.7))
            t_current += bin_width

        tt = np.array(tt_list)
        it = np.random.randint(0, 30, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=bin_width)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert result["branching_ratio"] < 1.0

    def test_supercritical_pattern(self):
        env = MockEnv(n_units=30)
        duration = 500

        tt_list = []
        t_current = 0
        bin_width = 4.0
        n_spikes_current = 2

        while n_spikes_current < 50 and t_current < duration:
            spikes = np.random.uniform(
                t_current, t_current + bin_width, n_spikes_current
            )
            tt_list.extend(spikes.tolist())
            # growing
            n_spikes_current = int(n_spikes_current * 1.5) + 1
            t_current += bin_width

        tt = np.array(tt_list)
        it = np.random.randint(0, 30, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=bin_width)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert result["branching_ratio"] > 1.0

    def test_single_spike_avalanche(self):
        env = MockEnv(n_units=10)
        duration = 1000

        tt = np.array([100.0, 200.0, 300.0, 400.0])
        it = np.array([0, 1, 2, 3])

        aa = AvalancheAnalysis(duration=duration, bin_width=5.0)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert result["n_avalanches"] >= 4
        assert result["mean_size"] >= 1.0
        assert result["mean_duration"] >= 1.0

    def test_empty_spikes(self):
        env = MockEnv(n_units=10)
        duration = 1000

        tt = np.array([])
        it = np.array([])

        aa = AvalancheAnalysis(duration=duration, bin_width=4.0)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert result["n_avalanches"] == 0
        assert result["mean_size"] == 0.0
        assert result["mean_duration"] == 0.0
        assert result["branching_ratio"] == 0.0

    def test_continuous_activity(self):
        env = MockEnv(n_units=20)
        duration = 1000

        np.random.seed(123)
        tt = np.sort(np.random.uniform(0, duration, 500))
        it = np.random.randint(0, 20, 500)

        aa = AvalancheAnalysis(duration=duration, bin_width=4.0)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert result["n_avalanches"] > 0

    def test_bin_width_sensitivity(self):
        env = MockEnv(n_units=10)
        duration = 1000

        tt = np.array([10, 15, 20, 50, 55, 60, 100, 105])
        it = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        aa_small = AvalancheAnalysis(duration=duration, bin_width=2.0)
        result_small = aa_small(env, it, tt, None, None, None, None)

        aa_large = AvalancheAnalysis(duration=duration, bin_width=50.0)
        result_large = aa_large(env, it, tt, None, None, None, None)

        assert result_large["mean_size"] >= result_small["mean_size"]

    def test_power_law_fitting(self):
        env = MockEnv(n_units=100)
        duration = 5000

        tt_list = []
        t_current = 0
        bin_width = 4.0

        for _ in range(100):
            n_spikes = np.random.randint(1, 20)
            spikes = np.random.uniform(t_current, t_current + bin_width, n_spikes)
            tt_list.extend(spikes.tolist())
            t_current += bin_width + np.random.uniform(5, 20)

        tt = np.array(tt_list)
        it = np.random.randint(0, 100, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=bin_width)
        result = aa(env, it, tt, None, None, None, None)

        assert result is not None
        assert -1.0 <= result["size_power_law_r2"] <= 1.0


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_mean_firing_rate_integration(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()

    it, tt, iv, vv, im, m = env_response

    mfr = MeanFiringRate(duration=250)
    result = mfr(env, it, tt, iv, vv, im, m)

    assert result is not None
    assert result["rate_hz"] >= 0
    assert result["total_spikes"] >= 0


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_active_fraction_integration(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()

    it, tt, iv, vv, im, m = env_response

    af = ActiveFraction(duration=250)
    result = af(env, it, tt, iv, vv, im, m)

    assert result is not None
    assert 0 <= result["active_fraction"] <= 1


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_stability_integration(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()

    it, tt, iv, vv, im, m = env_response

    stability = Stability(duration=250, tail_window=100)
    result = stability(env, it, tt, iv, vv, im, m)

    assert result is not None
    assert "is_stable" in result
    assert isinstance(result["is_stable"], bool)


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_lfp_integration(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()
    env.record_membrane_current()

    it, tt, iv, vv, im, m = env_response

    lfp = LFP(duration=250)
    result = lfp(env, it, tt, iv, vv, im, m)

    assert result is not None
    assert "lfp" in result


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_pipeline_integration(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()
    env.record_spikes()
    env.record_membrane_current()

    it, tt, iv, vv, im, m = env_response

    pipeline = Pipe(
        duration=250,
        stages=[
            Slice(start=50, stop=200),
            MeanFiringRate(duration=150),
        ],
    )

    result = pipeline(env, it, tt, iv, vv, im, m)

    assert result is not None
    assert "rate_hz" in result


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_avalanche_analysis_integration(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()

    it, tt, iv, vv, im, m = env_response

    aa = AvalancheAnalysis(duration=250, bin_width=4.0)
    result = aa(env, it, tt, iv, vv, im, m)

    assert result is not None
    assert "n_avalanches" in result
    assert "branching_ratio" in result
    assert result["n_avalanches"] >= 0
    assert result["branching_ratio"] >= 0.0
