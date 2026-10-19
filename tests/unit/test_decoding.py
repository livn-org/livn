from typing import ClassVar

import numpy as np
import pytest

from livn.decoding import (
    LFP,
    ActiveFraction,
    AvalancheAnalysis,
    BurstAnatomy,
    ChannelRecording,
    MeanFiringRate,
    Pipe,
    PopulationActiveFraction,
    PopulationSpikeDensity,
    Slice,
    Stability,
)
from livn.run import Run


def _recording(it=None, tt=None, iv=None, vv=None, im=None, mp=None, dt=0.1):
    return (
        Run(duration=250.0)
        .add_spikes(it, tt)
        .add_voltage(iv, vv, dt=dt)
        .add_current(im, mp, dt=dt)
    )


STIMULUS_AMPLITUDE = 1.5
RESPONSE_DURATION = 250


class MockEnv:
    def __init__(self, n_units=100, n_channels=16):
        self.n_units = n_units
        self.n_channels = n_channels
        self.comm = None
        self.calls = {"spikes": 0, "voltages": 0, "membrane_currents": 0}

        class MockIO:
            def __init__(self, n_channels):
                self.channel_ids = np.arange(n_channels, dtype=np.int32)

        self.io = MockIO(n_channels)

        class MockSystem:
            def __init__(self, n_units):
                self.gids = list(range(n_units))

        self.system = MockSystem(n_units)

    def record_spikes(self):
        self.calls["spikes"] += 1
        return self

    def record_voltage(self):
        self.calls["voltages"] += 1
        return self

    def record_membrane_current(self):
        self.calls["membrane_currents"] += 1
        return self

    def channel_recording(self, it, tt):
        if it is None or tt is None:
            return {}, {}

        mask_even = (it % 2) == 0
        return {
            0: it[mask_even],
            1: it[~mask_even],
        }, {
            0: tt[mask_even],
            1: tt[~mask_even],
        }

    def potential_recording(self, recording=None, conditions=None, key=None):
        m = None if conditions is None else conditions.membrane_currents_uA
        if m is None:
            return None
        n_samples = m.shape[1]
        return np.random.randn(self.n_channels, n_samples).astype(np.float32)


def make_mock_spikes(n_spikes, n_units, duration, seed=42):
    np.random.seed(seed)
    it = np.random.randint(0, n_units, n_spikes)
    tt = np.random.uniform(0, duration, n_spikes)
    tt = np.sort(tt)
    return it.astype(np.int32), tt.astype(np.float64)


def make_mock_membrane_current(n_units, duration, dt=0.1, seed=42):
    np.random.seed(seed)
    n_samples = int(duration / dt)
    im = np.arange(n_units)
    m = np.random.randn(n_units, n_samples).astype(np.float32)
    return im, m


class TestMeanFiringRate:
    def test_basic_computation(self):
        env = MockEnv(n_units=10)
        duration = 1000

        it = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        tt = np.linspace(0, 999, 20)

        mfr = MeanFiringRate(duration=duration)
        result = mfr(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert "rate" in result
        assert result["total_spikes"] == 20
        assert result["n_units"] == 10
        assert abs(result["rate"] - 2.0) < 0.01
        assert result["duration"] == 1000.0  # ms

    def test_empty_spikes(self):
        env = MockEnv(n_units=10)
        it = np.array([])
        tt = np.array([])

        mfr = MeanFiringRate(duration=1000)
        result = mfr(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert result["total_spikes"] == 0
        assert result["rate"] < 1e-6


class TestActiveFraction:
    def test_full_activation(self):
        env = MockEnv(n_units=10)

        it = np.arange(10)
        tt = np.linspace(0, 999, 10)

        af = ActiveFraction(duration=1000)
        result = af(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert result["active_fraction"] == 1.0
        assert result["active_units"] == 10
        assert len(result["silent_units"]) == 0

    def test_partial_activation(self):
        env = MockEnv(n_units=10)

        it = np.array([0, 1, 2, 3, 4])
        tt = np.linspace(0, 999, 5)

        af = ActiveFraction(duration=1000)
        result = af(_recording(it, tt, None, None, None, None), env)

        assert result["active_fraction"] == 0.5
        assert result["active_units"] == 5

    def test_min_spikes_threshold(self):
        env = MockEnv(n_units=5)

        it = np.array([0, 0, 1, 2, 3, 4])
        tt = np.linspace(0, 999, 6)

        af = ActiveFraction(duration=1000, min_spikes=2)
        result = af(_recording(it, tt, None, None, None, None), env)

        assert result["active_units"] == 1


class MockPopulationEnv:
    def __init__(self):
        self.comm = None

        class MockSystem:
            population_ranges: ClassVar[dict] = {"A": (0, 10), "B": (10, 4)}

        self.system = MockSystem()
        self.cells = {"A": dict.fromkeys([0, 1, 2, 3]), "B": dict.fromkeys([10, 11])}


class TestPopulationSpikeDensity:
    @staticmethod
    def _measure(it, tt, duration=250.0):
        return PopulationSpikeDensity(
            duration=int(duration),
            temporal_resolution=5.0,
            stability_resolution=5.0,
        )(Run(duration=duration).add_spikes(it, tt), MockPopulationEnv())

    def test_population_rate_is_the_active_rate_times_the_active_fraction(self):
        it = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        tt = np.array([10.0, 60.0, 110.0, 160.0, 30.0, 80.0, 130.0, 180.0])

        r = self._measure(it, tt)

        assert r["fraction_active"]["A"] == pytest.approx(0.5)
        assert r["mean_rate_population"]["A"] == pytest.approx(
            r["mean_rate"]["A"] * r["fraction_active"]["A"]
        )
        assert r["mean_rate_population"]["A"] < r["mean_rate"]["A"]

    def test_a_near_silent_population_reads_fast_by_the_active_rate_only(self):
        it = np.zeros(20, dtype=np.int64)
        tt = np.linspace(5.0, 245.0, 20)

        r = self._measure(it, tt)

        assert r["fraction_active"]["A"] == pytest.approx(0.25)
        assert r["mean_rate"]["A"] > 20.0, "the one cell that fires, fires fast"
        assert r["mean_rate_population"]["A"] == pytest.approx(
            r["mean_rate"]["A"] / 4.0
        )

    def test_both_rates_are_zero_when_nothing_fires(self):
        r = self._measure(np.array([], dtype=np.int64), np.array([]))

        assert r["mean_rate"]["A"] == 0.0
        assert r["mean_rate_population"]["A"] == 0.0
        assert r["fraction_active"]["A"] == 0.0


class TestPopulationActiveFraction:
    def test_fraction_is_over_cells_per_bin(self):
        it = np.array([0, 0, 1, 10, 2])
        tt = np.array([1.0, 2.0, 30.0, 10.0, 60.0])

        result = PopulationActiveFraction(duration=100, bin_size=50.0)(
            Run(duration=100.0).add_spikes(it, tt), MockPopulationEnv()
        )

        assert result["n_bins"] == 2
        assert result["mean_active_fraction"]["A"] == pytest.approx(0.375)
        assert result["std_active_fraction"]["A"] == pytest.approx(0.125)
        assert result["mean_active_fraction"]["B"] == pytest.approx(0.25)
        assert result["std_active_fraction"]["B"] == pytest.approx(0.25)

    def test_populations_with_no_simulated_cells_are_absent(self):
        env = MockPopulationEnv()
        env.cells = {"A": dict.fromkeys([0, 1])}

        result = PopulationActiveFraction(duration=100, bin_size=50.0)(
            Run(duration=100.0).add_spikes(np.array([0]), np.array([1.0])),
            env,
        )

        assert set(result["mean_active_fraction"]) == {"A"}


class TestStability:
    def test_stable_activity(self):
        env = MockEnv(n_units=10)
        duration = 5000

        np.random.seed(42)
        n_spikes = 50
        it, tt = make_mock_spikes(n_spikes, 10, duration)

        stability = Stability(duration=duration, max_rate=20, min_rate=0.01)
        result = stability(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert "is_stable" in result
        assert "tail_mean" in result
        assert "global_mean" in result

    def test_runaway_detection(self):
        env = MockEnv(n_units=10)
        duration = 2000

        it = np.repeat(np.arange(10), 50)
        tt = np.random.uniform(1500, 2000, 500)

        stability = Stability(duration=duration, tail=500, max_rate=10)
        result = stability(_recording(it, tt, None, None, None, None), env)

        assert result["is_runaway"] is True
        assert result["is_stable"] is False

    def test_quiescence_detection(self):
        env = MockEnv(n_units=10)
        duration = 5000

        it = np.array([], dtype=np.int32)
        tt = np.array([], dtype=np.float64)

        stability = Stability(duration=duration, min_rate=0.05)
        result = stability(_recording(it, tt, None, None, None, None), env)

        assert result["global_mean"] == 0.0
        assert result["is_quiescent"] is True
        assert result["is_stable"] is False


class TestLFP:
    def test_basic_extraction(self):
        env = MockEnv(n_units=50, n_channels=8)
        duration = 1000

        im, m = make_mock_membrane_current(50, duration)

        lfp = LFP(duration=duration, downsample_rate=1000)
        result = lfp(_recording(None, None, None, None, im, m), env)

        assert result is not None
        assert "lfp" in result
        assert result["n_channels"] == 8
        assert result["sample_rate"] > 0

    def test_downsampling(self):
        env = MockEnv(n_units=50, n_channels=4)
        duration = 1000
        env.membrane_current_recording_dt = 0.1

        im, m = make_mock_membrane_current(50, duration, dt=0.1)

        lfp = LFP(duration=duration, downsample_rate=1000)
        result = lfp(_recording(None, None, None, None, im, m), env)

        assert result["sample_rate"] == 1000.0

    def test_channel_selection(self):
        env = MockEnv(n_units=50, n_channels=16)
        duration = 500

        im, m = make_mock_membrane_current(50, duration)

        lfp = LFP(duration=duration, channels=[0, 1, 2, 3])
        result = lfp(_recording(None, None, None, None, im, m), env)

        assert result["n_channels"] == 4

    def test_no_membrane_current(self):
        env = MockEnv()

        lfp = LFP(duration=1000)
        result = lfp(_recording(None, None, None, None, None, None), env)

        assert result is not None
        assert result["n_channels"] == 0


class TestLFPBandPower:
    def test_basic_computation(self):
        env = MockEnv(n_units=50, n_channels=1)
        duration = 4000

        im, m = make_mock_membrane_current(50, duration, dt=1.0)

        lfp_decoder = LFP(
            duration=duration,
            compute_band_power={
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "beta": (13.0, 30.0),
            },
            nperseg=512,
        )
        result = lfp_decoder(_recording(None, None, None, None, im, m), env)

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
        result = lfp(_recording(None, None, None, None, im, m), env)

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
        result = lfp(_recording(None, None, None, None, im, m), env)

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
        im, m = make_mock_membrane_current(50, duration, dt=2.0)

        lfp = LFP(
            duration=duration,
            compute_band_power={"delta": (1.0, 4.0), "theta": (4.0, 8.0)},
        )
        result = lfp(_recording(None, None, None, None, im, m), env)

        assert "delta_relative" in result
        assert "theta_relative" in result
        total_relative = result["delta_relative"] + result["theta_relative"]
        assert total_relative <= 1.0


class TestPipe:
    def test_basic_chaining(self):
        env = MockEnv(n_units=10)
        duration = 1000

        it, tt = make_mock_spikes(20, 10, duration)

        pipeline = Pipe(
            duration=duration,
            stages=[MeanFiringRate(duration=duration)],
        )

        result = pipeline(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert "rate" in result

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


class _ImmutableEnv:
    def __init__(self, recorded=()):
        self.recorded = tuple(recorded)

    def record_spikes(self):
        return _ImmutableEnv((*self.recorded, "spikes"))

    def record_voltage(self, **options):
        return _ImmutableEnv((*self.recorded, "voltage"))

    def record_membrane_current(self):
        return _ImmutableEnv((*self.recorded, "membrane_current"))


class TestSetupReturnsTheEnv:
    def test_a_decoding_hands_back_the_env_that_records(self):
        env = _ImmutableEnv()
        recording = ChannelRecording(duration=100).setup(env)
        assert env.recorded == ()
        assert recording.recorded == ("spikes", "voltage", "membrane_current")

    def test_a_pipe_chains_its_stages_copies(self):
        pipeline = Pipe(
            duration=100,
            stages=[
                MeanFiringRate(duration=100),
                ChannelRecording(duration=100, voltages=False),
            ],
        )
        recording = pipeline.setup(_ImmutableEnv())
        assert recording.recorded == ("spikes", "membrane_current")

    def test_every_recording_decoding_returns_the_copy_that_records(self):
        from livn.decoding import (
            ExtracellularRecruitment,
            PeriStimulusHistogram,
            RecruitmentCurve,
            StimulusResponse,
            Waveforms,
        )

        spikes_only = (
            StimulusResponse(duration=100),
            RecruitmentCurve(duration=100),
            PeriStimulusHistogram(duration=100),
            ExtracellularRecruitment(duration=100),
        )
        for decoding in spikes_only:
            assert decoding.setup(_ImmutableEnv()).recorded == ("spikes",)
        both = ExtracellularRecruitment(duration=100, ceiling=50.0)
        assert both.setup(_ImmutableEnv()).recorded == ("spikes", "voltage")
        assert Waveforms(duration=100).setup(_ImmutableEnv()).recorded == (
            "spikes",
            "voltage",
        )

    def test_a_decoding_with_nothing_to_enable_returns_the_env_it_was_given(self):
        env = _ImmutableEnv()
        assert MeanFiringRate(duration=100).setup(env) is None
        assert (
            Pipe(duration=100, stages=[MeanFiringRate(duration=100)]).setup(env) is env
        )


class TestAvalancheAnalysis:
    def test_basic_computation(self):
        env = MockEnv(n_units=10)
        duration = 1000

        tt = np.concatenate(
            [
                np.random.uniform(10, 20, 5),
                np.random.uniform(60, 70, 8),
                np.random.uniform(120, 130, 3),
            ]
        )
        it = np.random.randint(0, 10, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=5.0)
        result = aa(_recording(it, tt, None, None, None, None), env)

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

        tt_list = []
        t_current = 0
        bin_width = 4.0
        n_spikes_current = 5

        for _ in range(20):
            spikes = np.random.uniform(
                t_current, t_current + bin_width, n_spikes_current
            )
            tt_list.extend(spikes.tolist())

            n_spikes_current = max(1, int(n_spikes_current + np.random.randn()))
            t_current += bin_width

        tt = np.array(tt_list)
        it = np.random.randint(0, 50, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=bin_width)
        result = aa(_recording(it, tt, None, None, None, None), env)

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
            n_spikes_current = max(0, int(n_spikes_current * 0.7))
            t_current += bin_width

        tt = np.array(tt_list)
        it = np.random.randint(0, 30, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=bin_width)
        result = aa(_recording(it, tt, None, None, None, None), env)

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
            n_spikes_current = int(n_spikes_current * 1.5) + 1
            t_current += bin_width

        tt = np.array(tt_list)
        it = np.random.randint(0, 30, len(tt))

        aa = AvalancheAnalysis(duration=duration, bin_width=bin_width)
        result = aa(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert result["branching_ratio"] > 1.0

    def test_single_spike_avalanche(self):
        env = MockEnv(n_units=10)
        duration = 1000

        tt = np.array([100.0, 200.0, 300.0, 400.0])
        it = np.array([0, 1, 2, 3])

        aa = AvalancheAnalysis(duration=duration, bin_width=5.0)
        result = aa(_recording(it, tt, None, None, None, None), env)

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
        result = aa(_recording(it, tt, None, None, None, None), env)

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
        result = aa(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert result["n_avalanches"] > 0

    def test_bin_width_sensitivity(self):
        env = MockEnv(n_units=10)
        duration = 1000

        tt = np.array([10, 15, 20, 50, 55, 60, 100, 105])
        it = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        aa_small = AvalancheAnalysis(duration=duration, bin_width=2.0)
        result_small = aa_small(_recording(it, tt, None, None, None, None), env)

        aa_large = AvalancheAnalysis(duration=duration, bin_width=50.0)
        result_large = aa_large(_recording(it, tt, None, None, None, None), env)

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
        result = aa(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert -1.0 <= result["size_power_law_r2"] <= 1.0


class TestChannelRecording:
    def test_setup_respects_selective_flags(self):
        env = MockEnv(n_units=8, n_channels=2)
        decoder = ChannelRecording(
            duration=100,
            spikes=True,
            voltages=False,
            membrane_currents=True,
        )

        decoder.setup(env)

        assert env.calls["spikes"] == 1
        assert env.calls["voltages"] == 0
        assert env.calls["membrane_currents"] == 1

    def test_call_with_spikes_and_membrane_currents(self):
        env = MockEnv(n_units=8, n_channels=2)
        decoder = ChannelRecording(
            duration=100,
            spikes=True,
            voltages=False,
            membrane_currents=True,
        )

        it = np.array([0, 1, 2, 3], dtype=np.int32)
        tt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        mp = np.arange(20, dtype=np.float32).reshape(2, 10)

        cit, ct, iv, vv, channel_ids, p = decoder(
            _recording(it, tt, None, None, None, mp), env
        )

        assert iv is None
        assert vv is None
        assert np.array_equal(channel_ids, np.array([0, 1], dtype=np.int32))
        assert set(cit.keys()) == {0, 1}
        assert set(ct.keys()) == {0, 1}
        assert p is not None
        assert p.shape == (2, 10)

    def test_call_without_spikes_or_membrane_currents(self):
        env = MockEnv(n_units=8, n_channels=2)
        decoder = ChannelRecording(
            duration=100,
            spikes=False,
            voltages=True,
            membrane_currents=False,
        )

        iv = np.array([0, 1], dtype=np.int32)
        vv = np.ones((2, 5), dtype=np.float32)

        cit, ct, iv_out, vv_out, channel_ids, p = decoder(_recording(iv=iv, vv=vv), env)

        assert cit is None
        assert ct is None
        assert np.array_equal(iv_out, iv)
        assert np.array_equal(vv_out, vv)
        assert np.array_equal(channel_ids, np.array([0, 1], dtype=np.int32))
        assert p is None


def _bursting(
    n_units,
    n_bursts,
    spikes_per_unit,
    spread,
    *,
    period=1000.0,
    start=500.0,
    jitter=3.0,
    background=200,
    duration=5000.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    it, tt = [], []
    step = spread / max(spikes_per_unit - 1, 1)
    for k in range(n_bursts):
        t0 = start + period * k
        for unit in range(n_units):
            for s in range(spikes_per_unit):
                it.append(unit)
                tt.append(t0 + s * step + rng.normal(0.0, jitter))
    for _ in range(background):
        it.append(int(rng.integers(0, n_units)))
        tt.append(float(rng.uniform(0.0, duration)))
    order = np.argsort(tt, kind="stable")
    return np.asarray(it)[order], np.asarray(tt)[order]


def _burst_anatomy(it, tt, n_units, duration, **kwargs):
    decoder = BurstAnatomy(duration=duration, **kwargs)
    return decoder(_recording(it, tt), MockEnv(n_units=n_units))


class TestBurstAnatomy:
    def test_reverberation_and_volley_differ(self):
        wide = _burst_anatomy(*_bursting(40, 5, 6, 100.0), 40, 5000.0)
        volley = _burst_anatomy(
            *_bursting(40, 5, 1, 0.0, jitter=4.0, seed=1), 40, 5000.0
        )

        assert wide["n_bursts"] == volley["n_bursts"] == 5
        assert 80.0 <= wide["burst_width"] <= 130.0
        assert volley["burst_width"] <= 30.0
        assert wide["spikes_per_unit_per_burst"] > 4.0
        assert volley["spikes_per_unit_per_burst"] < 2.0

    def test_recruitment_is_a_fraction_of_the_units(self):
        it, tt = _bursting(20, 4, 3, 60.0, seed=2)
        result = _burst_anatomy(it, tt, 40, 5000.0)

        # 20 of 40 units ever fire, so no burst can recruit more than half
        assert 0.4 <= result["units_recruited_per_burst"] <= 0.55

    def test_interval_cv_separates_clockwork_from_irregular(self):
        clockwork = _burst_anatomy(*_bursting(40, 6, 4, 80.0, seed=3), 40, 6500.0)

        rng = np.random.default_rng(4)
        it, tt = [], []
        at = 400.0
        onsets = []
        while at < 11_000.0:
            onsets.append(at)
            for unit in range(40):
                for s in range(4):
                    it.append(unit)
                    tt.append(at + s * 25.0 + rng.normal(0.0, 3.0))
            at += float(rng.exponential(1200.0)) + 300.0
        irregular = _burst_anatomy(np.asarray(it), np.asarray(tt), 40, 11_000.0)

        assert clockwork["burst_interval_cv"] < 0.05
        assert irregular["burst_interval_cv"] > 0.3
        assert len(onsets) >= 4

    def test_a_dip_inside_a_burst_does_not_split_it(self):
        rng = np.random.default_rng(5)
        it, tt = [], []
        for k in range(3):
            t0 = 500.0 + 1500.0 * k
            for offset in (0.0, 150.0):  # two volleys 150 ms apart
                for unit in range(40):
                    it.append(unit)
                    tt.append(t0 + offset + rng.normal(0.0, 2.0))
        it, tt = np.asarray(it), np.asarray(tt)

        merged = _burst_anatomy(it, tt, 40, 5000.0)
        split = _burst_anatomy(it, tt, 40, 5000.0, merge=50.0)

        assert merged["n_bursts"] == 3
        assert split["n_bursts"] == 6
        assert merged["spikes_per_unit_per_burst"] > 1.5
        assert split["spikes_per_unit_per_burst"] < 1.5

    def test_a_window_with_no_burst_is_unmeasured_not_zero(self):
        rng = np.random.default_rng(6)
        it = rng.integers(0, 40, 200)
        tt = rng.uniform(0.0, 5000.0, 200)

        result = _burst_anatomy(it, tt, 40, 5000.0)

        assert result["n_bursts"] == 0
        for name in (
            "burst_width",
            "spikes_per_unit_per_burst",
            "burst_onset_peak",
            "units_recruited_per_burst",
            "burst_interval_cv",
        ):
            assert np.isnan(result[name]), name
        # the rate peak is defined whether or not anything crossed threshold
        assert result["peak_bin_rate"] > 0.0

    def test_empty_window(self):
        result = _burst_anatomy(np.array([]), np.array([]), 40, 5000.0)

        assert result["n_bursts"] == 0
        assert result["burst_spike_fraction"] == 0.0
        assert np.isnan(result["burst_width"])

    def test_decoder_takes_its_unit_count_from_the_readout(self):
        it, tt = _bursting(40, 5, 5, 90.0, seed=7)
        env = MockEnv(n_units=40)

        result = BurstAnatomy(duration=5000)(_recording(it, tt), env)

        assert result["n_bursts"] == 5
        assert 0.0 < result["units_recruited_per_burst"] <= 1.0
        # a readout over half as many units doubles the per-unit peak
        half = BurstAnatomy(duration=5000)(_recording(it, tt), MockEnv(n_units=20))
        assert half["burst_onset_peak"] == pytest.approx(
            2.0 * result["burst_onset_peak"]
        )

    def test_windows_of_consecutive_bursts_are_disjoint(self):
        """Every spike is counted for at most one burst, at any burst rate."""
        it, tt = _bursting(40, 9, 3, 60.0, period=420.0, duration=5000.0, seed=8)

        result = _burst_anatomy(it, tt, 40, 5000.0, merge=100.0)

        assert result["n_bursts"] >= 2
        assert 0.0 < result["burst_spike_fraction"] <= 1.0


def test_baks_takes_ms_and_returns_hz():
    from livn.decoding import baks

    spikes = np.arange(0.0, 2000.0, 100.0)  # a 10 Hz train, in ms
    rate, h = baks(spikes, np.array([1000.0]))
    assert 5.0 < rate[0] < 15.0  # Hz
    assert 10.0 < h[0] < 1000.0  # bandwidth in ms
