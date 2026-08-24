import os
from typing import ClassVar

import numpy as np
import pytest

from livn.decoding import (
    LFP,
    ActiveFraction,
    ArrowDataset,
    AvalancheAnalysis,
    ChannelRecording,
    MeanFiringRate,
    Pipe,
    PopulationActiveFraction,
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

    def record_voltage(self):
        self.calls["voltages"] += 1

    def record_membrane_current(self):
        self.calls["membrane_currents"] += 1

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

    def potential_recording(self, m, gids=None):
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
        duration = 1000

        it = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        tt = np.linspace(0, 999, 20)

        mfr = MeanFiringRate(duration=duration)
        result = mfr(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert "rate_hz" in result
        assert result["total_spikes"] == 20
        assert result["n_units"] == 10
        assert abs(result["rate_hz"] - 2.0) < 0.01

    def test_empty_spikes(self):
        env = MockEnv(n_units=10)
        it = np.array([])
        tt = np.array([])

        mfr = MeanFiringRate(duration=1000)
        result = mfr(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert result["total_spikes"] == 0
        assert result["rate_hz"] < 1e-6


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

        stability = Stability(duration=duration, max_rate_hz=20, min_rate_hz=0.01)
        result = stability(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert "is_stable" in result
        assert "tail_mean_hz" in result
        assert "global_mean_hz" in result

    def test_runaway_detection(self):
        env = MockEnv(n_units=10)
        duration = 2000

        it = np.repeat(np.arange(10), 50)
        tt = np.random.uniform(1500, 2000, 500)

        stability = Stability(duration=duration, tail_window=500, max_rate_hz=10)
        result = stability(_recording(it, tt, None, None, None, None), env)

        assert result["is_runaway"] is True
        assert result["is_stable"] is False

    def test_quiescence_detection(self):
        env = MockEnv(n_units=10)
        duration = 5000

        it = np.array([], dtype=np.int32)
        tt = np.array([], dtype=np.float64)

        stability = Stability(duration=duration, min_rate_hz=0.05)
        result = stability(_recording(it, tt, None, None, None, None), env)

        assert result["global_mean_hz"] == 0.0
        assert result["is_quiescent"] is True
        assert result["is_stable"] is False


class TestLFP:
    def test_basic_extraction(self):
        env = MockEnv(n_units=50, n_channels=8)
        duration = 1000

        im, m = make_mock_membrane_current(50, duration)

        lfp = LFP(duration=duration, downsample_hz=1000)
        result = lfp(_recording(None, None, None, None, im, m), env)

        assert result is not None
        assert "lfp" in result
        assert result["n_channels"] == 8
        assert result["sample_rate_hz"] > 0

    def test_downsampling(self):
        env = MockEnv(n_units=50, n_channels=4)
        duration = 1000
        env.membrane_current_recording_dt = 0.1

        im, m = make_mock_membrane_current(50, duration, dt=0.1)

        lfp = LFP(duration=duration, downsample_hz=1000)
        result = lfp(_recording(None, None, None, None, im, m), env)

        assert result["sample_rate_hz"] == 1000.0

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
        duration_ms = 4000

        im, m = make_mock_membrane_current(50, duration_ms, dt=1.0)

        lfp_decoder = LFP(
            duration=duration_ms,
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


class TestArrowDataset:
    @pytest.fixture(autouse=True)
    def _require_datasets(self):
        pytest.importorskip("datasets")
        pytest.importorskip("pyarrow")

    def test_single_write_and_load(self, tmp_path):
        env = MockEnv(n_units=10)
        duration = 1000
        it, tt = make_mock_spikes(20, 10, duration)
        directory = str(tmp_path / "arrow_out")

        ad = ArrowDataset(
            duration=duration,
            directory=directory,
            voltages=False,
            membrane_currents=False,
        )
        result = ad(_recording(it, tt, None, None, None, None), env)

        assert result is not None
        assert os.path.isfile(os.path.join(directory, "data-00000.arrow"))

        ds = ad.dataset()
        assert ds is not None
        assert len(ds) == 1
        assert "duration" in ds.column_names
        assert "it" in ds.column_names
        assert "tt" in ds.column_names

    def test_multiple_writes_create_shards(self, tmp_path):
        env = MockEnv(n_units=10)
        duration = 500
        directory = str(tmp_path / "multi")

        ad = ArrowDataset(
            duration=duration,
            directory=directory,
            voltages=False,
            membrane_currents=False,
        )

        for seed in range(3):
            it, tt = make_mock_spikes(15, 10, duration, seed=seed)
            ad(_recording(it, tt, None, None, None, None), env)

        for i in range(3):
            assert os.path.isfile(os.path.join(directory, f"data-{i:05d}.arrow"))

        ds = ad.dataset()
        assert len(ds) == 3

    def test_resume_after_recreate(self, tmp_path):
        env = MockEnv(n_units=10)
        duration = 500
        directory = str(tmp_path / "resume")

        ad1 = ArrowDataset(
            duration=duration,
            directory=directory,
            voltages=False,
            membrane_currents=False,
        )
        it, tt = make_mock_spikes(10, 10, duration, seed=0)
        ad1(_recording(it, tt, None, None, None, None), env)

        ad2 = ArrowDataset(
            duration=duration,
            directory=directory,
            voltages=False,
            membrane_currents=False,
        )
        it, tt = make_mock_spikes(10, 10, duration, seed=1)
        ad2(_recording(it, tt, None, None, None, None), env)

        assert os.path.isfile(os.path.join(directory, "data-00000.arrow"))
        assert os.path.isfile(os.path.join(directory, "data-00001.arrow"))

        ds = ad2.dataset()
        assert len(ds) == 2

    def test_all_data_types(self, tmp_path):
        env = MockEnv(n_units=5)
        duration = 200
        directory = str(tmp_path / "all_types")

        it, tt = make_mock_spikes(10, 5, duration)
        iv = np.arange(5, dtype=np.int32)
        vv = np.random.randn(5, 100).astype(np.float32)
        im, mp = make_mock_membrane_current(5, duration)

        ad = ArrowDataset(
            duration=duration,
            directory=directory,
            spikes=True,
            voltages=True,
            membrane_currents=True,
        )
        ad(_recording(it, tt, iv, vv, im, mp), env)

        ds = ad.dataset()
        assert len(ds) == 1
        assert "it" in ds.column_names
        assert "tt" in ds.column_names
        assert "iv" in ds.column_names
        assert "vv" in ds.column_names
        assert "im" in ds.column_names
        assert "mp" in ds.column_names
        assert ds[0]["duration"] == duration

    def test_selective_recording(self, tmp_path):
        env = MockEnv(n_units=5)
        duration = 200
        directory = str(tmp_path / "selective")

        it, tt = make_mock_spikes(10, 5, duration)
        im, mp = make_mock_membrane_current(5, duration)

        ad = ArrowDataset(
            duration=duration,
            directory=directory,
            spikes=True,
            voltages=False,
            membrane_currents=False,
        )
        ad(_recording(it, tt, None, None, im, mp), env)

        ds = ad.dataset()
        assert "it" in ds.column_names
        assert "iv" not in ds.column_names
        assert "im" not in ds.column_names

    def test_empty_directory_dataset(self, tmp_path):
        directory = str(tmp_path / "empty")
        os.makedirs(directory)

        ad = ArrowDataset(
            duration=100,
            directory=directory,
            voltages=False,
            membrane_currents=False,
        )
        ds = ad.dataset()
        assert ds is None


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
