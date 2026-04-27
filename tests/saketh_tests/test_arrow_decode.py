import os

import numpy as np
import pytest

from livn.env import Env
from livn.decoding import (
    Slice,
    ChannelRecording,
    Pipe,
    MeanFiringRate,
    ActiveFraction,
    Stability,
    LFP,
    AvalancheAnalysis,
    ArrowDataset,
)
from livn.backend import backend

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
        result = ad(env, it, tt, None, None, None, None)

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
            ad(env, it, tt, None, None, None, None)

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
        ad1(env, it, tt, None, None, None, None)

        # recreate — simulates object destruction/recreation
        ad2 = ArrowDataset(
            duration=duration,
            directory=directory,
            voltages=False,
            membrane_currents=False,
        )
        it, tt = make_mock_spikes(10, 10, duration, seed=1)
        ad2(env, it, tt, None, None, None, None)

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
        ad(env, it, tt, iv, vv, im, mp)

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
        ad(env, it, tt, None, None, im, mp)

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