import json
import os
import shutil

import numpy as np
import pytest
from datasets import load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns
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
import livn.decoding as _decoding_mod
from livn.backend import backend

# Absolute path of the livn_ui project root (two levels up from this file)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_EXPERIMENT_ROOT = os.path.join(_PROJECT_ROOT, "livn_experiments")
_EXPERIMENTS2_ROOT = os.path.join(_PROJECT_ROOT, "experiments2")

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

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="arrow_out",
            voltages=False,
            membrane_currents=False,
        )
        result = ad(env, it, tt, None, None, None, None)

        assert result is not None
        assert os.path.isfile(os.path.join(ad.directory, "data-00000.arrow"))

        ds = ad.dataset()
        assert ds is not None
        assert len(ds) == 1
        assert "duration" in ds.column_names
        assert "it" in ds.column_names
        assert "tt" in ds.column_names

    def test_multiple_writes_create_shards(self, tmp_path):
        env = MockEnv(n_units=10)
        duration = 500

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="multi",
            voltages=False,
            membrane_currents=False,
        )

        for seed in range(3):
            it, tt = make_mock_spikes(15, 10, duration, seed=seed)
            ad(env, it, tt, None, None, None, None)

        for i in range(3):
            assert os.path.isfile(os.path.join(ad.directory, f"data-{i:05d}.arrow"))

        ds = ad.dataset()
        assert len(ds) == 3

    def test_resume_after_recreate(self, tmp_path):
        env = MockEnv(n_units=10)
        duration = 500

        ad1 = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="resume",
            voltages=False,
            membrane_currents=False,
        )
        it, tt = make_mock_spikes(10, 10, duration, seed=0)
        ad1(env, it, tt, None, None, None, None)

        # recreate — simulates object destruction/recreation
        ad2 = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="resume",
            voltages=False,
            membrane_currents=False,
        )
        it, tt = make_mock_spikes(10, 10, duration, seed=1)
        ad2(env, it, tt, None, None, None, None)

        assert os.path.isfile(os.path.join(ad2.directory, "data-00000.arrow"))
        assert os.path.isfile(os.path.join(ad2.directory, "data-00001.arrow"))

        ds = ad2.dataset()
        assert len(ds) == 2

    def test_all_data_types(self, tmp_path):
        env = MockEnv(n_units=5)
        duration = 200

        it, tt = make_mock_spikes(10, 5, duration)
        iv = np.arange(5, dtype=np.int32)
        vv = np.random.randn(5, 100).astype(np.float32)
        im, mp = make_mock_membrane_current(5, duration)

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="all_types",
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

        it, tt = make_mock_spikes(10, 5, duration)
        im, mp = make_mock_membrane_current(5, duration)

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="selective",
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
        ad = ArrowDataset(
            duration=100,
            experiment_root=str(tmp_path),
            name="empty",
            voltages=False,
            membrane_currents=False,
        )
        ds = ad.dataset()
        assert ds is None

    def test_save_dataset_flag(self, tmp_path):
        env = MockEnv(n_units=10)
        duration = 300

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="saved",
            save_dataset=True,
            voltages=False,
            membrane_currents=False,
        )

        for seed in range(3):
            it, tt = make_mock_spikes(10, 10, duration, seed=seed)
            ad(env, it, tt, None, None, None, None)

        assert os.path.isdir(ad.dataset_path)
        reloaded = load_from_disk(ad.dataset_path)
        assert len(reloaded) == 3

    def test_persistent_run_accumulates_into_one_dataset(self, tmp_path):
        """
        Multiple calls on the same ArrowDataset (persistent simulation — state
        not cleared between calls) should produce one shard per call, and
        dataset() should unify them into a single dataset whose spike times
        span each call's window without overlap.
        """
        env = MockEnv(n_units=10)
        segment_duration = 500
        n_segments = 3

        ad = ArrowDataset(
            duration=segment_duration,
            experiment_root=str(tmp_path),
            name="persistent",
            voltages=False,
            membrane_currents=False,
        )

        # Simulate persistent run: each call covers a non-overlapping time window.
        # In a real env the simulator advances time; here we encode that in spike times.
        all_spike_counts = []
        for seg in range(n_segments):
            np.random.seed(seg)
            n_spikes = np.random.randint(5, 20)
            it = np.random.randint(0, 10, n_spikes).astype(np.int32)
            # spikes are relative to segment start (as env.run returns them)
            tt = np.sort(np.random.uniform(0, segment_duration, n_spikes)).astype(np.float64)
            all_spike_counts.append(n_spikes)
            ad(env, it, tt, None, None, None, None)

        ds = ad.dataset()

        # One row per segment
        assert len(ds) == n_segments

        # Each row preserves its spike count
        for seg in range(n_segments):
            assert len(ds[seg]["tt"]) == all_spike_counts[seg]
            # spike times are within the segment window
            assert min(ds[seg]["tt"]) >= 0
            assert max(ds[seg]["tt"]) <= segment_duration

    def test_save_metadata_writes_json(self, tmp_path):
        import json
        from pydantic import BaseModel

        class FakeEncoding(BaseModel):
            kind: str = "sinusoid"
            freq_hz: float = 10.0

        class FakeSystem:
            uri = "/data/EI1.h5"
            populations = ["E", "I"]
            gids = list(range(100))

        class FakeModel:
            pass

        class MetadataEnv(MockEnv):
            def __init__(self):
                super().__init__(n_units=5)
                self.system = FakeSystem()
                self.encoding = FakeEncoding()
                self.model = FakeModel()

        env = MetadataEnv()
        duration = 200
        it, tt = make_mock_spikes(8, 5, duration)

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="metadata",
            save_metadata=True,
            voltages=False,
            membrane_currents=False,
        )
        ad(env, it, tt, None, None, None, None)

        meta_path = os.path.join(ad.directory, "metadata.json")
        assert os.path.isfile(meta_path)

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["system"]["uri"] == "/data/EI1.h5"
        assert meta["system"]["populations"] == ["E", "I"]
        assert meta["system"]["n_populations"] == 2
        assert meta["system"]["n_neurons"] == 100
        assert meta["encoding"]["kind"] == "sinusoid"
        assert meta["encoding"]["freq_hz"] == 10.0
        assert meta["model"] == "FakeModel"
        assert meta["duration"] == duration

    def test_experiment_root_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LIVN_EXPERIMENT_ROOT", str(tmp_path))
        env = MockEnv(n_units=5)
        duration = 200

        ad = ArrowDataset(
            duration=duration,
            name="env_root_test",
            voltages=False,
            membrane_currents=False,
        )
        assert ad.experiment_root == str(tmp_path)
        assert ad.directory == str(tmp_path / "env_root_test")

        it, tt = make_mock_spikes(5, 5, duration)
        ad(env, it, tt, None, None, None, None)
        assert os.path.isfile(os.path.join(ad.directory, "data-00000.arrow"))


class TestRegistryAndRoots:
    """Tests that ArrowDataset maintains the per-root experiments.json registry
    and the global ~/.livn/roots.json across multiple experiment roots."""

    @pytest.fixture(autouse=True)
    def _require_datasets(self):
        pytest.importorskip("datasets")
        pytest.importorskip("pyarrow")

    @pytest.fixture()
    def roots_file(self, tmp_path, monkeypatch):
        """Redirect the global roots file to a temp location.
        Experiment root directories (livn_experiments, experiments2) are left on
        disk after the test so they can be inspected."""
        fake_roots = str(tmp_path / "roots.json")
        monkeypatch.setattr(_decoding_mod, "_GLOBAL_ROOTS_FILE", fake_roots)
        return fake_roots

    def test_default_root_creates_livn_experiments(self, roots_file):
        """Writing an experiment to the default root creates livn_ui/livn_experiments."""
        env = MockEnv(n_units=5)
        duration = 200
        it, tt = make_mock_spikes(5, 5, duration)

        ad = ArrowDataset(
            duration=duration,
            experiment_root=_DEFAULT_EXPERIMENT_ROOT,
            name="exp_default",
            voltages=False,
            membrane_currents=False,
        )
        ad(env, it, tt, None, None, None, None)

        assert os.path.isdir(_DEFAULT_EXPERIMENT_ROOT)
        assert os.path.isfile(os.path.join(ad.directory, "data-00000.arrow"))

        registry = json.loads(
            open(os.path.join(_DEFAULT_EXPERIMENT_ROOT, "experiments.json")).read()
        )
        assert "exp_default" in registry
        assert registry["exp_default"]["path"] == ad.directory

        roots = json.loads(open(roots_file).read())
        assert os.path.abspath(_DEFAULT_EXPERIMENT_ROOT) in roots

    def test_custom_root_creates_experiments2(self, roots_file):
        """Writing an experiment to experiments2 creates livn_ui/experiments2."""
        env = MockEnv(n_units=5)
        duration = 200
        it, tt = make_mock_spikes(5, 5, duration)

        ad = ArrowDataset(
            duration=duration,
            experiment_root=_EXPERIMENTS2_ROOT,
            name="exp_custom",
            voltages=False,
            membrane_currents=False,
        )
        ad(env, it, tt, None, None, None, None)

        assert os.path.isdir(_EXPERIMENTS2_ROOT)
        assert os.path.isfile(os.path.join(ad.directory, "data-00000.arrow"))

        registry = json.loads(
            open(os.path.join(_EXPERIMENTS2_ROOT, "experiments.json")).read()
        )
        assert "exp_custom" in registry

        roots = json.loads(open(roots_file).read())
        assert os.path.abspath(_EXPERIMENTS2_ROOT) in roots

    def test_both_roots_in_global_registry(self, roots_file):
        """Both livn_experiments and experiments2 appear in the global roots.json."""
        env = MockEnv(n_units=5)
        duration = 200

        for root, name in [
            (_DEFAULT_EXPERIMENT_ROOT, "from_default"),
            (_EXPERIMENTS2_ROOT, "from_experiments2"),
        ]:
            it, tt = make_mock_spikes(5, 5, duration)
            ad = ArrowDataset(
                duration=duration,
                experiment_root=root,
                name=name,
                voltages=False,
                membrane_currents=False,
            )
            ad(env, it, tt, None, None, None, None)

        roots = json.loads(open(roots_file).read())
        assert os.path.abspath(_DEFAULT_EXPERIMENT_ROOT) in roots
        assert os.path.abspath(_EXPERIMENTS2_ROOT) in roots

    def test_per_root_registry_tracks_multiple_experiments(self, roots_file):
        """Per-root experiments.json accumulates entries across different experiment names."""
        env = MockEnv(n_units=5)
        duration = 200

        names = ["alpha", "beta", "gamma"]
        for name in names:
            it, tt = make_mock_spikes(5, 5, duration)
            ad = ArrowDataset(
                duration=duration,
                experiment_root=_DEFAULT_EXPERIMENT_ROOT,
                name=name,
                voltages=False,
                membrane_currents=False,
            )
            ad(env, it, tt, None, None, None, None)

        registry = json.loads(
            open(os.path.join(_DEFAULT_EXPERIMENT_ROOT, "experiments.json")).read()
        )
        for name in names:
            assert name in registry
            assert "path" in registry[name]
            assert "created_at" in registry[name]

    def test_dataset_and_metadata_saved_to_experiment_folder(self, roots_file):
        """HuggingFace dataset and metadata.json land inside the named experiment folder
        for both livn_experiments (default root) and experiments2 (custom root)."""
        from pydantic import BaseModel

        class FakeEncoding(BaseModel):
            kind: str = "sinusoid"
            freq_hz: float = 10.0

        class FakeSystem:
            uri = "/data/EI1.h5"
            populations = ["E", "I"]
            gids = list(range(50))

        class FakeModel:
            pass

        class RichEnv(MockEnv):
            def __init__(self):
                super().__init__(n_units=5)
                self.system = FakeSystem()
                self.encoding = FakeEncoding()
                self.model = FakeModel()

        env = RichEnv()
        duration = 300

        for root, name in [
            (_DEFAULT_EXPERIMENT_ROOT, "ds_meta_default"),
            (_EXPERIMENTS2_ROOT, "ds_meta_custom"),
        ]:
            it, tt = make_mock_spikes(10, 5, duration)
            ad = ArrowDataset(
                duration=duration,
                experiment_root=root,
                name=name,
                save_dataset=True,
                save_metadata=True,
                voltages=False,
                membrane_currents=False,
            )
            ad(env, it, tt, None, None, None, None)

            # HuggingFace dataset saved inside the experiment folder
            assert os.path.isdir(ad.dataset_path), (
                f"dataset/ dir missing at {ad.dataset_path}"
            )
            reloaded = load_from_disk(ad.dataset_path)
            assert len(reloaded) >= 1
            assert "it" in reloaded.column_names
            assert "tt" in reloaded.column_names
            assert reloaded[0]["duration"] == duration

            # metadata.json saved inside the experiment folder
            meta_path = os.path.join(ad.directory, "metadata.json")
            assert os.path.isfile(meta_path), f"metadata.json missing at {meta_path}"
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["duration"] == duration
            assert meta["system"]["uri"] == "/data/EI1.h5"
            assert meta["system"]["populations"] == ["E", "I"]
            assert meta["encoding"]["kind"] == "sinusoid"
            assert meta["model"] == "FakeModel"