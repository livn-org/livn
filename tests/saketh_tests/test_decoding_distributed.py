"""
Tests for GatherAndMerge and ArrowDataset decoding.

Validates:
- decoding.__call__ runs ONCE per simulation call (not per timestep)
- GatherAndMerge correctly gathers and merges distributed per-rank arrays
- ArrowDataset writes exactly one shard per __call__
- Pipe chains Slice → GatherAndMerge in a single call
"""

import os

import numpy as np
import pytest

from livn.decoding import ArrowDataset, GatherAndMerge, Pipe, Slice


# ---------------------------------------------------------------------------
# Minimal mock env — no MPI, no real simulation
# ---------------------------------------------------------------------------


class MockEnv:
    comm = None
    voltage_recording_dt = 0.1
    membrane_current_recording_dt = 0.1

    def __init__(self, n_units=20):
        self.n_units = n_units
        self.setup_calls = {"spikes": 0, "voltages": 0, "membrane_currents": 0}

    def record_spikes(self):
        self.setup_calls["spikes"] += 1

    def record_voltage(self):
        self.setup_calls["voltages"] += 1

    def record_membrane_current(self):
        self.setup_calls["membrane_currents"] += 1


def make_spikes(n_spikes, n_units, duration_ms, seed=0):
    rng = np.random.default_rng(seed)
    it = rng.integers(0, n_units, n_spikes).astype(np.int32)
    tt = np.sort(rng.uniform(0, duration_ms, n_spikes)).astype(np.float64)
    return it, tt


def make_voltage(n_units, duration_ms, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = int(duration_ms / dt)
    iv = np.arange(n_units, dtype=np.int32)
    vv = rng.standard_normal((n_units, n_samples)).astype(np.float32)
    return iv, vv


def make_membrane(n_units, duration_ms, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = int(duration_ms / dt)
    im = np.arange(n_units, dtype=np.int32)
    mp = rng.standard_normal((n_units, n_samples)).astype(np.float32)
    return im, mp


# ---------------------------------------------------------------------------
# GatherAndMerge: single-call semantics
# ---------------------------------------------------------------------------


class TestGatherAndMerge:
    def test_returns_arrays_not_none(self):
        """On a single rank (no MPI) the root path always runs."""
        env = MockEnv()
        duration = 500
        it, tt = make_spikes(30, 20, duration)

        gm = GatherAndMerge(
            duration=duration, voltages=False, membrane_currents=False
        )
        result = gm(env, it, tt, None, None, None, None)

        assert result is not None
        r_it, r_tt, r_iv, r_vv, r_im, r_mp = result
        assert r_it is not None
        assert r_tt is not None

    def test_spike_arrays_preserved(self):
        """gather+merge on a single rank should return the original spike data."""
        env = MockEnv()
        duration = 500
        it, tt = make_spikes(30, 20, duration)

        gm = GatherAndMerge(
            duration=duration, voltages=False, membrane_currents=False
        )
        r_it, r_tt, *_ = gm(env, it, tt, None, None, None, None)

        np.testing.assert_array_equal(np.sort(r_it), np.sort(it))
        np.testing.assert_array_equal(np.sort(r_tt), np.sort(tt))

    def test_voltage_arrays_preserved(self):
        env = MockEnv()
        duration = 200
        iv, vv = make_voltage(5, duration)

        gm = GatherAndMerge(
            duration=duration, spikes=False, membrane_currents=False
        )
        _, _, r_iv, r_vv, *_ = gm(env, None, None, iv, vv, None, None)

        np.testing.assert_array_equal(r_iv, iv)
        np.testing.assert_array_equal(r_vv, vv)

    def test_disabled_modalities_return_none(self):
        """Data for a disabled modality should be None in the output."""
        env = MockEnv()
        duration = 200
        it, tt = make_spikes(10, 5, duration)
        iv, vv = make_voltage(5, duration)
        im, mp = make_membrane(5, duration)

        gm = GatherAndMerge(
            duration=duration, spikes=True, voltages=False, membrane_currents=False
        )
        r_it, r_tt, r_iv, r_vv, r_im, r_mp = gm(
            env, it, tt, iv, vv, im, mp
        )

        assert r_it is not None
        assert r_tt is not None
        assert r_iv is None
        assert r_vv is None
        assert r_im is None
        assert r_mp is None

    def test_setup_arms_requested_recorders(self):
        """setup() should call record_* for each enabled modality."""
        env = MockEnv()
        gm = GatherAndMerge(
            duration=100, spikes=True, voltages=True, membrane_currents=False
        )
        gm.setup(env)

        assert env.setup_calls["spikes"] == 1
        assert env.setup_calls["voltages"] == 1
        assert env.setup_calls["membrane_currents"] == 0

    def test_called_once_not_per_timestep(self):
        """
        Decoding is invoked on the already-complete simulation output.
        We verify that a single __call__ returns the full duration's data,
        not a per-timestep slice.
        """
        env = MockEnv()
        duration = 1000  # 1000 ms
        n_spikes = 200
        it, tt = make_spikes(n_spikes, 20, duration)

        gm = GatherAndMerge(
            duration=duration, voltages=False, membrane_currents=False
        )

        # One call — all spikes should be present in the output.
        r_it, r_tt, *_ = gm(env, it, tt, None, None, None, None)

        assert len(r_tt) == n_spikes
        assert r_tt.min() >= 0.0
        assert r_tt.max() <= float(duration)

    def test_simulated_two_rank_merge(self):
        """
        Simulate what MPI gather produces by manually constructing the
        list-of-per-rank-arrays that P.gather would return, then calling
        P.merge directly. Validates the merge step in isolation.
        """
        from livn.utils import P

        rng = np.random.default_rng(7)
        rank0_it = rng.integers(0, 50, 15).astype(np.int32)
        rank1_it = rng.integers(0, 50, 12).astype(np.int32)
        rank0_tt = np.sort(rng.uniform(0, 500, 15)).astype(np.float64)
        rank1_tt = np.sort(rng.uniform(0, 500, 12)).astype(np.float64)

        # P.merge concatenates the list of per-rank arrays
        merged_it, merged_tt = P.merge(
            [rank0_it, rank1_it], [rank0_tt, rank1_tt]
        )

        assert len(merged_it) == 15 + 12
        assert len(merged_tt) == 15 + 12
        np.testing.assert_array_equal(
            merged_it, np.concatenate([rank0_it, rank1_it])
        )


# ---------------------------------------------------------------------------
# ArrowDataset: one shard per call
# ---------------------------------------------------------------------------


class TestArrowDatasetCallSemantics:
    @pytest.fixture(autouse=True)
    def _require_deps(self):
        pytest.importorskip("datasets")
        pytest.importorskip("pyarrow")

    def test_one_shard_per_call(self, tmp_path):
        """Each __call__ writes exactly one new shard file."""
        env = MockEnv()
        duration = 300

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="shards",
            voltages=False,
            membrane_currents=False,
        )

        for i in range(4):
            it, tt = make_spikes(10, 10, duration, seed=i)
            ad(env, it, tt, None, None, None, None)
            shard_count = len(
                [f for f in os.listdir(ad.directory) if f.endswith(".arrow")]
            )
            assert shard_count == i + 1, f"Expected {i+1} shards after call {i}"

    def test_dataset_row_count_matches_calls(self, tmp_path):
        """dataset() should contain one row per simulation call."""
        env = MockEnv()
        duration = 200
        n_calls = 5

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="rows",
            voltages=False,
            membrane_currents=False,
        )
        for i in range(n_calls):
            it, tt = make_spikes(8, 10, duration, seed=i)
            ad(env, it, tt, None, None, None, None)

        ds = ad.dataset()
        assert len(ds) == n_calls

    def test_shard_contains_full_spike_data(self, tmp_path):
        """Each row should hold all spikes from that single simulation call."""
        env = MockEnv()
        duration = 500

        it, tt = make_spikes(40, 20, duration, seed=3)

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="full",
            voltages=False,
            membrane_currents=False,
        )
        ad(env, it, tt, None, None, None, None)

        ds = ad.dataset()
        row = ds[0]
        np.testing.assert_array_equal(np.sort(row["it"]), np.sort(it))
        np.testing.assert_allclose(np.sort(row["tt"]), np.sort(tt))

    def test_duration_stored_per_row(self, tmp_path):
        """The 'duration' column should match the decoding duration."""
        env = MockEnv()
        duration = 750

        ad = ArrowDataset(
            duration=duration,
            experiment_root=str(tmp_path),
            name="dur",
            voltages=False,
            membrane_currents=False,
        )
        it, tt = make_spikes(5, 5, duration)
        ad(env, it, tt, None, None, None, None)

        ds = ad.dataset()
        assert ds[0]["duration"] == duration


# ---------------------------------------------------------------------------
# Pipe: Slice → GatherAndMerge runs in a single __call__
# ---------------------------------------------------------------------------


class TestPipeSliceThenGather:
    def test_slice_filters_before_gather(self):
        """
        Slice trims the spike window; GatherAndMerge then collects.
        The full pipeline runs in one __call__ on the complete output,
        not once per timestep.
        """
        env = MockEnv()
        duration = 1000
        it, tt = make_spikes(100, 20, duration)

        # Keep only [200, 600) ms
        pipe = Pipe(
            duration=duration,
            stages=[
                Slice(stop=600, start=200),
                GatherAndMerge(
                    duration=400, voltages=False, membrane_currents=False
                ),
            ],
        )
        pipe.setup(env)
        result = pipe(env, it, tt, None, None, None, None)

        r_it, r_tt, *_ = result
        assert r_tt is not None
        assert r_tt.min() >= 0.0        # times are re-zeroed by Slice
        assert r_tt.max() < 400.0       # duration of the slice

        # Only spikes originally in [200, 600) should survive
        expected_mask = (tt >= 200) & (tt < 600)
        assert len(r_tt) == int(expected_mask.sum())

    def test_pipe_single_call_covers_full_duration(self):
        """
        One pipe.__call__ processes spikes spanning the full duration.
        There is no per-timestep invocation.
        """
        env = MockEnv()
        duration = 2000
        it, tt = make_spikes(500, 20, duration)

        pipe = Pipe(
            duration=duration,
            stages=[
                GatherAndMerge(
                    duration=duration, voltages=False, membrane_currents=False
                )
            ],
        )
        pipe.setup(env)
        result = pipe(env, it, tt, None, None, None, None)

        r_it, r_tt, *_ = result
        # All spikes should be present — no truncation from per-step calls
        assert len(r_tt) == 500
