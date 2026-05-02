"""Generate synthetic experiments and verify datasets load from disk.

Run with:
    PYTHONPATH="" pytest tests/test_dataset_server.py -v

Experiments are written to the default livn_experiments directory
(same place the real ArrowDataset decoder uses) so the running server
can serve them immediately without any extra config.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from livn.decoding import ArrowDataset, _default_experiment_root


EXP_NAMES = {"exp_small": 2, "exp_large": 5}  # name -> n_shards


def _make_experiment(name: str, n_shards: int) -> ArrowDataset:
    """Write *n_shards* synthetic rows using the default experiment root."""
    rng = np.random.default_rng(seed=42)
    decoder = ArrowDataset(name=name, save_dataset=True, duration=250)

    for _ in range(n_shards):
        n_spikes = int(rng.integers(5, 20))
        row = {
            "duration": 250,
            "it": rng.integers(0, 100, size=n_spikes).astype(np.int32),
            "tt": rng.uniform(0, 250, size=n_spikes).astype(np.float32),
        }
        decoder._write_shard(row)

    ds = decoder.dataset()
    assert ds is not None
    ds.save_to_disk(decoder.dataset_path)
    return decoder


@pytest.fixture(scope="module", autouse=True)
def create_experiments():
    """Create (or overwrite) the synthetic experiments once per test session."""
    for name, n_shards in EXP_NAMES.items():
        _make_experiment(name, n_shards)


@pytest.fixture(scope="module")
def exp_root():
    return _default_experiment_root()


@pytest.fixture(scope="module")
def exp_paths(exp_root):
    registry_path = os.path.join(exp_root, "experiments.json")
    with open(registry_path) as f:
        registry = json.load(f)
    return {name: info["path"] for name in EXP_NAMES for name, info in [(name, registry[name])]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_experiments_registered(exp_root):
    registry_path = os.path.join(exp_root, "experiments.json")
    with open(registry_path) as f:
        registry = json.load(f)
    for name in EXP_NAMES:
        assert name in registry, f"{name} not in registry"


def test_dataset_dirs_exist(exp_paths):
    for name, path in exp_paths.items():
        assert os.path.isdir(os.path.join(path, "dataset")), f"dataset/ missing for {name}"


def test_arrow_shard_counts(exp_paths):
    for name, path in exp_paths.items():
        shards = [f for f in os.listdir(path) if f.startswith("data-") and f.endswith(".arrow")]
        assert len(shards) == EXP_NAMES[name], f"{name}: expected {EXP_NAMES[name]} shards, got {len(shards)}"


def test_load_from_disk_row_counts(exp_paths):
    from datasets import load_from_disk

    for name, path in exp_paths.items():
        ds = load_from_disk(os.path.join(path, "dataset"))
        assert ds.num_rows == EXP_NAMES[name], f"{name}: expected {EXP_NAMES[name]} rows, got {ds.num_rows}"


def test_load_from_disk_columns(exp_paths):
    from datasets import load_from_disk

    ds = load_from_disk(os.path.join(exp_paths["exp_small"], "dataset"))
    assert "it" in ds.features
    assert "tt" in ds.features
    assert "duration" in ds.features


def test_spike_values_valid(exp_paths):
    from datasets import load_from_disk

    ds = load_from_disk(os.path.join(exp_paths["exp_small"], "dataset"))
    for row in ds:
        assert len(row["it"]) == len(row["tt"])
        assert all(0 <= n < 100 for n in row["it"])
        assert all(0.0 <= t <= 250.0 for t in row["tt"])
        assert row["duration"] == 250
