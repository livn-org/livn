"""
Generate real experiment datasets using Brian2 backend.

Writes to the default livn_experiments directory so the file server
and Pyodide UI can load them immediately.

Usage:
    LIVN_BACKEND=brian2 python scripts/generate_datasets.py
"""
import os
import sys
import shutil

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from livn.env import Env
from livn.system import predefined
from livn.decoding import ArrowDataset, _default_experiment_root

DURATION = 1000   # ms per shard
NOISE    = {"sigma": 0.5, "rate": 1000}

EXPERIMENTS = [
    # (name, system, n_shards, record_voltage)
    ("EI2_spikes",   "EI2", 5,  False),
    ("EI2_full",     "EI2", 5,  True),
    ("EI1_spikes",   "EI1", 3,  False),
]


def run_one(exp_name: str, sys_name: str, n_shards: int, record_voltage: bool):
    exp_root = _default_experiment_root()
    exp_dir  = os.path.join(exp_root, exp_name)

    # Wipe previous data so shard counts are deterministic
    if os.path.isdir(exp_dir):
        for f in os.listdir(exp_dir):
            if f.startswith("data-") and f.endswith(".arrow"):
                os.remove(os.path.join(exp_dir, f))
        ds_path = os.path.join(exp_dir, "dataset")
        if os.path.isdir(ds_path):
            shutil.rmtree(ds_path)
        meta = os.path.join(exp_dir, "metadata.json")
        if os.path.isfile(meta):
            os.remove(meta)

    decoder = ArrowDataset(
        name=exp_name,
        save_dataset=True,
        save_metadata=True,
        duration=DURATION,
        spikes=True,
        voltages=record_voltage,
        membrane_currents=True,
    )

    for shard_idx in range(n_shards):
        # Fresh env per shard so each row is an independent trial
        env = Env(predefined(sys_name), seed=shard_idx)
        env.init()
        env.set_noise(NOISE)
        env.record_spikes()
        if record_voltage:
            env.record_voltage()
        env.record_membrane_current()

        it, tt, iv, vv, im, mp = env.run(DURATION)

        n_spk = len(it) if it is not None else 0
        print(f"  shard {shard_idx+1}/{n_shards}: {n_spk} spikes")

        decoder(env, it, tt, iv, vv, im, mp)

    ds = decoder.dataset()
    print(f"  => {ds.num_rows} rows, features: {list(ds.features)}")
    print(f"  path: {decoder.directory}")


if __name__ == "__main__":
    import brian2
    brian2.set_device("runtime")   # avoid C++ compile for quick generation

    for exp_name, sys_name, n_shards, record_voltage in EXPERIMENTS:
        print(f"\n[{exp_name}] system={sys_name}, {n_shards} shards, voltage={record_voltage}")
        run_one(exp_name, sys_name, n_shards, record_voltage)

    print("\nDone. Experiments written to:", _default_experiment_root())
