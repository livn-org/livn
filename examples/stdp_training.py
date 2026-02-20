# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
#   "matplotlib",
# ]
# ///

import os

os.environ["LIVN_BACKEND"] = "neuron"

import matplotlib.pyplot as plt
import numpy as np

from livn.env import Env


BASELINE_MS = 200  # no learning warmup
TRAINING_MS = 1000
EVAL_MS = 200  # frozen weights
WEIGHT_REC_DT = 1.0


env = Env("systems/graphs/EI1").init()
env.apply_model_defaults()
env.record_spikes()
env.record_voltage()
env.record_weights(dt=WEIGHT_REC_DT)

print(f"Phase 1: Baseline ({BASELINE_MS} ms, no plasticity)")
it_base, t_base, iv_base, v_base, *_ = env.run(BASELINE_MS)

weights_before = env.get_weights()
print(f" {len(weights_before)} STDP-capable synapses")

print(f"Phase 2: Training ({TRAINING_MS} ms, STDP + synaptic scaling)")
env.enable_plasticity()

it_train_all, t_train_all = [], []
iv_train_all, v_train_all = [], []
remaining = TRAINING_MS
while remaining > 0:
    chunk = min(100, remaining)
    it_chunk, t_chunk, iv_chunk, v_chunk, *_ = env.run(chunk)
    it_train_all.append(np.asarray(it_chunk))
    t_train_all.append(np.asarray(t_chunk))
    env.normalize_weights()
    remaining -= chunk

it_train = np.concatenate(it_train_all) if it_train_all else np.array([])
t_train = np.concatenate(t_train_all) if t_train_all else np.array([])

weights_after_training = env.get_weights()

print(f"Phase 3: Evaluation ({EVAL_MS} ms, plasticity frozen)")
env.disable_plasticity()

it_eval, t_eval, iv_eval, v_eval, *_ = env.run(EVAL_MS)

weights_final = env.get_weights()

w_before = np.array(list(weights_before.values()))
w_after = np.array(list(weights_after_training.values()))
w_final = np.array(list(weights_final.values()))

if env.w_recs:
    max_dev = 0.0
    for key, vec in env.w_recs.items():
        trace = np.array(vec.as_numpy())
        dev = np.max(np.abs(trace - 1.0))
        max_dev = max(max_dev, dev)
    print(f" Peak weight deviation during training: {max_dev:.4f}")

frozen = np.sum(np.abs(w_final - w_after) > 1e-9)
print(f" Synapses drifted during eval (should be 0): {frozen}")


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Weight evolution over training
ax = axes[0, 0]
n_traces = min(20, len(env.w_recs))
for i, (key, vec) in enumerate(env.w_recs.items()):
    if i >= n_traces:
        break
    trace = np.array(vec.as_numpy())
    time_axis = np.arange(len(trace)) * WEIGHT_REC_DT
    ax.plot(time_axis, trace, alpha=0.6, linewidth=0.8)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Weight multiplier (w)")
ax.set_title(f"Weight evolution (first {n_traces} synapses, scaled every 100 ms)")
ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="initial w=1")
ax.axvline(x=BASELINE_MS, color="red", linestyle="-", alpha=0.3, label="STDP on")
ax.axvline(
    x=BASELINE_MS + TRAINING_MS,
    color="orange",
    linestyle="-",
    alpha=0.3,
    label="STDP off",
)
ax.legend(fontsize=8)

# Weight distributions before/after
ax = axes[0, 1]
bins = np.linspace(
    min(w_before.min(), w_after.min()) - 0.05,
    max(w_before.max(), w_after.max()) + 0.05,
    40,
)
ax.hist(w_before, bins=bins, alpha=0.6, label="Before training", color="steelblue")
ax.hist(w_after, bins=bins, alpha=0.6, label="After training", color="coral")
ax.axvline(x=1.0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Weight multiplier (w)")
ax.set_ylabel("Count")
ax.set_title("Weight distribution")
ax.legend(fontsize=8)

ax = axes[1, 0]
total_offset = 0

for phase_label, it_data, t_data, t_offset, color in [
    ("Baseline", it_base, t_base, 0, "gray"),
    ("Training", it_train, t_train, BASELINE_MS, "tab:blue"),
    ("Eval", it_eval, t_eval, BASELINE_MS + TRAINING_MS, "tab:green"),
]:
    if len(it_data) > 0 and len(t_data) > 0:
        ax.scatter(
            np.array(t_data) + t_offset,
            np.array(it_data),
            s=1,
            alpha=0.5,
            color=color,
            label=phase_label,
        )

ax.axvline(x=BASELINE_MS, color="red", linestyle="-", alpha=0.5, label="STDP on")
ax.axvline(
    x=BASELINE_MS + TRAINING_MS,
    color="orange",
    linestyle="-",
    alpha=0.5,
    label="STDP off",
)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Neuron GID")
ax.set_title("Spike raster")
ax.legend(fontsize=7, markerscale=5)

ax = axes[1, 1]
delta_w = w_after - w_before
sort_idx = np.argsort(delta_w)
colors = ["coral" if d > 0 else "steelblue" for d in delta_w[sort_idx]]
ax.bar(range(len(delta_w)), delta_w[sort_idx], color=colors, width=1.0)
ax.axhline(y=0, color="k", linewidth=0.5)
ax.set_xlabel("Synapse (sorted)")
ax.set_ylabel("delta w")
ax.set_title("Per-synapse weight change")

plt.tight_layout()
plt.savefig("stdp_training.png", dpi=150)
print("\nPlot saved to stdp_training.png")
plt.show()

env.close()
