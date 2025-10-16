# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
# ]
# ///

import numpy as np

from livn.backend import backend
from livn.env import Env
from livn.models.lif import LIF
from livn.utils import P
from livn.system import predefined

system_name = "S1"

# pick a model
model = LIF() if backend() == "brian2" else None

env = Env(predefined(system_name), model).init()

env.apply_model_defaults()
env.record_spikes()
env.record_voltage()
env.record_membrane_current()


warmup = 0
trial_length = 100
t_stim = 10
t_end = warmup + trial_length


# Set up a 20ms stimulus in channel 1 and 4
inputs = np.zeros([t_end, 16])
for r in range(20):
    for c in [1, 4]:
        inputs[warmup + t_stim + r, c] = 750
stimulus = env.cell_stimulus(inputs)

# Run with stimulation
it, t, iv, v, im, m = env.run(t_end, stimulus=stimulus)

# per-rank electrode potential, sum-reduced
p = P.reduce_sum(env.potential_recording(m), all=True)

it, t = it[t >= warmup], t[t >= warmup] - warmup
t_end = t_end - warmup

cit, ct = env.channel_recording(it, t)

cit, ct, iv, v = P.gather(cit, ct, iv, v)

if P.is_root():
    cit, ct, iv, v = P.merge(cit, ct, iv, v)

    per_channel_firing_rate = {
        key: np.nan_to_num(
            np.mean(np.unique(val, return_counts=True)[1] / (t_end / 100))
        )
        for key, val in cit.items()
    }
    print("Output firing rates: ", per_channel_firing_rate)

    rates = [
        np.nan_to_num(np.mean(np.unique(val, return_counts=True)[1] / (t_end / 100)))
        for channel, val in cit.items()
    ]
    rate_mean = np.mean(rates)
    rate_std = np.std(rates)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))

    for channel in cit:
        i = cit[channel]
        t = ct[channel]
        plt.plot(t, i, ".", ms=1, label=channel)

    plt.legend()
    plt.ylabel("Channel")
    plt.xlabel("Time (ms)")
    plt.title(f"Raster Plot (Mean rate: {rate_mean:.2f} Hz, Std: {rate_std:.2f} Hz)")
    plt.tight_layout()
    fig.savefig("spikes.png")

    plt.figure(figsize=(12, 6))
    for neuron_idx in range(min(3, v.shape[0])):
        q = v[neuron_idx, :]
        plt.plot(np.arange(len(q)), q, label=f"Neuron {neuron_idx}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage")
    plt.title("Voltage Traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig("traces.png")

    plt.figure(figsize=(12, 6))
    timesteps = p.shape[0]
    for ch in range(p.shape[1]):
        plt.plot(np.arange(timesteps), p[:, ch], label=f"Ch {ch}", alpha=0.7)
    plt.xlabel("Time (samples)")
    plt.ylabel("Electrode potential (µV)")
    plt.title("Electrode Potentials per Channel")
    if p.shape[1] <= 8:
        plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("potentials.png")
