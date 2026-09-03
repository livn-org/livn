# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
#   "matplotlib",
# ]
# ///
import matplotlib.pyplot as plt
import numpy as np

from livn.env import Env
from livn.policy import PulseSweepPolicy
from livn.system import predefined

ELECTRODE = 0

env = Env(predefined("EI")).init()
env.apply_default_params()
env.record_spikes()

policy = PulseSweepPolicy(
    amplitudes=(200.0, 800.0),
    repeats=1,
    trial_ms=20.0,
    onset_ms=10.0,
    pulse_ms=1.0,
    dt=0.1,
).for_array(len(env.io.channel_ids), channels=[ELECTRODE])

run = env.run(policy.extent_ms, stimulus=policy)

ids = np.asarray(run.spike_ids)
times = np.asarray(run.spike_times)

print(f"{len(times)} spikes from {len(np.unique(ids))} cells")
for pulse, amplitude in policy.schedule():
    responded = ((times >= pulse) & (times < pulse + 20.0)).sum()
    print(f"  {amplitude:>6.0f} uA -> {responded:>5d} spikes in the 20 ms after it")

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(times, ids, "|", color="#16386e", ms=2.5, mew=0.6, alpha=0.8)
for pulse, amplitude in policy.schedule():
    ax.axvline(pulse, color="#d81b1b", lw=1.2, alpha=0.8)
    ax.annotate(
        f"{amplitude:.0f} uA",
        (pulse, 0.98),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=9,
        color="#d81b1b",
    )
ax.set_xlabel("time (ms)")
ax.set_ylabel("cell")
ax.set_xlim(0, policy.extent_ms)

fig.tight_layout()
fig.savefig("stimulus_sweep.png", dpi=150)
print("wrote stimulus_sweep.png")
