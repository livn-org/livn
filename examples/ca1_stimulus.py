# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "miv_simulator @ git+https://github.com/GazzolaLab/MiV-Simulator.git@e8e417f",
#   "livn",
#   "huggingface_hub",
# ]
# ///
import os
import numpy as np

os.environ["LIVN_BACKEND"] = "neuron"

from livn.env import Env
from livn.system import predefined
from livn.utils import P
from livn.decoding import ChannelRecording
from neuron import h

# CA1 debug system containing a single PYR cell
env = Env(predefined("CA1d")).init()

t_end = 500  # ms

# Drive input cells directly with spike trains, transmitting to PYR
for pop_name in ["EC", "CA3"]:
    for gid, cell in env.artificial_cells.get(pop_name, {}).items():
        # Poisson-like spike train: ~50 Hz burst from 100-200ms
        spike_times = np.arange(100, 200, 20.0) + np.random.uniform(0, 5)
        cell.play(h.Vector(spike_times.astype(np.float64)))

# Observe through electrode
cit, ct, civ, cv, cim, cm = env(
    decoding=ChannelRecording(duration=t_end),
)

if P.is_root():
    print(
        "Channel firing rates: ",
        {
            key: np.nan_to_num(
                np.mean(np.unique(val, return_counts=True)[1] / (t_end / 100))
            )
            for key, val in cit.items()
        },
    )
