# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
#   "distwq",
# ]
# ///
"""
mpirun -n {subworld_size * num_workers + 1} python examples/distributed_workers.py
"""

import os

import numpy as np

from livn.integrations.distwq import DistributedEnv
from livn.utils import P
from livn.types import Encoding
from livn.decoding import ChannelRecording


class Constant(Encoding):
    def __call__(self, env, t_end, inputs):
        t_stim = inputs
        # Set up a 20ms stimulus in channel 1 and 4
        channel_inputs = np.zeros([t_end, 16])
        for r in range(20):
            for c in [1, 4]:
                channel_inputs[t_stim + r, c] = 750
        return env.cell_stimulus(channel_inputs)


env = DistributedEnv(
    "./systems/data/S1",
    subworld_size=3,  # processors per workers
)

env.init()

env.record_membrane_current()
env.record_spikes()
env.apply_model_defaults()

if P.is_root(os.getenv("DISTWQ_CONTROLLER_RANK", 0)):
    responses = env(
        ChannelRecording(duration=100),
        # different features to be processed by different workers
        inputs=[10, 20],
        encoding=Constant(),
    )
    for rid, response in enumerate(responses):
        cit, ct, iv, vv, im, p = response

        per_channel_firing_rate = {
            key: np.nan_to_num(np.mean(np.unique(val, return_counts=True)[1]))
            for key, val in cit.items()
        }
        print(rid, " firing rates: ", per_channel_firing_rate)

env.shutdown()
