# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
# ]
# ///
"""
mpirun -n {subworld_size * num_workers + 1} python examples/distributed_workers.py
"""

import numpy as np

from livn.env.distributed import DistributedEnv
from livn.types import Encoding
from livn.decoding import ChannelRecording


class Constant(Encoding):
    def __call__(self, env, t_end, inputs):
        t_stim = inputs
        # Set up a 20ms stimulus in channel 1 and 4
        channel_inputs = np.zeros([t_end, 16])
        for r in range(20):
            for c in [1, 4]:
                channel_inputs[t_stim + r, c] = 1.5
        return env.cell_stimulus(channel_inputs)


env = DistributedEnv(
    "./systems/graphs/EI1",
    subworld_size=3,  # processors per workers
)

env.init()

env.record_membrane_current()
env.record_spikes()
env.apply_model_defaults()

if env.is_root():
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
