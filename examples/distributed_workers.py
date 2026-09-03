# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn[neuron]",
# ]
# ///
"""
mpirun -n {subworld_size * num_workers + 1} python examples/distributed_workers.py
"""

import numpy as np

from livn.decoding import ChannelRecording
from livn.env.distributed import DistributedEnv
from livn.io import MEA, electrode_array_coordinates_for_area
from livn.system import System
from livn.types import Encoding

SYSTEM = "./systems/graphs/EI"

# the cultures ship without an array, so mount one over the system's extent
(xmin, ymin, _), (xmax, ymax, _) = System(SYSTEM).bounding_box
mea = MEA(electrode_array_coordinates_for_area(400, ((xmin, ymin), (xmax, ymax))))


class Constant(Encoding):
    def __call__(self, env, t_end, inputs):
        t_stim = inputs
        # Set up a 20ms stimulus in channel 1 and 4
        channel_inputs = np.zeros([t_end, env.io.num_channels])
        for r in range(20):
            for c in [1, 4]:
                channel_inputs[t_stim + r, c] = 1.5
        return env.cell_stimulus(channel_inputs)


env = DistributedEnv(
    SYSTEM,
    io=mea,
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
