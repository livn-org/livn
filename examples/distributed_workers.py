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

from livn.integrations.distwq import DistributedEnv
from livn.utils import P
import numpy as np
from livn.types import Decoding, Encoding
import os


class Constant(Encoding):
    def __call__(self, env, t_end, features):
        t_stim = features
        # Set up a 20ms stimulus in channel 1 and 4
        inputs = np.zeros([t_end, 16])
        for r in range(20):
            for c in [1, 4]:
                inputs[t_stim + r, c] = 750
        return env.cell_stimulus(inputs)


class Outread(Decoding):
    def __call__(self, env, it, tt, iv, vv, im, m):
        # per-rank electrode potential, sum-reduced for each channel [T, #channels]
        p = P.reduce_sum(env.potential_recording(m), all=True, comm=env.comm)

        cit, ct = env.channel_recording(it, tt)

        cit, ct, iv, vv = P.gather(cit, ct, iv, vv, comm=env.comm)

        if P.is_root(comm=env.comm):
            cit, ct, iv, vv = P.merge(cit, ct, iv, vv)
            return cit, ct, iv, vv, env.io.channel_ids, p


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
        Outread(duration=100),
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
