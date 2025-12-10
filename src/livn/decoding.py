from livn.types import Decoding
from livn.utils import P



class ChannelRecording(Decoding):
    def setup(self, env):
        env.record_spikes()

    def __call__(self, env, it, tt, iv, vv, im, m):
        # per-rank electrode potential, sum-reduced for each channel [T, #channels]
        p = P.reduce_sum(env.potential_recording(m), all=True, comm=env.comm)

        cit, ct = env.channel_recording(it, tt)

        cit, ct, iv, vv = P.gather(cit, ct, iv, vv, comm=env.comm)

        if P.is_root(comm=env.comm):
            cit, ct, iv, vv = P.merge(cit, ct, iv, vv)
            return cit, ct, iv, vv, env.io.channel_ids, p
