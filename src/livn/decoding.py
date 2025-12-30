from livn.types import Decoding
from livn.utils import P
from pydantic import Field


class Slice(Decoding):
    """Slice decoding

    Slices the system response into [start -> stop]

    Args:
        start: Start time in ms
        duration: Duration of the slice in ms
    """

    start: int = 0
    duration: int = Field(validation_alias="stop")

    def __call__(self, env, it, tt, iv, vv, im, m):
        stop = self.duration

        # spikes
        if it is not None and tt is not None:
            mask = (tt >= self.start) & (tt < stop)
            it = it[mask]
            tt = tt[mask] - self.start

        # voltage [n_neurons, T]
        if iv is not None and vv is not None:
            v_dt = env.voltage_recording_dt
            start_idx = int(self.start / v_dt)
            stop_idx = int(stop / v_dt)
            vv = vv[:, start_idx:stop_idx]

        # membrane currents [n_neurons, T]
        if im is not None and m is not None:
            m_dt = env.membrane_current_recording_dt
            start_idx = int(self.start / m_dt)
            stop_idx = int(stop / m_dt)
            m = m[:, start_idx:stop_idx]

        return it, tt, iv, vv, im, m


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
