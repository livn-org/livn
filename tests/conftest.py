import numpy as np
import pytest


class _RunRecording:
    """A :class:`~livn.run.Run` held in memory, as a ``livn.data.Recording`` of one segment.

    The run needs a channel ``potential`` for ``read``; ``io`` is whatever recorded it.
    """

    def __init__(self, run, io=None):
        self.run = run
        self.io = io
        potential = run.potential
        self.fs = float("nan") if potential is None else 1000.0 / float(potential.dt)
        self.channel_ids = (
            np.zeros(0, dtype=np.int64)
            if potential is None
            else np.asarray(potential.ids)
        )
        self.bad_channels = np.zeros(0, dtype=np.int64)
        self.metadata: dict = {"format": "run"}

    @property
    def segments(self) -> int:
        return 1

    def _segment(self, segment: int) -> None:
        if segment != 0:
            raise IndexError(f"a run is one segment, not segment {segment}")

    def _potential(self):
        from livn.data import Missing

        if self.run.potential is None or self.run.potential.values is None:
            raise Missing("the run holds no channel potential")
        return self.run.potential

    def num_samples(self, segment: int = 0) -> int:
        self._segment(segment)
        return int(self._potential().values.shape[1])

    def valid_spans(self, segment: int = 0) -> np.ndarray:
        return np.asarray([[0, self.num_samples(segment)]], dtype=np.int64)

    def read(self, segment, start, length, channels=None) -> np.ndarray:
        self._segment(segment)
        potential = self._potential()
        n = potential.values.shape[1]
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > n:
            raise IndexError(
                f"the window [{start}, {start + length}) is outside 0..{n}"
            )
        values = np.asarray(potential.values)
        if channels is not None:
            ids = np.asarray(potential.ids).tolist()
            values = values[[ids.index(int(c)) for c in np.asarray(channels).ravel()]]
        return np.ascontiguousarray(values[:, start : start + length].T, np.float32)

    def spikes(self, segment: int = 0):
        from livn.data import Missing

        self._segment(segment)
        if self.run.spikes is None:
            raise Missing("the run recorded no spikes")
        return self.run.spikes

    def inputs(self, segment, start, length) -> np.ndarray:
        from livn.data import Missing

        raise Missing("a Run carries no input series")

    def ground_truth(self, name: str, segment: int = 0):
        from livn.data import Missing

        self._segment(segment)
        key = "current" if name == "currents" else name
        channel = self.run.channel(key)
        if channel is None:
            raise Missing(f"the run holds no {name!r}")
        return channel

    def to_run(self, segment=0, start=0, length=None):
        self._segment(segment)
        if start == 0 and length is None:
            return self.run
        dt = 1000.0 / self.fs
        stop = self.num_samples() if length is None else start + length
        return self.run.slice(start * dt, stop * dt)


@pytest.fixture
def run_recording():
    return _RunRecording
