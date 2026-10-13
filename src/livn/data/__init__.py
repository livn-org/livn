from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from livn.io import IO
    from livn.run import Events, Run

__all__ = ["Missing", "Recording", "open", "store"]


class Missing(LookupError):
    """The recording does not hold what was promised."""


@runtime_checkable
class Recording(Protocol):
    fs: float
    io: IO | None
    channel_ids: np.ndarray
    bad_channels: np.ndarray
    metadata: dict

    @property
    def segments(self) -> int: ...

    def num_samples(self, segment: int) -> int: ...

    def read(self, segment, start, length, channels=None) -> np.ndarray:
        """float32 ``(length, channels)`` in uV; ``channels`` are channel ids"""
        ...

    def spikes(self, segment) -> Events:
        """The sorted or ground-truth units' spikes, ids = unit ids"""
        ...

    def inputs(self, segment, start, length) -> np.ndarray:
        """The delivered input, float32 ``(length, k)``; raises :class:`Missing` if none"""
        ...

    def ground_truth(self, name, segment=0) -> Any: ...

    def to_run(self, segment=0, start=0, length=None) -> Run: ...


def open(
    path_or_uri: str | os.PathLike, *, backend: str | None = None, **kw
) -> Recording:
    """Open a recording.

    ``backend`` is ``"rsf"`` or ``"arrow"``; by default it is read off the location where an
    ``hf://<owner>/<repo>/<config>`` uri is downloaded (only that config's files) and opened
    as Arrow, a directory with ``traces.zarr`` is an rsf session, and a saved ``datasets``
    directory is Arrow. Keyword arguments go to the backend (``split=`` for Arrow;
    ``revision=``, ``cache_dir=``, ``token=`` for a hub uri).
    """
    location = os.fspath(path_or_uri)

    if location.startswith("hf://"):
        if backend not in (None, "arrow"):
            raise ValueError(f"{location} is a hub dataset, which opens as 'arrow'")
        from livn.data import arrow

        hub = {k: kw.pop(k) for k in ("revision", "cache_dir", "token") if k in kw}
        return arrow.ArrowRecording(arrow.snapshot(location, **hub), **kw)

    if backend is None:
        if os.path.isdir(os.path.join(location, "traces.zarr")):
            backend = "rsf"
        else:
            from livn.data import arrow

            if arrow.is_dataset(location):
                backend = "arrow"
            else:
                raise ValueError(
                    f"{location} is neither an rsf session (traces.zarr) nor a saved "
                    "Arrow dataset; pass backend= to say which to try"
                )

    if backend == "rsf":
        from livn.data import rsf

        return rsf.RsfRecording(location, **kw)
    if backend == "arrow":
        from livn.data import arrow

        return arrow.ArrowRecording(location, **kw)
    raise ValueError(f"unknown backend {backend!r}; choose 'rsf' or 'arrow'")


def store(run: Run | None, path: str | os.PathLike, *, backend: str = "rsf", **kw):
    """Write ``run`` with a backend: ``rsf.store(run, path, env=..., ...)`` or
    ``arrow.store(run, path, env=..., ...)``; the keywords are the backend's."""
    if backend == "rsf":
        from livn.data import rsf

        return rsf.store(run, path, **kw)
    if backend == "arrow":
        from livn.data import arrow

        return arrow.store(run, path, **kw)
    raise ValueError(f"unknown backend {backend!r}; choose 'rsf' or 'arrow'")
