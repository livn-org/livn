import json
import os
from collections.abc import Mapping, Sequence

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

    def environment(self, frame=None, **kwargs) -> dict:
        return {}

    def channels(self) -> dict:
        return {}

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


_STARTED_AT = "2026-10-04T09:12:33.000000+00:00"
_RECORDING_ID = "0f6c1d2e3b4a59687766554433221100"


def _node(directory: str, attributes: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "zarr.json"), "w", encoding="utf-8") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": attributes}, f)


def _array(group, name: str, values) -> None:
    values = np.asarray(values)
    array = group.create_array(
        name,
        shape=values.shape,
        dtype=values.dtype,
        chunks=tuple(max(1, s) for s in values.shape),
        compressors=None,
        fill_value=0,
    )
    if values.size:
        array[...] = values


def _strings(group, name: str, values: Sequence[str]) -> None:
    array = group.create_array(
        name, shape=(len(values),), dtype=str, chunks=(max(1, len(values)),)
    )
    if len(values):
        array[:] = np.asarray([str(v) for v in values], dtype=object)


def _physical_session(
    path: str,
    *,
    fs: float = 10_000.0,
    n_frames: int = 1000,
    channels: Sequence[str] = ("a-000", "a-001"),
    traces: np.ndarray | None = None,
    gains: Sequence[float] | None = None,
    dtype: str = "int16",
    tick_rate: float | None = None,
    anchors: Sequence[tuple[int, int]] | None = ((0, 912_384_050),),
    gaps: Sequence[tuple[int, int]] | None = (),
    dropped: Sequence[tuple[int, int, str]] = (),
    markers: Sequence[tuple[int, str]] = (),
    clipped: Sequence[int] | None = None,
    stimulation: Sequence[Mapping] = (),
    stim_events: Sequence[tuple] | None = (),
    finished: bool = True,
    notes: Mapping[str, Sequence[tuple[str, dict]]] | None = None,
    producer: str = "acme",
    label: str = "slice 3",
    started_at: str = _STARTED_AT,
    conditions: Sequence[tuple] | None = None,
    conditions_items: Mapping | None = None,
    system: Mapping | None = None,
    extra_attributes: Mapping | None = None,
    documents: Mapping[str, object] | None = None,
) -> str:
    import zarr

    channels = list(channels)
    if traces is None:
        traces = np.arange(n_frames * len(channels)).reshape(n_frames, len(channels))
    traces = np.asarray(traces).astype(dtype)
    gains = [0.195] * len(channels) if gains is None else list(gains)
    first_tick = None if not anchors or anchors[0][0] != 0 else int(anchors[0][1])

    store = os.path.join(path, "traces.zarr")
    unknown = {
        "role": "acquisition",
        "status": "unknown",
        "transform": None,
        "attribution": "amplifier",
    }
    group = {"id": "a1b2c3d4e5f60718293a4b5c6d7e8f90", "sequence": 0}
    record = {
        "format": 1,
        "recording_id": _RECORDING_ID,
        "env": {
            "substrate": "physical",
            "system": None if system is None else dict(system),
            "model": None,
            "seed": None,
        },
        "space": "channel",
        "rows": {c: [unknown] for c in channels},
        "started_at": started_at,
        "first_sample_tick": first_tick,
        "group": group,
    }
    root = zarr.open_group(
        store,
        mode="w",
        attributes={
            "sampling_frequency": fs,
            "num_segments": 1,
            "annotations": {"rsf_provenance": record},
        },
    )
    _strings(root, "channel_ids", channels)
    _array(root, "traces_seg0", traces)
    _array(root, "properties/gain_to_uV", np.asarray(gains, dtype=np.float64))
    _array(root, "properties/offset_to_uV", np.zeros(len(channels)))
    _strings(root, "properties/channel", channels)
    _strings(root, "properties/band", ["wide"] * len(channels))
    if clipped is not None:
        _array(root, "properties/clipped_samples", np.asarray(clipped, dtype=np.uint64))

    rsf = os.path.join(store, "rsf")
    tables = {
        "tick_anchors": {
            "frame": np.asarray([a[0] for a in anchors or ()], dtype=np.uint64),
            "tick": np.asarray([a[1] for a in anchors or ()], dtype=np.uint64),
        },
        "stim_markers": {
            "tick": np.asarray([m[0] for m in markers], dtype=np.uint64),
            "channel": [m[1] for m in markers],
        },
        "dropped_spans": {
            "frame_start": np.asarray([d[0] for d in dropped], dtype=np.uint64),
            "n_frames": np.asarray([d[1] for d in dropped], dtype=np.uint32),
            "channel": [d[2] for d in dropped],
        },
    }
    if stim_events is not None:  # (frame, channel, value, program, event, timing)
        tables["stim_events"] = {
            "frame": np.asarray([r[0] for r in stim_events], dtype=np.uint64),
            "channel": [r[1] for r in stim_events],
            "value": np.asarray([r[2] for r in stim_events], dtype=np.float32),
            "program": np.asarray([r[3] for r in stim_events], dtype=np.uint32),
            "event": np.asarray([r[4] for r in stim_events], dtype=np.uint32),
            "timing": [r[5] for r in stim_events],
        }
    if gaps is not None:
        tables["gaps"] = {
            "frame": np.asarray([g[0] for g in gaps], dtype=np.uint64),
            "n_frames": np.asarray([g[1] for g in gaps], dtype=np.uint64),
        }
    if conditions is not None:  # (frame, quantity, item, value, units, basis)
        tables["conditions"] = {
            "frame": np.asarray([r[0] for r in conditions], dtype=np.uint64),
            "quantity": [r[1] for r in conditions],
            "item": [r[2] for r in conditions],
            "value": np.asarray([r[3] for r in conditions], dtype=np.float64),
            "units": [r[4] for r in conditions],
            "basis": [r[5] for r in conditions],
        }
    for name, columns in tables.items():
        table = zarr.open_group(
            os.path.join(rsf, name),
            mode="w",
            attributes={"items": dict(conditions_items)}
            if name == "conditions" and conditions_items is not None
            else None,
        )
        for column, values in columns.items():
            if isinstance(values, list):
                _strings(table, column, values)
            else:
                _array(table, column, values)

    if stimulation:
        _node(os.path.join(rsf, "stimulation"), {})
        for i, attribute in enumerate(stimulation):
            _node(
                os.path.join(rsf, "stimulation", f"{i:04d}"), {"stimulation": attribute}
            )

    for name, document in (documents or {}).items():
        os.makedirs(rsf, exist_ok=True)
        with open(os.path.join(rsf, name), "w", encoding="utf-8") as f:
            json.dump(document, f)

    for kind, versions in (notes or {}).items():
        folder = os.path.join(rsf, "notes", kind)
        os.makedirs(folder, exist_ok=True)
        for key, version in versions:
            with open(os.path.join(folder, key), "w", encoding="utf-8") as f:
                json.dump(version, f)

    attributes = {
        "format": 1,
        "recording_id": _RECORDING_ID,
        "created_at": started_at,
        "label": label,
        "producer": {"name": producer, "version": "0.1.0"},
        "backend_id": "mcs",
        "compression": {
            "codec": "none",
            "level": 0,
            "shuffle": "none",
            "lsb_truncate_bits": 0,
        },
        "compressed_bytes": 0,
        "group": group,
    }
    if finished:
        attributes["finished_at"] = "2026-10-04T09:20:00.000000+00:00"
    if tick_rate is not None:
        attributes["hardware_tick_rate_hz"] = float(tick_rate)
    _node(rsf, {**attributes, **(extra_attributes or {})})
    return path


def _rsf_note(
    item: str, at: str, text: str, *, removed: bool = False, frame=None
) -> dict:
    return {
        "item": item,
        "at": at,
        "zone": "Europe/Berlin",
        "author": "Mara Okafor",
        "removed": removed,
        "text": text,
        "frame": frame,
    }


@pytest.fixture
def physical_session():
    return _physical_session


@pytest.fixture
def rsf_note():
    return _rsf_note
