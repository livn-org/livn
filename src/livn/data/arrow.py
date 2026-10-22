"""The Arrow layout.

    <root>/dataset_dict.json
    <root>/<split>/{state.json, dataset_info.json, data-*.arrow}
    <root>/conversion_metadata.json   sampling rate, channel count, bad channels, provenance
    <root>/templates.npz              unit templates and ``probe_positions``
    <root>/dataset_stats.json         normalisation stats with their ``preprocess_fingerprint``
    <root>/io.json                    livn's IO document (written by :func:`store`)

Channel space:
--------------
``sample_id`` str                              segment name
``cp``       list<list<float32>> ``[C][T]``    channel potential in uV
``cit``      list<list<int32>> ``[C][n]``      unit id of each spike seen on channel c
``ctt``      list<list<float64>> ``[C][n]``    its time in ms relative to the segment
``in``       list<list<float32>> ``[K][T]``    optional: delivered input on the frame grid


Optionally in cell space:
- the simulator's ground truth: ``it``/``tt`` (spike unit ids and times),
- ``iv``/``vv`` (+ ``vs`` sections, ``voltage_dt``) voltage,
- ``im``/``mp`` (+ ``ms``, ``current_dt``) membrane current.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from livn.data import Missing

if TYPE_CHECKING:
    from livn.io import IO
    from livn.policy import Policy
    from livn.run import Events, Run, Series

CONVERSION = "conversion_metadata.json"
TEMPLATES = "templates.npz"
STATS = "dataset_stats.json"
IO_DOCUMENT = "io.json"
DATASET_DICT = "dataset_dict.json"
PATCH = "patch.npz"

# cell-space ground truth: run channel -> columns
_GROUND_TRUTH = {
    "spikes": ("it", "tt"),
    "voltage": ("iv", "vv", "vs", "voltage_dt"),
    "current": ("im", "mp", "ms", "current_dt"),
}


class ArrowError(ValueError):
    """A dataset that cannot be read or written in this layout."""


def _pa():
    try:
        import pyarrow as pa
    except ImportError as e:
        raise ImportError(
            "livn.data.arrow needs pyarrow and datasets; install them with "
            "`pip install datasets`"
        ) from e
    return pa


def _datasets():
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "livn.data.arrow needs the `datasets` package; install it with "
            "`pip install datasets`"
        ) from e
    return datasets


def _json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _split_directory(root: str, split: str | None) -> str:
    if split is not None and os.path.isfile(os.path.join(root, split, "state.json")):
        return os.path.join(root, split)
    if os.path.isfile(os.path.join(root, "state.json")):
        return root
    have = sorted(
        d
        for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "state.json"))
    )
    raise ArrowError(
        f"{root} has no split {split!r}"
        + (f"; its splits are {have}" if have else " and is not a saved dataset")
    )


def is_dataset(path: str | os.PathLike) -> bool:
    path = os.fspath(path)
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "state.json")):
        return True
    if os.path.isfile(os.path.join(path, DATASET_DICT)):
        return True
    return any(
        os.path.isfile(os.path.join(path, d, "state.json")) for d in os.listdir(path)
    )


class ArrowRecording:
    """The rows of one split as segments, read lazily from the memory-mapped Arrow files.

    Args:
        path: The dataset root (with split directories) or one saved split
        split: Which split to open when ``path`` has several
    """

    def __init__(self, path: str | os.PathLike, split: str | None = "train"):
        self.path = os.fspath(path)
        self.split = split
        self._open()

    def _open(self) -> None:
        datasets = _datasets()

        root = self.path
        directory = _split_directory(root, self.split)
        self._table_dataset = datasets.load_from_disk(directory)
        if getattr(self._table_dataset, "_indices", None) is not None:
            raise ArrowError(f"{directory} holds an indexed view, not saved rows")
        self._table = self._table_dataset.data
        self.columns = list(self._table.column_names)
        self._chunks: dict = {}
        self._starts: dict = {}

        # the side files live next to the splits; a single saved split has them beside it
        side = root
        parent = os.path.dirname(os.path.abspath(root))
        if (
            directory == root
            and not os.path.isfile(os.path.join(root, CONVERSION))
            and os.path.isfile(os.path.join(parent, CONVERSION))
        ):
            side = parent

        conversion: dict = {}
        if os.path.isfile(os.path.join(side, CONVERSION)):
            conversion = _json(os.path.join(side, CONVERSION))
        templates = None
        if os.path.isfile(os.path.join(side, TEMPLATES)):
            with np.load(os.path.join(side, TEMPLATES)) as z:
                templates = {k: z[k] for k in z.files}
        stats = None
        if os.path.isfile(os.path.join(side, STATS)):
            stats = _json(os.path.join(side, STATS))

        fs = conversion.get("sampling_frequency")
        if fs is None and templates is not None and "sampling_frequency" in templates:
            fs = float(templates["sampling_frequency"])
        self.fs = float("nan") if fs is None else float(fs)

        n_channels = conversion.get("num_channels")
        if n_channels is None and "cp" in self.columns and len(self._table_dataset):
            n_channels = self._channels("cp", 0)
        n_channels = int(n_channels or 0)
        ids = conversion.get("channel_ids")
        self.channel_ids = (
            np.asarray(ids, dtype=np.int64)
            if ids is not None
            else np.arange(n_channels, dtype=np.int64)
        )
        self.bad_channels = np.asarray(
            conversion.get("bad_channel_ids", []), dtype=np.int64
        )

        self.metadata: dict[str, Any] = {
            "format": "arrow",
            "path": root,
            "split": self.split,
            "columns": list(self.columns),
            "conversion": conversion,
            "templates": templates,
            "dataset_stats": stats,
            "preprocess_fingerprint": None
            if stats is None
            else stats.get("preprocess_fingerprint"),
            "patch": (
                os.path.join(side, PATCH)
                if os.path.isfile(os.path.join(side, PATCH))
                else None
            ),
            "unit_peak_channel": conversion.get("unit_peak_channel"),
            "spike_peak_channel": {},
        }

        self.io = self._restore_io(side, templates)

    def _restore_io(self, side: str, templates):
        document = os.path.join(side, IO_DOCUMENT)
        if os.path.isfile(document):
            from livn.data.rsf import _read_operator

            io = _read_operator(document)
            if io is not None:
                return io
        if templates is not None and "probe_positions" in templates:
            from livn.io import MEA

            positions = np.asarray(templates["probe_positions"], dtype=np.float64)
            ids = self.channel_ids
            if len(ids) != len(positions):
                ids = np.arange(len(positions))
            xyz = np.zeros((len(positions), 3))
            xyz[:, : positions.shape[1]] = positions
            return MEA(np.column_stack([ids, xyz]))
        return None

    def __getstate__(self) -> dict:
        return {"path": self.path, "split": self.split}

    def __setstate__(self, state: dict) -> None:
        self.path, self.split = state["path"], state["split"]
        self._open()

    @property
    def segments(self) -> int:
        return int(self._table.num_rows)

    def sample_id(self, segment: int) -> str:
        return str(self._table.column("sample_id")[self._row(segment)].as_py())

    def num_samples(self, segment: int) -> int:
        column = "cp" if "cp" in self.columns else None
        if column is None:
            raise Missing(f"{self.path} has no `cp` column: it holds no channel signal")
        return int(self._lengths(column, self._row(segment))[0])

    def valid_spans(self, segment: int) -> np.ndarray:
        """The whole segment for which the Arrow layout states no lost or dropped frames."""
        return np.asarray([[0, self.num_samples(segment)]], dtype=np.int64)

    def read(self, segment, start, length, channels=None) -> np.ndarray:
        """Samples ``[start, start + length)`` of the given channel ids (all by default), as
        float32 ``(length, channels)`` in uV."""
        if "cp" not in self.columns:
            raise Missing(f"{self.path} has no `cp` column: it holds no channel signal")
        return self._window("cp", segment, start, length, channels)

    def environment(self, frame=None, **kwargs) -> dict:
        """``{}`` for unknown."""
        return {}

    def inputs(self, segment, start, length) -> np.ndarray:
        """The delivered input on the frame grid, float32 ``(length, channels)``."""
        if "in" not in self.columns:
            raise Missing(
                f"{self.path} has no `in` column: it holds no input series, and zeros "
                "would be a different dataset"
            )
        return self._window("in", segment, start, length, None)

    def spikes(self, segment: int = 0) -> Events:
        """The segment's units' spikes, ordered by time, ids = unit ids.

        Cell-space ``it``/``tt`` are used when the row has them. Otherwise the per-channel
        ``cit``/``ctt`` are merged: a spike seen on several channels is one spike, and the
        channel that sees a unit most often is its peak channel, kept in
        ``metadata["spike_peak_channel"][segment]``.
        """
        from livn.run import Events

        row = self._row(segment)
        duration = self._duration(row)
        if "it" in self.columns and "tt" in self.columns:
            ids = self._flat("it", row)[0]
            times = self._flat("tt", row)[0]
        elif "cit" in self.columns and "ctt" in self.columns:
            (units, counts), (stamps, _) = (
                self._flat("cit", row),
                self._flat("ctt", row),
            )
            channel = np.repeat(np.arange(len(counts)), counts)
            if len(units) != len(stamps):
                raise ArrowError(f"segment {segment}: cit and ctt differ in length")
            ids, times, peaks = _merge_channels(units, stamps, channel)
            self.metadata["spike_peak_channel"][segment] = {
                int(u): int(self.channel_ids[c])
                if len(self.channel_ids) > c
                else int(c)
                for u, c in peaks.items()
            }
        else:
            raise Missing(f"{self.path} has neither `cit`/`ctt` nor `it`/`tt` columns")
        order = np.argsort(times, kind="stable")
        return Events(
            ids=np.asarray(ids)[order].astype(np.int64),
            times=np.asarray(times)[order].astype(np.float64),
            t0=0.0,
            duration=duration,
        )

    def peak_channels(self, segment: int = 0) -> dict[int, int]:
        """``{unit: channel id}``: from the conversion metadata when it ships them, else the
        channel that sees each unit most often in this segment."""
        shipped = self.metadata["unit_peak_channel"]
        if shipped:
            return {int(k): int(v) for k, v in shipped.items()}
        self.spikes(segment)
        return dict(self.metadata["spike_peak_channel"].get(segment, {}))

    def ground_truth(self, name: str, segment: int = 0):
        """``"spikes"`` (Events), ``"voltage"``/``"currents"`` (Series of the cell-space
        columns), ``"templates"`` (the shipped unit templates) or ``"patch"`` (the patch
        recording, read from ``patch.npz`` on demand and not small)."""
        if name == "spikes":
            return self.spikes(segment)
        if name == "templates":
            if self.metadata["templates"] is None:
                raise Missing(f"{self.path} ships no {TEMPLATES}")
            return self.metadata["templates"]
        if name == "patch":
            if self.metadata["patch"] is None:
                raise Missing(f"{self.path} ships no {PATCH}")
            with np.load(self.metadata["patch"]) as z:
                return {k: z[k] for k in z.files}
        if name in ("current", "currents"):
            name = "current"
        elif name != "voltage":
            raise Missing(f"an Arrow dataset holds no ground truth {name!r}")
        series = self._cell_series(name, self._row(segment))
        if series is None:
            raise Missing(
                f"{self.path} has no cell-space {name} columns "
                f"({_GROUND_TRUTH[name][0]}/{_GROUND_TRUTH[name][1]})"
            )
        return series

    def to_run(self, segment=0, start=0, length=None) -> Run:
        from livn.run import Run, Series

        row = self._row(segment)
        have_signal = "cp" in self.columns
        n = self.num_samples(segment) if have_signal else None
        if have_signal:
            length = n - start if length is None else length
            start, length = _check_window(start, length, n)
            dt = 1000.0 / self.fs
            t0, duration = start * dt, length * dt
            run = Run(
                t0=t0,
                duration=duration,
                potential=Series(
                    ids=self.channel_ids,
                    values=self.read(segment, start, length).T,
                    dt=dt,
                    t0=t0,
                ),
            )
        else:
            if start or length is not None:
                raise Missing(f"{self.path} has no `cp` column to window")
            duration = self._duration(row)
            t0 = 0.0
            run = Run(t0=0.0, duration=duration)

        try:
            spikes = self.spikes(segment)
        except Missing:
            spikes = None
        if spikes is not None:
            times, ids = spikes.times, spikes.ids
            keep = (times >= t0) & (
                np.ones(len(times), dtype=bool)
                if duration is None
                else times < t0 + duration
            )
            run = run.add_spikes(ids[keep], times[keep] - t0)
        for name in ("voltage", "current"):
            series = self._cell_series(name, row)
            if series is None:
                continue
            if have_signal and (start or length != n):
                series = series.window(t0, t0 + duration, name)
            run = run.add(
                name,
                series.ids,
                series.values,
                dt=series.dt,
                kind="series",
                t0=t0,
                sections=series.sections,
            )
        return run

    def _row(self, segment: int) -> int:
        n = self.segments
        row = int(segment)
        if row < 0:
            row += n
        if not 0 <= row < n:
            raise IndexError(
                f"segment {segment} is outside the {n} rows of {self.path}"
            )
        return row

    def _locate(self, column: str, row: int):
        """``(chunk index, row within chunk)`` of a table row."""
        if column not in self._starts:
            chunked = self._table.column(column)
            sizes = [len(c) for c in chunked.chunks]
            self._starts[column] = np.concatenate([[0], np.cumsum(sizes)]).tolist()
        starts = self._starts[column]
        chunk = bisect_right(starts, row) - 1
        return chunk, row - starts[chunk]

    def _buffers(self, column: str, row: int):
        """``(outer offsets, inner offsets, flat values, local row)`` of a list-of-list column,
        as zero-copy numpy views of the mapped chunk (cached per chunk)."""
        chunk, local = self._locate(column, row)
        key = (column, chunk)
        views = self._chunks.get(key)
        if views is None:
            array = self._table.column(column).chunk(chunk)
            inner = array.values
            views = (
                _view(array.offsets),
                _view(inner.offsets),
                _view(inner.values),
            )
            self._chunks[key] = views
        return (*views, local)

    def _lengths(self, column: str, row: int) -> np.ndarray:
        """Length of each inner list of a row, so of each channel."""
        outer, inner, _, local = self._buffers(column, row)
        return np.diff(inner[outer[local] : outer[local + 1] + 1])

    def _channels(self, column: str, row: int) -> int:
        return len(self._lengths(column, row))

    def _flat(self, column: str, row: int):
        """``(all values of the row, per-inner-list counts)`` of a list column; a plain list
        column has one 'channel'."""
        chunk, local = self._locate(column, row)
        array = self._table.column(column).chunk(chunk)
        if _is_nested(array):
            outer, inner, flat, local = self._buffers(column, row)
            lo, hi = outer[local], outer[local + 1]
            return flat[inner[lo] : inner[hi]], np.diff(inner[lo : hi + 1])
        offsets = _view(array.offsets)
        flat = _view(array.values)
        return flat[offsets[local] : offsets[local + 1]], np.asarray(
            [offsets[local + 1] - offsets[local]]
        )

    def _window(self, column: str, segment, start, length, channels) -> np.ndarray:
        row = self._row(segment)
        outer, inner, flat, local = self._buffers(column, row)
        lo, hi = int(outer[local]), int(outer[local + 1])
        available = np.diff(inner[lo : hi + 1])
        shortest = int(available.min()) if len(available) else 0
        start, length = _check_window(start, length, shortest)
        if channels is None:
            picked = np.arange(hi - lo)
        else:
            lookup = {int(c): i for i, c in enumerate(self.channel_ids)}
            try:
                picked = np.asarray(
                    [lookup[int(c)] for c in np.asarray(channels).ravel()]
                )
            except KeyError as e:
                raise KeyError(
                    f"{e.args[0]!r} is not a channel of {self.path}"
                ) from None
        out = np.empty((length, len(picked)), dtype=np.float32)
        for j, c in enumerate(picked):
            base = int(inner[lo + c]) + start
            out[:, j] = flat[base : base + length]
        return out

    def _duration(self, row: int) -> float | None:
        """Segment length in ms, or ``None`` when no column says it."""
        if "cp" in self.columns and self.fs == self.fs:
            return float(self._lengths("cp", row)[0]) * 1000.0 / self.fs
        return None

    def _cell_series(self, name: str, row: int) -> Series | None:
        from livn.run import Series

        ids_c, values_c, sections_c, dt_c = _GROUND_TRUTH[name]
        if ids_c not in self.columns or values_c not in self.columns:
            return None
        ids = self._flat(ids_c, row)[0]
        flat, _ = self._flat(values_c, row)
        if len(ids) == 0:
            return None
        values = np.asarray(flat, dtype=np.float32).reshape(len(ids), -1)
        sections = None
        if sections_c in self.columns:
            sections = np.asarray(self._table.column(sections_c)[row].as_py())
        dt = (
            float(self._table.column(dt_c)[row].as_py())
            if dt_c in self.columns
            else 1000.0 / self.fs
        )
        return Series(
            ids=np.asarray(ids, dtype=np.int64),
            values=values,
            dt=dt,
            t0=0.0,
            sections=sections,
        )


def _is_nested(array) -> bool:
    pa = _pa()

    def is_list(t):
        return pa.types.is_list(t) or pa.types.is_large_list(t)

    return is_list(array.type) and is_list(array.type.value_type)


def _view(array) -> np.ndarray:
    try:
        return array.to_numpy(zero_copy_only=True)
    except Exception:
        return array.to_numpy(zero_copy_only=False)


def _check_window(start, length, n: int) -> tuple[int, int]:
    start, length = int(start), int(length)
    if start < 0 or length < 0 or start + length > n:
        raise IndexError(f"the window [{start}, {start + length}) is outside 0..{n}")
    return start, length


def _merge_channels(units, stamps, channel):
    if len(units) == 0:
        return units, stamps, {}
    order = np.lexsort((stamps, units))
    units, stamps, channel = units[order], stamps[order], channel[order]
    first = np.ones(len(units), dtype=bool)
    first[1:] = (units[1:] != units[:-1]) | (stamps[1:] != stamps[:-1])
    peaks = {}
    for unit in np.unique(units):
        seen = channel[units == unit]
        peaks[int(unit)] = int(np.argmax(np.bincount(seen)))
    return units[first], stamps[first], peaks


def snapshot(uri: str, *, revision: str | None = None, **kw) -> str:
    """Download ``hf://<owner>/<repo>/<config>`` and return the local config directory."""
    from huggingface_hub import snapshot_download

    if not uri.startswith("hf://"):
        raise ArrowError(f"{uri!r} is not an hf:// uri")
    parts = [p for p in uri[len("hf://") :].split("/") if p]
    if len(parts) < 2:
        raise ArrowError(f"{uri!r} must be hf://<owner>/<repo>[/<config>]")
    repo_id, config = "/".join(parts[:2]), "/".join(parts[2:])
    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=[f"{config}/**"] if config else None,
        revision=revision,
        **kw,
    )
    return os.path.join(local, config) if config else local


def _nested(rows: Sequence[np.ndarray], value_type):
    pa = _pa()
    lengths = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    flat = np.concatenate([np.asarray(r) for r in rows]) if len(rows) else np.zeros(0)
    inner = pa.ListArray.from_arrays(
        pa.array(offsets, pa.int32()), pa.array(flat, value_type)
    )
    return pa.ListArray.from_arrays(pa.array([0, len(rows)], pa.int32()), inner)


def _flat_list(values, value_type):
    pa = _pa()
    values = np.asarray(values)
    return pa.ListArray.from_arrays(
        pa.array([0, len(values)], pa.int32()), pa.array(values, value_type)
    )


def store(
    run: Run | None,
    path: str | os.PathLike,
    *,
    env=None,
    io: IO | None = None,
    split: str = "train",
    sample_id: str | None = None,
    stimulus: Policy | None = None,
    inputs: np.ndarray | None = None,
    inputs_dt: float | None = None,
    ground_truth: Iterable[str] = (),
    channel_spikes: tuple[Mapping, Mapping] | None = None,
    bad_channels: Sequence[int] | None = None,
    metadata: Mapping | None = None,
    templates: Mapping | None = None,
    dataset_stats: Mapping | None = None,
) -> str | None:
    """Append ``run`` as one row (a segment) of ``split`` under ``path``.

    The first call creates the dataset; later calls add a shard and keep every column of the
    first one. Under MPI the caller gathers first (``run.gather(env.comm)``); this writes on
    the root rank and is a no-op elsewhere.

    Channel space (``cp``, ``cit``/``ctt``) comes from the run's ``potential`` (or, given an
    ``env``, the projection of its membrane currents) and from the spikes assigned to
    channels by ``env.channel_recording`` (or ``channel_spikes=(cit, ctt)`` dicts by channel
    id). Cell space is whatever ``ground_truth`` names of ``"spikes"``, ``"voltage"``,
    ``"current"``.

    Args:
        run: The run, or ``None`` (off the root rank after a gather)
        path: The dataset root
        env: Supplies the IO, the channel assignment of spikes and, without a potential,
            the projection of membrane currents
        io: The recording IO when there is no env; stored beside the splits as ``io.json``
        split: The split to add the row to
        sample_id: The row's name; defaults to its index in the split
        stimulus, inputs, inputs_dt: As for :func:`livn.data.rsf.store`; stored as ``in``
        ground_truth: Cell-space channels to write
        channel_spikes: Per-channel ``(unit ids, times)`` when no env can compute them
        bad_channels: Channel ids to record as unusable
        metadata: Extra keys for ``conversion_metadata.json`` (first write only)
        templates: Arrays for ``templates.npz`` (first write only)
        dataset_stats: Normalisation stats for ``dataset_stats.json`` (first write only)

    Returns:
        The dataset root on the root rank, ``None`` elsewhere
    """
    from livn.utils import P

    if run is None or not P.is_root(comm=getattr(env, "comm", None)):
        return None
    datasets = _datasets()
    pa = _pa()

    root = os.fspath(path)
    io = io if io is not None else getattr(env, "io", None)
    ground_truth = tuple(ground_truth)
    unknown = sorted(set(ground_truth) - set(_GROUND_TRUTH))
    if unknown:
        raise ArrowError(
            f"ground_truth names {unknown}; choose from {sorted(_GROUND_TRUTH)}"
        )

    columns: dict[str, Any] = {}
    fs = None
    given = None
    if stimulus is not None or inputs is not None:
        from livn.data import rsf

        _, given = rsf._stimulus(stimulus, inputs, inputs_dt)

    potential = run.potential
    if potential is None and env is not None and run.current is not None:
        from livn.data import rsf
        from livn.run import Series

        dt = float(run.current_dt)
        try:
            values, _ = rsf._project(run, env, io, dt, delivered=given)
        except rsf.RSFError as e:
            raise ArrowError(str(e)) from e
        operator = getattr(io, "outputs", io)
        potential = Series(
            ids=np.asarray(operator.electrode_coordinates)[:, 0].astype(np.int64),
            values=values,
            dt=dt,
            t0=run.t0,
        )
    ids = None
    n_frames = None
    if potential is not None:
        values = np.asarray(potential.values, dtype=np.float32)
        ids = np.asarray(potential.ids).astype(np.int64)
        fs = 1000.0 / float(potential.dt)
        n_frames = values.shape[1]
        columns["cp"] = _nested(list(values), pa.float32())

    if run.spikes is not None and ids is not None:
        if channel_spikes is None:
            if env is None:
                raise ArrowError(
                    "a run with spikes needs the env to assign them to channels "
                    "(env.channel_recording), or channel_spikes=(cit, ctt)"
                )
            channel_spikes = env.channel_recording(run.spike_ids, run.spike_times)
        cit, ctt = channel_spikes
        columns["cit"] = _nested(
            [np.asarray(cit.get(int(c), []), dtype=np.int32) for c in ids], pa.int32()
        )
        columns["ctt"] = _nested(
            [np.asarray(ctt.get(int(c), []), dtype=np.float64) for c in ids],
            pa.float64(),
        )

    if given is not None:
        from livn.data import rsf

        if io is None:
            raise ArrowError("inputs need the IO whose channels they drive")
        if fs is None:
            raise ArrowError(
                "inputs are on the frame grid of a channel signal; the run has none"
            )
        delivered = rsf._delivered(io, given, 1000.0 / fs, n_frames)
        columns["in"] = _nested(list(delivered["values"].T), pa.float32())

    columns |= _cell_columns(run, ground_truth)
    if fs is None:
        for dt_key in ("voltage_dt", "current_dt"):
            if dt_key in columns:
                fs = 1000.0 / float(columns[dt_key][0].as_py())
                break
    if not columns:
        raise ArrowError(
            "the run has nothing to write: no channel potential and no ground_truth requested"
        )

    split_dir = os.path.join(root, split)
    index = 0
    if os.path.isfile(os.path.join(split_dir, "state.json")):
        index = len(datasets.load_from_disk(split_dir))
    name = f"{index:010d}" if sample_id is None else str(sample_id)
    table = pa.table({"sample_id": pa.array([name]), **columns})

    _check_side_files(root, fs)
    _append(root, split, table)

    _write_side_files(
        root,
        fs=fs,
        ids=ids,
        io=io,
        bad_channels=bad_channels,
        metadata=metadata,
        templates=templates,
        dataset_stats=dataset_stats,
    )
    _register_split(root, split)
    return root


def _cell_columns(run: Run, names: Iterable[str]) -> dict[str, Any]:
    pa = _pa()
    columns: dict[str, Any] = {}
    for name in names:
        if name == "spikes":
            if run.spikes is None:
                raise ArrowError("ground_truth asks for spikes and the run has none")
            columns["it"] = _flat_list(np.asarray(run.spike_ids), pa.int32())
            columns["tt"] = _flat_list(np.asarray(run.spike_times), pa.float64())
            continue
        channel = run.channel("voltage" if name == "voltage" else "current")
        if channel is None or channel.values is None:
            raise ArrowError(f"ground_truth asks for {name} and the run has none")
        ids_c, values_c, sections_c, dt_c = _GROUND_TRUTH[name]
        values = np.asarray(channel.values, dtype=np.float32)
        columns[ids_c] = _flat_list(np.asarray(channel.ids), pa.int32())
        columns[values_c] = _nested(list(values), pa.float32())
        if channel.sections is not None:
            columns[sections_c] = pa.ListArray.from_arrays(
                pa.array([0, len(channel.sections)], pa.int32()),
                pa.array([str(s) for s in np.asarray(channel.sections)]),
            )
        columns[dt_c] = pa.array([float(channel.dt)], pa.float64())
    return columns


def _check_side_files(root: str, fs) -> None:
    path = os.path.join(root, CONVERSION)
    if fs is None or not os.path.isfile(path):
        return
    have = _json(path).get("sampling_frequency")
    if have is not None and abs(float(have) - float(fs)) > 1e-6:
        raise ArrowError(
            f"{root} was written at {have} Hz and this run is sampled at {fs} Hz"
        )


def _append(root: str, split: str, table) -> None:
    """Add ``table`` as a shard of ``root/split`` without rewriting the shards before it."""
    datasets = _datasets()

    split_dir = os.path.join(root, split)
    os.makedirs(root, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix=".shard-", dir=root)
    try:
        was_disabled = datasets.are_progress_bars_disabled()
        datasets.disable_progress_bars()
        try:
            datasets.Dataset(table).save_to_disk(scratch)
        finally:
            if not was_disabled:
                datasets.enable_progress_bars()
        state = _json(os.path.join(scratch, "state.json"))
        info = _json(os.path.join(scratch, "dataset_info.json"))
        (made,) = (f["filename"] for f in state["_data_files"])

        if not os.path.isfile(os.path.join(split_dir, "state.json")):
            os.makedirs(split_dir, exist_ok=True)
            shard = "data-00000.arrow"
            shutil.move(os.path.join(scratch, made), os.path.join(split_dir, shard))
            state["_data_files"] = [{"filename": shard}]
        else:
            have_info = _json(os.path.join(split_dir, "dataset_info.json"))
            if have_info["features"] != info["features"]:
                raise ArrowError(
                    f"{split_dir} has columns {sorted(have_info['features'])} and this "
                    f"row has {sorted(info['features'])}; every row of a split carries "
                    "the same ones"
                )
            state = _json(os.path.join(split_dir, "state.json"))
            shard = f"data-{len(state['_data_files']):05d}.arrow"
            shutil.move(os.path.join(scratch, made), os.path.join(split_dir, shard))
            state["_data_files"].append({"filename": shard})
            info = have_info
        _dump(os.path.join(split_dir, "dataset_info.json"), info)
        _dump(os.path.join(split_dir, "state.json"), state)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _dump(path: str, document: Mapping) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, default=_default)
        f.write("\n")


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def _register_split(root: str, split: str) -> None:
    path = os.path.join(root, DATASET_DICT)
    splits = _json(path)["splits"] if os.path.isfile(path) else []
    splits = list(splits)
    if split not in splits:
        splits.append(split)
        _dump(path, {"splits": splits})


def _write_side_files(
    root, *, fs, ids, io, bad_channels, metadata, templates, dataset_stats
) -> None:
    path = os.path.join(root, CONVERSION)
    if not os.path.isfile(path):
        document: dict[str, Any] = {}
        if fs is not None:
            document["sampling_frequency"] = float(fs)
        if ids is not None:
            document["num_channels"] = len(ids)
            document["channel_ids"] = [int(i) for i in ids]
        if bad_channels is not None:
            document["bad_channel_ids"] = [int(c) for c in bad_channels]
        document |= dict(metadata or {})
        _dump(path, document)

    if io is not None and not os.path.isfile(os.path.join(root, IO_DOCUMENT)):
        from types import SimpleNamespace

        from livn.data import rsf

        device_io, operator, stimulates = rsf._observation(SimpleNamespace(io=io))
        _dump(
            os.path.join(root, IO_DOCUMENT),
            rsf._observe_document(device_io, operator, stimulates, set()),
        )
    if templates is not None and not os.path.isfile(os.path.join(root, TEMPLATES)):
        np.savez(os.path.join(root, TEMPLATES), **dict(templates))
    if dataset_stats is not None and not os.path.isfile(os.path.join(root, STATS)):
        _dump(os.path.join(root, STATS), dataset_stats)
