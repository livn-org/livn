from __future__ import annotations

import dataclasses
import inspect
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from livn.conditions import Conditions, _register_pending
from livn.io.transfer import Bandpass
from livn.utils import canonical_hash

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float, Int

__all__ = [
    "CMR",
    "LIVE",
    "Standardize",
    "Stats",
    "Window",
    "Windows",
    "windows",
]


class Window(Conditions):
    context: Float[Array, "*batch timestep n_channels"] | None = None
    prediction: Float[Array, "*batch timestep n_channels"] | None = None
    spikes: Float[Array, "*batch n_units max_spikes"] | None = None
    unit_mask: Bool[Array, "*batch n_units"] | None = None
    channel_mask: Bool[Array, "*batch n_channels"] | None = None
    recording: Int[Array, " *batch"] | None = None


LIVE = {"original": True, "space": "channel", "role": "acquisition"}


@dataclasses.dataclass(frozen=True)
class _Store:
    times: np.ndarray
    ids: np.ndarray
    units: np.ndarray
    cover: np.ndarray | None
    space: str | None


@dataclasses.dataclass(frozen=True)
class CMR:
    """Common median reference: subtract the per-sample median over the good channels"""

    def __call__(self, x: np.ndarray, fs: float, good: np.ndarray) -> np.ndarray:
        if not good.any():
            return x
        return x - np.median(x[:, good], axis=1, keepdims=True)

    def margin(self, fs: float) -> int:
        return 0

    def fingerprint(self) -> str:
        return "cmr"


def _preprocessed(transform, x: np.ndarray, fs: float, good: np.ndarray) -> np.ndarray:
    if isinstance(transform, Bandpass):
        return transform.filter(x, fs, axis=0)
    return transform(x, fs, good)


def _fingerprint_of(transform: Callable) -> str:
    fingerprint = getattr(transform, "fingerprint", None)
    if fingerprint is not None:
        return fingerprint()
    return getattr(transform, "__name__", type(transform).__name__)


@dataclasses.dataclass(frozen=True)
class Standardize:
    """Robust per-channel standardisation: ``(x - median) / scale`` with ``scale = (p99 - p1) / 2``.

    Args:
        stats: ``"fit"`` fits them from the data (see below), a dict
            ``{"median": ..., "scale": ...}`` (per channel or scalar) is used as given, and a
            path reads such a JSON file.
        cache: Directory for fitted stats. It must not be inside a dataset directory. A cached
            file is used only when its fingerprint and sampling match. Stats computed elsewhere
            are passed as ``stats=`` (a dict or a path); nothing is read from a dataset
            implicitly.
        n_windows: How many windows the fit samples. They are drawn deterministically from the
            pipeline's seed, after decimation and preprocessing.
    """

    stats: str | dict | os.PathLike = "fit"
    cache: str | os.PathLike | None = None
    n_windows: int = 64


@dataclasses.dataclass(frozen=True)
class Stats:
    """The normalisation a pipeline applies, with the fingerprint of what produced the signal.

    ``median`` and ``scale`` are ``None`` when the pipeline does not normalise. ``source`` is
    ``"none"``, ``"given"``, ``"cache"`` or ``"fit"``.
    """

    fingerprint: str
    median: np.ndarray | None = None
    scale: np.ndarray | None = None
    source: str = "none"

    def to_json(self) -> dict:
        return {
            "preprocess_fingerprint": self.fingerprint,
            "median": None if self.median is None else self.median.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }


def _stats_from(document: dict, channels: int, fingerprint: str, source: str) -> Stats:
    block = document.get("per_channel") or document.get("global", document)
    median = np.broadcast_to(
        np.asarray(block.get("median", 0.0), np.float32), (channels,)
    ).copy()
    scale = np.broadcast_to(np.asarray(block["scale"], np.float32), (channels,)).copy()
    return Stats(fingerprint, median, scale, source)


def _barrier(name: str) -> None:
    import jax

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


class _Index:
    """The index space: element ``i`` is a ``(recording, segment, slot)``"""

    def __init__(self, entries: list[tuple[int, int, int]]):
        self.entries = np.asarray(entries, dtype=np.int64).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        r, s, k = self.entries[index]
        return {"recording": np.int64(r), "segment": np.int64(s), "slot": np.int64(k)}


class _Windower:
    """Everything one window needs; picklable, reopens lazily in worker processes"""

    def __init__(
        self,
        recordings,
        *,
        q,
        context,
        gap,
        prediction,
        starts,
        per_segment,
        preprocess,
        mask_bad,
        spikes,
        inputs,
        max_spikes,
        unit_ids,
        policy,
        preprocess_at="native",
        sortings=None,
    ):
        self.recordings = recordings
        self.q = q
        self.context, self.gap, self.prediction = context, gap, prediction
        self.starts, self.per_segment = starts, per_segment
        self.preprocess = tuple(preprocess)
        self.mask_bad = mask_bad
        self.spikes, self.inputs = spikes, inputs
        self.max_spikes = max_spikes
        self.unit_ids = unit_ids
        self.channel_units = False
        self.policy = policy
        self.preprocess_at = preprocess_at
        self.sortings = [None] * len(recordings) if sortings is None else list(sortings)
        self.stats: Stats | None = None
        self._spikes: dict[tuple[int, int], _Store] = {}
        self._rows: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._starts: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._good = [
            ~np.isin(np.asarray(r.channel_ids), np.asarray(r.bad_channels))
            for r in recordings
        ]
        fs = recordings[0].fs

        self.pad = (
            0
            if preprocess_at == "target"
            else sum(int(t.margin(fs)) for t in self.preprocess if hasattr(t, "margin"))
        )
        self.pad = (
            -(-self.pad // q) * q + 11 * q
        )  # decimation kernel half-length, in whole steps

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_spikes"] = {}
        state["_rows"] = {}
        state["_starts"] = {}
        return state

    @property
    def total(self) -> int:
        """Target samples in a window: context, gap and prediction"""
        return self.context + self.gap + self.prediction

    def spans(self, r, segment) -> np.ndarray:
        spans = np.asarray(self.recordings[r].valid_spans(segment), np.int64)
        spans = spans.reshape(-1, 2)
        if self.spikes:
            cover = self.store(r, segment).cover
            if cover is not None:
                spans = _intersect(spans, cover)
        return spans

    def start_spans(self, r, segment) -> tuple[np.ndarray, np.ndarray]:
        """``(first, count)`` of the window starts (target-sample units)."""
        key = (r, segment)
        if key not in self._starts:
            spans = self.spans(r, segment)
            first = -(-spans[:, 0] // self.q)
            count = np.maximum(
                (spans[:, 1] - self.total * self.q) // self.q - first + 1, 0
            )
            self._starts[key] = (first[count > 0], count[count > 0])
        return self._starts[key]

    def start_range(self, r, segment) -> int:
        """Number of valid window starts in the segment"""
        return int(self.start_spans(r, segment)[1].sum())

    def nth_start(self, r, segment, i: int) -> int:
        """The ``i``-th valid window start of the segment, in target samples"""
        first, count = self.start_spans(r, segment)
        ends = np.cumsum(count)
        j = int(np.searchsorted(ends, i, side="right"))
        return int(first[j] + i - (ends[j] - count[j]))

    def item(self, element: dict, rng: np.random.Generator) -> dict:
        r, s, k = (
            int(element["recording"]),
            int(element["segment"]),
            int(element["slot"]),
        )
        rec = self.recordings[r]
        n = self.start_range(r, s)
        if self.starts == "random":
            index = int(rng.integers(0, n))
        elif self.starts == "grid":
            index = (
                0
                if self.per_segment == 1
                else round(k * (n - 1) / (self.per_segment - 1))
            )
        elif self.starts == "head":
            index = 0
        else:
            index = n - 1
        start = self.nth_start(r, s, index)
        item = {
            "recording": r,
            "segment": s,
            "slot": k,
            "start": start * self.q,
            "length": self.total * self.q,
        }
        if self.policy is not None:
            item = dict(self.policy(item, rng, rec))
            self._check(item)
        return item

    def _check(self, item) -> None:
        start = item["start"]
        spans = self.spans(item["recording"], item["segment"])
        inside = (spans[:, 0] <= start) & (start + item["length"] <= spans[:, 1])
        if start % self.q or not inside.any():
            covered = " (and the coverage of its spikes' store)" if self.spikes else ""
            raise ValueError(
                f"window_policy returned start {start}, outside the valid spans{covered} of "
                f"segment {item['segment']} or off the decimation grid (every {self.q} "
                "samples)"
            )

    def signal(self, item) -> np.ndarray:
        """float32 ``(total, channels)``"""
        from scipy import signal

        rec = self.recordings[item["recording"]]
        seg, start, length = item["segment"], item["start"], item["length"]
        have = rec.num_samples(seg)
        lo, hi = start - self.pad, start + length + self.pad
        x = rec.read(seg, max(lo, 0), min(hi, have) - max(lo, 0)).astype(np.float64)
        if lo < 0 or hi > have:
            x = np.pad(x, ((max(-lo, 0), max(hi - have, 0)), (0, 0)), mode="edge")
        good = self._good[item["recording"]]
        if not self.mask_bad:
            # unmasked, a bad channel is a channel like any other
            good = np.ones_like(good)
        if self.preprocess_at == "native":
            for transform in self.preprocess:
                x = _preprocessed(transform, x, rec.fs, good)
        if self.q > 1:
            x = signal.resample_poly(x, 1, self.q, axis=0)
        crop = self.pad // self.q
        x = x[crop : crop + self.total]
        if self.preprocess_at == "target":
            for transform in self.preprocess:
                x = _preprocessed(transform, x, rec.fs / self.q, good)
        return np.ascontiguousarray(x, np.float32)

    def store(self, r, s) -> _Store:
        if (r, s) not in self._spikes:
            rec = self.recordings[r]
            name = self.sortings[r]
            ev = rec.spikes(s) if name is None else rec.spikes(s, sorting=name)
            ids = np.asarray(ev.ids)
            units = getattr(ev, "units", None)
            units = np.unique(ids) if units is None else np.asarray(units)
            coverage = getattr(ev, "coverage", None)
            cover = None
            if coverage is not None:
                ms = np.asarray(coverage, np.float64).reshape(-1, 2)
                cover = np.round(ms * rec.fs / 1e3).astype(np.int64)
                extraction = getattr(rec, "extraction", None)
                state = None if extraction is None or name is None else extraction(name)
                after = (state or {}).get("state_from_frame")
                if after is not None:
                    cover = _intersect(
                        cover, np.asarray([[int(after), np.iinfo(np.int64).max]])
                    )
            self._spikes[(r, s)] = _Store(
                times=np.asarray(ev.times, np.float64),
                ids=ids,
                units=units,
                cover=cover,
                space=getattr(ev, "space", None),
            )
        return self._spikes[(r, s)]

    def rows(self, r, s) -> tuple[np.ndarray, np.ndarray]:
        if (r, s) not in self._rows:
            store = self.store(r, s)
            if self.channel_units:
                column = {
                    str(c): i for i, c in enumerate(self.recordings[r].channel_ids)
                }
                unknown = sorted({str(u) for u in store.units} - set(column))
                if unknown:
                    raise ValueError(
                        f"the sorting of recording {r} has units {unknown[:5]} that are "
                        "not its channels"
                    )
                rows = np.asarray([column[str(i)] for i in store.ids], np.int64)
                mask = np.zeros(len(self.unit_ids), bool)
                mask[[column[str(u)] for u in store.units]] = True
            else:
                strings = np.asarray(self.unit_ids).dtype.kind not in "iu"
                keys = _unit_keys(self.unit_ids, strings)
                rows = np.searchsorted(keys, _unit_keys(store.ids, strings))
                mask = np.isin(keys, _unit_keys(store.units, strings))
            self._rows[(r, s)] = (rows.reshape(-1), mask)
        return self._rows[(r, s)]

    def events(self, item) -> np.ndarray:
        r, s = item["recording"], item["segment"]
        times = self.store(r, s).times
        rows_of, _ = self.rows(r, s)
        fs = self.recordings[r].fs
        t0 = item["start"] / fs * 1e3
        t1 = t0 + item["length"] / fs * 1e3
        keep = (times >= t0) & (times < t1)
        out = np.full((len(self.unit_ids), self.max_spikes), np.inf, np.float32)
        rows = rows_of[keep]
        kept = times[keep]
        for row in np.unique(rows):
            mine = np.sort(kept[rows == row])[: self.max_spikes] - t0
            out[row, : len(mine)] = mine
        return out

    def draw(self, element: dict, rng: np.random.Generator) -> Window:
        item = self.item(element, rng)
        x = self.signal(item)
        rec = self.recordings[item["recording"]]
        good = self._good[item["recording"]]
        if self.stats is not None and self.stats.median is not None:
            x = (x - self.stats.median) / self.stats.scale
        if self.mask_bad:
            x = np.where(good, x, 0.0).astype(np.float32)
        out: dict[str, Any] = {"context": x[: self.context]}
        if self.prediction:
            out["prediction"] = x[self.context + self.gap :]
        if self.spikes:
            out["spikes"] = self.events(item)
            out["unit_mask"] = self.rows(item["recording"], item["segment"])[1]
        if self.inputs:
            frames = _stimulus_at_sample(
                rec, item["segment"], item["start"], item["length"]
            )
            out["inputs"] = np.ascontiguousarray(
                frames[:: self.q][: self.total], np.float32
            )
        out["channel_mask"] = good if self.mask_bad else np.ones_like(good)
        out["recording"] = np.int32(item["recording"])
        out["segment"] = np.int32(item["segment"])
        out["start"] = np.int64(item["start"])
        return Window(**out)


def _intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = [
        (max(a0, b0), min(a1, b1))
        for a0, a1 in np.asarray(a, np.int64).reshape(-1, 2).tolist()
        for b0, b1 in np.asarray(b, np.int64).reshape(-1, 2).tolist()
        if max(a0, b0) < min(a1, b1)
    ]
    return np.asarray(sorted(out), np.int64).reshape(-1, 2)


def _unit_keys(ids, strings: bool = False) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.dtype.kind in "iu" and not strings:
        return ids.astype(np.int64)
    return np.asarray([str(i) for i in ids.tolist()], dtype=str)


def _resolve_sorting(rec, r: int, sorting) -> str | None:
    if sorting is None:
        return None
    if callable(sorting) and not isinstance(sorting, (str, Mapping)):
        chosen = sorting(rec)
        if chosen is not None and not isinstance(chosen, str):
            chosen = getattr(chosen, "path", None)
            if chosen is None:
                raise TypeError(
                    "a sorting selector returns a store path, a Sorting or None"
                )
        return chosen
    listed = getattr(rec, "sortings", None)
    if listed is None:
        raise ValueError(
            f"recording {r} ({type(rec).__name__}) has no sorting stores; sorting= needs "
            "an RSF session (or a callable that returns None for the ground truth)"
        )
    if isinstance(sorting, str):
        return sorting
    if not isinstance(sorting, Mapping):
        raise TypeError(
            "sorting= is None (the ground truth), a store path or name, a mapping of "
            f"Sorting fields, or a callable, not {sorting!r}"
        )
    stores = listed()
    matched = [
        st
        for st in stores
        if all(getattr(st, key, _NOTHING) == value for key, value in sorting.items())
    ]
    if len(matched) != 1:
        which = "several" if matched else "none"
        paths = [getattr(st, "path", st) for st in matched or stores]
        raise ValueError(
            f"the sorting selector {dict(sorting)} matches {which} of the stores of "
            f"recording {r} ({getattr(rec, 'path', '')}): {paths}; name one by its path"
        )
    return matched[0].path


_NOTHING = object()


def _stimulus_at_sample(rec, segment, start, length) -> np.ndarray:
    try:
        takes_latency = "latency" in inspect.signature(rec.inputs).parameters
    except (TypeError, ValueError):
        takes_latency = False
    if takes_latency:
        return rec.inputs(segment, start, length, latency=True)
    return rec.inputs(segment, start, length)


class Windows:
    def __init__(
        self, dataset, windower, batch_size, threads, processes, unit_ids, epoch_steps
    ):
        self._dataset = dataset
        self._windower = windower
        self.batch_size = batch_size
        self.threads = threads
        self.processes = processes
        self.unit_ids = unit_ids
        self.epoch_steps = epoch_steps

    @property
    def stats(self) -> Stats:
        """The normalisation applied (``median`` / ``scale`` ``None`` when there is none)"""
        return self._windower.stats

    def batch(self, step: int) -> Window:
        """The full global batch of ``step``"""
        return self._dataset.batch(self.batch_size)[int(step)]

    def iterate(self, start_step: int = 0, process=None, processes=None):
        """This process's slice of the global batches of ``start_step``, ``start_step + 1``, ...

        ``process`` / ``processes`` default to the jax process index and count. The stream is
        infinite; the global batch size must be divisible by ``processes``.
        """
        import grain

        if process is None or processes is None:
            import jax

            process = jax.process_index() if process is None else process
            processes = jax.process_count() if processes is None else processes
        if self.batch_size % processes:
            raise ValueError(
                f"batch_size {self.batch_size} is not divisible by {processes} processes"
            )
        if not 0 <= process < processes:
            raise ValueError(f"process {process} is not in 0..{processes - 1}")
        blocks = self._dataset.batch(self.batch_size // processes)
        mine = blocks[process + int(start_step) * processes :: processes]
        stream = mine.to_iter_dataset(
            grain.ReadOptions(
                num_threads=self.threads, prefetch_buffer_size=max(2, self.threads)
            )
        )
        if self.processes:
            from absl import flags

            if not flags.FLAGS.is_parsed():
                flags.FLAGS.mark_as_parsed()
            stream = stream.mp_prefetch(
                grain.MultiprocessingOptions(num_workers=self.processes)
            )
        return iter(stream)


def _select(recordings, split) -> list[tuple[int, int]]:
    chosen = []
    for r, rec in enumerate(recordings):
        own = rec.metadata.get("split")
        if split is not None and own is not None and own != split:
            continue
        per_segment = rec.metadata.get("segment_split")
        for s in range(rec.segments):
            if (
                split is not None
                and per_segment is not None
                and per_segment[s] != split
            ):
                continue
            chosen.append((r, s))
    if not chosen:
        raise ValueError(
            f"no segments are in split {split!r}; the recordings are in splits "
            f"{[rec.metadata.get('split') for rec in recordings]} (pass split=None for all)"
        )
    return chosen


def _fit_stats(
    windower: _Windower, entries, seed, n_windows, channels, fingerprint
) -> Stats:
    """Median and ``(p99 - p1) / 2`` per channel over a fixed, seed-determined sample"""
    rng = np.random.default_rng([seed, 0x57A7])
    picks = rng.integers(0, len(entries), size=n_windows)
    parts = []
    for n, pick in enumerate(picks):
        r, s, k = (int(v) for v in entries[pick])
        item = windower.item(
            {"recording": r, "segment": s, "slot": k},
            np.random.default_rng([seed, 0x57A7, n]),
        )
        parts.append(windower.signal(item))
    x = np.concatenate(parts, axis=0).astype(np.float64)
    median = np.median(x, axis=0)
    low, high = np.percentile(x, [1, 99], axis=0)
    scale = (high - low) / 2
    scale = np.where(scale > 0, scale, 1.0)
    return Stats(
        fingerprint, median.astype(np.float32), scale.astype(np.float32), "fit"
    )


def _identity(recordings, selected) -> list:
    return [
        [
            str(getattr(recordings[r], "path", "")),
            r,
            s,
            int(recordings[r].num_samples(s)),
        ]
        for r, s in selected
    ]


def _resolve_stats(
    normalize: Standardize | None,
    windower: _Windower,
    recordings,
    selected,
    entries,
    seed,
    fingerprint,
) -> Stats:
    if normalize is None:
        return Stats(fingerprint)
    channels = len(recordings[0].channel_ids)
    spec = normalize.stats
    if isinstance(spec, dict):
        return _stats_from(spec, channels, fingerprint, "given")
    if not isinstance(spec, str) or spec != "fit":
        document = json.loads(Path(spec).read_text())
        return _stats_from(document, channels, fingerprint, "given")

    path = None
    if normalize.cache is not None:
        cache = Path(normalize.cache).resolve()
        for rec in recordings:
            base = getattr(rec, "path", None)
            if base is not None and cache.is_relative_to(Path(str(base)).resolve()):
                raise ValueError(
                    f"stats cache {cache} is inside the dataset {base}; "
                    "cache under the run directory instead"
                )
        key = canonical_hash(
            [
                fingerprint,
                seed,
                normalize.n_windows,
                windower.total,
                windower.starts,
                windower.per_segment,
                _identity(recordings, selected),
            ]
            + ([windower.sortings] if any(windower.sortings) else [])
        )
        path = cache / f"stats-{key[:16]}.json"

    def read() -> Stats | None:
        if path is None or not path.exists():
            return None
        document = json.loads(path.read_text())
        if document.get("preprocess_fingerprint") != fingerprint:
            return None
        return _stats_from(document, channels, fingerprint, "cache")

    import jax

    if path is not None and jax.process_count() > 1:
        if jax.process_index() == 0 and read() is None:
            _write(
                path,
                _fit_stats(
                    windower, entries, seed, normalize.n_windows, channels, fingerprint
                ),
            )
        _barrier("livn.optimization.data.stats")
        found = read()
        if found is None:
            raise FileNotFoundError(f"process 0 wrote no stats at {path}")
        return found

    found = read()
    if found is not None:
        return found
    fitted = _fit_stats(
        windower, entries, seed, normalize.n_windows, channels, fingerprint
    )
    if path is not None:
        _write(path, fitted)
    return fitted


def _write(path: Path, stats: Stats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(stats.to_json()))
    os.replace(tmp, path)


def windows(
    recording,
    *,
    context: float,
    gap: float = 0.0,
    prediction: float = 0.0,
    fs: float | None = None,
    preprocess: Sequence[Callable] = (),
    normalize: Standardize | None = None,
    starts: str = "random",
    per_segment: int = 10,
    split: str | None = "train",
    spikes: bool = False,
    sorting: str | Mapping | Callable | None = None,
    inputs: bool = False,
    max_spikes: int = 64,
    mask_bad_channels: bool = True,
    batch_size: int,
    seed: int = 0,
    threads: int = 8,
    processes: int = 0,
    window_policy: Callable | None = None,
    preprocess_at: str = "native",
) -> Windows:
    """Windows of one or several recordings as step-addressed batches, :class:`Windows`.

    A window is ``context``, then ``gap`` and ``prediction`` of signal, read at the
    recording's rate, preprocessed there, decimated to ``fs`` (zero-phase FIR, with the read
    padded at both ends so the kernel never sees an edge) and normalised.

    Args:
        recording: A :class:`livn.data.Recording` or a list of them (same channels and rate)
        context, gap, prediction: Window layout; ``Window.prediction`` is set only when
            ``prediction > 0``
        fs: Target rate in Hz; the recording's rate must be an integer multiple
        preprocess: Callables ``f(x[T, C] float64, fs, good[C]) -> x`` applied in order at the
            native rate, e.g. ``(Bandpass(300, 3000, 3, phase="zero"), CMR())`` with
            :class:`livn.io.Bandpass`. ``margin(fs)`` and
            ``fingerprint()`` methods, when present, set the read padding and the stats
            fingerprint.
        normalize: A :class:`Standardize`, or ``None`` to leave the scale alone. Stats are
            fingerprinted by the decimation factor, the target rate, the preprocessing chain and
            ``preprocess_at``, as ``decim=<q>;fs=<hz>[;pre=<a>+<b>][;at=target]`` (e.g.
            ``decim=4;fs=2500;pre=bp(300,3000,3)+cmr;at=target``), so a cached fit is reused only
            for the same signal.
        starts: ``"random"`` draws every window's start from ``(seed, index)`` (fresh each
            epoch, order shuffled); ``"grid"`` spreads ``per_segment`` starts evenly over each
            segment; ``"head"`` / ``"tail"`` take the first / last window of each segment (one
            slot). The deterministic modes are not shuffled, so ``Windows.epoch_steps``
            steps are one pass in order.
        per_segment: Window slots per segment (the epoch length is segments x slots)
        split: Which segments to use: recordings whose ``metadata["split"]`` differs are
            skipped, and ``metadata["segment_split"]`` (one name per segment) selects segments.
            ``None`` takes everything.
        spikes: Set ``Window.spikes``, ``(U, max_spikes)`` per unit of the recordings' spikes
            (their ground truth unless ``sorting`` is used).
        sorting: Where ``Window.spikes`` come from when spikes=True:
            - None for the ground truth
            - a store path or name (``"sortings/live.zarr"``, ``"live"``)
            - a mapping of :class:`livn.data.rsf.Sorting` fields to match
            - a callable ``f(recording) -> path | Sorting | None``.
        inputs: Set ``Window.inputs``, the delivered stimulus (``Recording.inputs``, strided
            to the target rate, not filtered) on the window's grid, as it reaches the sample:
            a channel whose calibration states a ``latency_ms`` has its edges later by it.
            Raises :class:`livn.data.Missing` here if a recording has none.
        max_spikes: Capacity of a unit's row in a window
        mask_bad_channels: Zero the recording's bad channels and exclude them from ``CMR``;
            ``channel_mask`` says which are good
        batch_size: The GLOBAL batch size; a process takes ``batch_size / processes``
        seed: Everything is a function of this and the global sample index
        threads: Reader threads (grain ``ReadOptions``); numpy and scipy release the GIL
        processes: Worker processes (grain ``mp_prefetch``); 0 keeps the work in threads. The
            recordings must pickle.
        window_policy: ``f(item, rng, recording) -> item``, called after the default start is
            drawn and before reading. ``item`` is a dict with ``recording``, ``segment``,
            ``slot``, ``start`` and ``length`` (native samples); the policy may return other
            ``segment`` / ``start`` values, e.g. to redraw until the window holds a spike.
            It must be a pure function of its arguments to keep windows reproducible; ``rng``
            is the window's own generator.
        preprocess_at: ``"native"`` runs ``preprocess`` on the padded read before decimation (the
            filter never sees a window edge); ``"target"`` decimates first and filters the
            cropped window at ``fs`` (the filter's edge transients are part of the window, and
            the chain's sampling rate argument is ``fs``)
    """
    import grain
    import jax  # noqa: F401  (grain batches windows as jax pytrees)

    from livn.data import Recording

    _register_pending()

    recordings = [recording] if isinstance(recording, Recording) else list(recording)
    if not recordings:
        raise ValueError("no recordings")
    if starts not in ("random", "grid", "head", "tail"):
        raise ValueError(f"unknown starts {starts!r}")
    if preprocess_at not in ("native", "target"):
        raise ValueError(
            f"preprocess_at is 'native' or 'target', not {preprocess_at!r}"
        )
    first = recordings[0]
    for rec in recordings[1:]:
        if rec.fs != first.fs or len(rec.channel_ids) != len(first.channel_ids):
            raise ValueError(
                "recordings must share the sampling rate and channel count"
            )

    target = float(fs) if fs is not None else float(first.fs)
    q = round(first.fs / target)
    if q < 1 or abs(q * target - first.fs) > 1e-6 * first.fs:
        raise ValueError(
            f"fs={target:g} is not a divisor of the recording rate {first.fs:g}"
        )
    context_samples = round(context * target / 1e3)
    gap_samples = round(gap * target / 1e3)
    prediction_samples = round(prediction * target / 1e3)
    if context_samples < 1:
        raise ValueError(f"context={context} is under one sample at {target:g} Hz")
    if per_segment < 1:
        raise ValueError("per_segment must be at least 1")
    slots = 1 if starts in ("head", "tail") else per_segment

    selected = _select(recordings, split)
    entries = [(r, s, k) for r, s in selected for k in range(slots)]
    total = context_samples + gap_samples + prediction_samples

    if sorting is not None and not spikes:
        raise ValueError(
            "sorting= determines where Window.spikes come from, pass spikes=True"
        )
    chosen = {r for r, _ in selected}
    sortings = [
        _resolve_sorting(rec, r, sorting) if spikes and r in chosen else None
        for r, rec in enumerate(recordings)
    ]
    if inputs:
        r, s = selected[0]
        recordings[r].inputs(s, 0, 1)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    windower = _Windower(
        recordings,
        q=q,
        context=context_samples,
        gap=gap_samples,
        prediction=prediction_samples,
        starts=starts,
        per_segment=slots,
        preprocess=preprocess,
        mask_bad=mask_bad_channels,
        spikes=spikes,
        inputs=inputs,
        max_spikes=max_spikes,
        unit_ids=np.zeros(0, np.int64),
        policy=window_policy,
        preprocess_at=preprocess_at,
        sortings=sortings,
    )
    if spikes:
        stores = [windower.store(r, s) for r, s in selected]
        spaces = {st.space for st in stores}
        if spaces == {"channel"}:
            windower.channel_units = True
            windower.unit_ids = np.asarray(recordings[selected[0][0]].channel_ids)
        elif "channel" in spaces:
            raise ValueError(
                "the sortings mix channel-space stores (units are rows) with source-space "
                "ones (units are inferred neurons); window one kind at a time"
            )
        else:
            strings = any(np.asarray(st.units).dtype.kind not in "iu" for st in stores)
            windower.unit_ids = np.unique(
                np.concatenate(
                    [_unit_keys(st.units, strings) for st in stores]
                    + [_unit_keys(np.zeros(0, np.int64), strings)]
                )
            )
    for r, s in selected:
        if windower.start_range(r, s) < 1:
            covered = (
                ", inside the coverage of its spikes' store,"
                if spikes and windower.store(r, s).cover is not None
                else ""
            )
            raise ValueError(
                f"segment {s} of recording {r} has {recordings[r].num_samples(s)} samples, "
                f"and no valid span of them{covered} holds one window ({total * q})"
            )
    fingerprint = (
        f"decim={q};fs={target:g}"
        + (
            ";pre=" + "+".join(_fingerprint_of(t) for t in preprocess)
            if preprocess
            else ""
        )
        + (";at=target" if preprocess_at == "target" and preprocess else "")
    )
    windower.stats = _resolve_stats(
        normalize,
        windower,
        recordings,
        selected,
        np.asarray(entries),
        seed,
        fingerprint,
    )

    dataset = grain.MapDataset.source(_Index(entries)).seed(seed)
    if starts == "random":
        dataset = dataset.shuffle()
    dataset = dataset.repeat(None).random_map(windower.draw)

    return Windows(
        dataset,
        windower,
        batch_size,
        threads,
        processes,
        windower.unit_ids,
        len(entries) // batch_size,
    )
