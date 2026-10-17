from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CMR", "Bandpass", "Pipeline", "Standardize", "Stats", "windows"]


@dataclasses.dataclass(frozen=True)
class Bandpass:
    """Zero-phase Butterworth band-pass (``sosfiltfilt``) at the recording's native rate.

    ``hi=None`` (or ``hi`` at the Nyquist rate) is a high-pass.
    """

    lo: float
    hi: float | None = None
    order: int = 3

    def __call__(self, x: np.ndarray, fs: float, good: np.ndarray) -> np.ndarray:
        from scipy import signal

        nyquist = fs / 2
        if self.hi is None or self.hi >= nyquist:
            sos = signal.butter(self.order, self.lo, "highpass", fs=fs, output="sos")
        else:
            sos = signal.butter(
                self.order, [self.lo, self.hi], "bandpass", fs=fs, output="sos"
            )
        return signal.sosfiltfilt(sos, x, axis=0)

    def margin(self, fs: float) -> int:
        """Samples of read padding that keep the filter's edge transient out of the window"""
        return int(np.ceil(8 * fs / self.lo))

    def fingerprint(self) -> str:
        return f"bp({self.lo:g},{'' if self.hi is None else format(self.hi, 'g')},{self.order})"


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
        self.policy = policy
        self.preprocess_at = preprocess_at
        self.stats: Stats | None = None
        self._spikes: dict[tuple[int, int], tuple] = {}
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
        return state

    @property
    def total(self) -> int:
        """Target samples in a window: context, gap and prediction"""
        return self.context + self.gap + self.prediction

    def start_range(self, rec, segment) -> int:
        """Number of valid window starts (target-sample units) in the segment"""
        return rec.num_samples(segment) // self.q - self.total + 1

    def item(self, element: dict, rng: np.random.Generator) -> dict:
        r, s, k = (
            int(element["recording"]),
            int(element["segment"]),
            int(element["slot"]),
        )
        rec = self.recordings[r]
        n = self.start_range(rec, s)
        if self.starts == "random":
            start = int(rng.integers(0, n))
        elif self.starts == "grid":
            start = (
                0
                if self.per_segment == 1
                else round(k * (n - 1) / (self.per_segment - 1))
            )
        elif self.starts == "head":
            start = 0
        else:
            start = n - 1
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
        rec = self.recordings[item["recording"]]
        start = item["start"]
        if (
            start % self.q
            or not 0 <= start <= rec.num_samples(item["segment"]) - item["length"]
        ):
            raise ValueError(
                f"window_policy returned start {start}, outside segment {item['segment']} "
                f"or off the decimation grid (every {self.q} samples)"
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
        if self.preprocess_at == "native":
            for transform in self.preprocess:
                x = transform(x, rec.fs, good)
        if self.q > 1:
            x = signal.resample_poly(x, 1, self.q, axis=0)
        crop = self.pad // self.q
        x = x[crop : crop + self.total]
        if self.preprocess_at == "target":
            for transform in self.preprocess:
                x = transform(x, rec.fs / self.q, good)
        return np.ascontiguousarray(x, np.float32)

    def events(self, item) -> np.ndarray:
        r, s = item["recording"], item["segment"]
        if (r, s) not in self._spikes:
            ev = self.recordings[r].spikes(s)
            times, ids = np.asarray(ev.times, np.float64), np.asarray(ev.ids)
            self._spikes[(r, s)] = (times, ids)
        times, ids = self._spikes[(r, s)]
        fs = self.recordings[r].fs
        t0 = item["start"] / fs * 1e3
        t1 = t0 + item["length"] / fs * 1e3
        keep = (times >= t0) & (times < t1)
        out = np.full((len(self.unit_ids), self.max_spikes), np.inf, np.float32)
        rows = np.searchsorted(self.unit_ids, ids[keep])
        kept = times[keep]
        for row in np.unique(rows):
            mine = np.sort(kept[rows == row])[: self.max_spikes] - t0
            out[row, : len(mine)] = mine
        return out

    def draw(self, element: dict, rng: np.random.Generator) -> dict:
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
        if self.inputs:
            frames = rec.inputs(item["segment"], item["start"], item["length"])
            out["inputs"] = np.ascontiguousarray(
                frames[:: self.q][: self.total], np.float32
            )
        out["channel_mask"] = good if self.mask_bad else np.ones_like(good)
        out["recording"] = np.int32(item["recording"])
        out["segment"] = np.int32(item["segment"])
        out["start"] = np.int64(item["start"])
        return out


class Pipeline:
    """A step-addressed pipeline of fixed-shape batches (for ``livn.optimization.fit``)."""

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

    def batch(self, step: int) -> dict:
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
        key = json.dumps(
            [
                fingerprint,
                seed,
                normalize.n_windows,
                windower.total,
                windower.starts,
                windower.per_segment,
                _identity(recordings, selected),
            ],
            sort_keys=True,
        )
        path = cache / f"stats-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

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
    context_ms: float,
    gap_ms: float = 0.0,
    prediction_ms: float = 0.0,
    fs: float | None = None,
    preprocess: Sequence[Callable] = (),
    normalize: Standardize | None = None,
    starts: str = "random",
    per_segment: int = 10,
    split: str | None = "train",
    spikes: bool = False,
    inputs: bool = False,
    max_spikes: int = 64,
    mask_bad_channels: bool = True,
    batch_size: int,
    seed: int = 0,
    threads: int = 8,
    processes: int = 0,
    window_policy: Callable | None = None,
    preprocess_at: str = "native",
) -> Pipeline:
    """Windows of one or several recordings as a step-addressed :class:`Pipeline`.

    A window is ``context_ms``, then ``gap_ms`` and ``prediction_ms`` of signal, read at the
    recording's rate, preprocessed there, decimated to ``fs`` (zero-phase FIR, with the read
    padded at both ends so the kernel never sees an edge) and normalised.

    Args:
        recording: A :class:`livn.data.Recording` or a list of them (same channels and rate)
        context_ms, gap_ms, prediction_ms: Window layout; ``prediction`` is in the batch only
            when ``prediction_ms > 0``
        fs: Target rate in Hz; the recording's rate must be an integer multiple
        preprocess: Callables ``f(x[T, C] float64, fs, good[C]) -> x`` applied in order at the
            native rate, e.g. ``(Bandpass(300, 3000), CMR())``. ``margin(fs)`` and
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
            slot). The deterministic modes are not shuffled, so ``Pipeline.epoch_steps``
            steps are one pass in order.
        per_segment: Window slots per segment (the epoch length is segments x slots)
        split: Which segments to use: recordings whose ``metadata["split"]`` differs are
            skipped, and ``metadata["segment_split"]`` (one name per segment) selects segments.
            ``None`` takes everything.
        spikes: Add ``spikes``, ``(U, max_spikes)`` per unit of the recordings' spikes, ``U`` =
            the union of unit ids over the selected segments (``Pipeline.unit_ids``)
        inputs: Add the delivered input series (strided to the target rate, not filtered).
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

    from livn.data import Recording

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
    context = round(context_ms * target / 1e3)
    gap = round(gap_ms * target / 1e3)
    prediction = round(prediction_ms * target / 1e3)
    if context < 1:
        raise ValueError(
            f"context_ms={context_ms} is under one sample at {target:g} Hz"
        )
    if per_segment < 1:
        raise ValueError("per_segment must be at least 1")
    slots = 1 if starts in ("head", "tail") else per_segment

    selected = _select(recordings, split)
    entries = [(r, s, k) for r, s in selected for k in range(slots)]
    total = context + gap + prediction
    for r, s in selected:
        if recordings[r].num_samples(s) // q < total:
            raise ValueError(
                f"segment {s} of recording {r} has {recordings[r].num_samples(s)} samples, "
                f"fewer than one window ({total * q})"
            )

    unit_ids = np.zeros(0, np.int64)
    if spikes:
        found = [
            np.unique(np.asarray(recordings[r].spikes(s).ids)) for r, s in selected
        ]
        unit_ids = np.unique(np.concatenate(found)).astype(np.int64)
    if inputs:
        r, s = selected[0]
        recordings[r].inputs(s, 0, 1)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    windower = _Windower(
        recordings,
        q=q,
        context=context,
        gap=gap,
        prediction=prediction,
        starts=starts,
        per_segment=slots,
        preprocess=preprocess,
        mask_bad=mask_bad_channels,
        spikes=spikes,
        inputs=inputs,
        max_spikes=max_spikes,
        unit_ids=unit_ids,
        policy=window_policy,
        preprocess_at=preprocess_at,
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

    return Pipeline(
        dataset,
        windower,
        batch_size,
        threads,
        processes,
        unit_ids,
        len(entries) // batch_size,
    )
