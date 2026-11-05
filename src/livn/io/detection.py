"""A threshold detector.

Sigma is ``median(|y|) / 0.6745`` of the band-passed signal following

    Quiroga, R. Quian, et al. “Unsupervised Spike Detection and Sorting
    with Wavelets and Superparamagnetic Clustering.” Neural Computation,
    vol. 16, no. 8, Aug. 2004, pp. 1661-87. IEEE Xplore,
    https://doi.org/10.1162/089976604774201631.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from livn.io.transfer import Bandpass

METHOD = "threshold"
FILTER = "butterworth_bandpass"
DESIGN = "rbj_biquads"
SIGMA_RULE = "mad"  # median(|y|) / 0.6745
THRESHOLD_RULE = "frozen_sigma"
SIGMA_DIVISOR = 0.6745
SIGMA_STRIDE = 5


class MethodError(ValueError):
    """Invalid method object."""


def threshold_method(
    k: float,
    sigma_window: float | tuple[float, float],
    *,
    sigma_stride: int = SIGMA_STRIDE,
    low_cut: float = 300.0,
    high_cut: float = 3000.0,
    dead_time: float = 1.0,
) -> dict[str, Any]:
    """Closed-loop threshold rule."""
    window = (
        [0.0, float(sigma_window)]
        if isinstance(sigma_window, (int, float))
        else [float(v) for v in sigma_window]
    )
    threshold = {
        "rule": THRESHOLD_RULE,
        "k": float(k),
        "polarity": "negative",
        "sigma": SIGMA_RULE,
        "sigma_window_ms": window,
        "dead_time_ms": float(dead_time),
    }
    if int(sigma_stride) != SIGMA_STRIDE:
        threshold["sigma_stride"] = int(sigma_stride)
    return check_method(
        {
            "kind": METHOD,
            "filter": {
                "kind": FILTER,
                "low_cut_hz": float(low_cut),
                "high_cut_hz": float(high_cut),
                "order": 2,
                "causal": True,
                "design": DESIGN,
            },
            "threshold": threshold,
        }
    )


def check_method(method: Mapping) -> dict[str, Any]:
    if not isinstance(method, Mapping):
        raise MethodError(f"a method is an object, not {method!r}")
    plain = json.loads(json.dumps(method, allow_nan=False))
    if plain.get("kind") != METHOD:
        raise MethodError(f"kind is {METHOD!r}, not {plain.get('kind')!r}")
    flt, thr = plain.get("filter"), plain.get("threshold")
    if not isinstance(flt, dict) or not isinstance(thr, dict):
        raise MethodError("a threshold method has a 'filter' and a 'threshold' object")

    def want(block, name, ok, what):
        value = block.get(name)
        if not ok(value):
            raise MethodError(f"{name} is {what}, not {value!r}")
        return value

    want(flt, "kind", lambda v: v == FILTER, repr(FILTER))
    want(
        flt,
        "order",
        lambda v: type(v) is int and 1 <= v <= 8,
        "a whole number from 1 to 8",
    )
    want(flt, "causal", lambda v: v is True, "true (the detector is causal)")
    want(flt, "design", lambda v: v == DESIGN, repr(DESIGN))

    low = want(
        flt,
        "low_cut_hz",
        lambda v: type(v) in (int, float) and v > 0,
        "a positive frequency",
    )
    high = want(
        flt,
        "high_cut_hz",
        lambda v: type(v) in (int, float) and v > 0,
        "a positive frequency",
    )
    if not low < high:
        raise MethodError(f"low_cut_hz {low} is not below high_cut_hz {high}")

    want(thr, "rule", lambda v: v == THRESHOLD_RULE, repr(THRESHOLD_RULE))
    want(thr, "k", lambda v: type(v) in (int, float) and v >= 1, "1 or more")
    want(thr, "polarity", lambda v: v == "negative", "'negative'")
    want(thr, "sigma", lambda v: v == SIGMA_RULE, repr(SIGMA_RULE))
    window = want(
        thr,
        "sigma_window_ms",
        lambda v: (
            type(v) is list and len(v) == 2 and all(type(t) in (int, float) for t in v)
        ),
        "[start, end] in ms",
    )
    if not 0 <= window[0] < window[1]:
        raise MethodError(f"sigma_window_ms {window} is not a window from 0 ms on")
    want(
        thr,
        "sigma_stride",
        lambda v: v is None or (type(v) is int and v >= 1),
        "a whole number of frames, 1 or more (absent: 5)",
    )
    want(
        thr,
        "dead_time_ms",
        lambda v: type(v) in (int, float) and v > 0,
        "a positive duration in ms",
    )
    return plain


@dataclass(frozen=True)
class Detection:
    frames: list[np.ndarray]
    sigma_uV: np.ndarray
    threshold_uV: np.ndarray
    state_from_frame: int


class Detector:
    def __init__(
        self,
        method: Mapping,
        fs: float,
        n_rows: int,
        sigma_uV: np.ndarray | None = None,
    ):
        self.method = check_method(method)
        self.fs = float(fs)
        self.n_rows = int(n_rows)
        flt = self.method["filter"]
        if not flt["high_cut_hz"] < self.fs / 2:
            raise MethodError(
                f"high_cut_hz {flt['high_cut_hz']} is not below the Nyquist rate "
                f"{self.fs / 2:g} Hz"
            )
        self._filter = Bandpass.from_filter(flt)
        self._state = self._filter.zeros_state(self.fs, (1, self.n_rows), axis=0)
        thr = self.method["threshold"]
        start, end = thr["sigma_window_ms"]
        self._start = math.floor(start / 1000.0 * self.fs + 0.5)
        self._end = math.floor(end / 1000.0 * self.fs + 0.5)
        self._stride = int(thr.get("sigma_stride") or SIGMA_STRIDE)
        self._k = float(thr["k"])
        self._dead = max(int(self.fs / 1000.0 * float(thr["dead_time_ms"])), 1)
        self._n = 0  # frames fed
        self._head: list[np.ndarray] = []  # |y| of the window's strided frames, float32
        if sigma_uV is None:
            if self._end <= self._start:
                raise MethodError(
                    f"sigma_window_ms {thr['sigma_window_ms']} is under a frame"
                )
            self.sigma: np.ndarray | None = None
            self.threshold: np.ndarray | None = None
            self.state_from_frame = self._end
        else:
            sigma = np.asarray(sigma_uV, dtype=np.float64)
            if sigma.shape != (self.n_rows,):
                raise ValueError(
                    f"sigma_uV is one per row ({self.n_rows}), not {sigma.shape}"
                )
            self.sigma, self.threshold = sigma, -self._k * sigma
            self.state_from_frame = 0
        # the last frame each row's dead time covers
        self._quiet = np.full(self.n_rows, -1, dtype=np.int64)
        self._events: list[list[np.ndarray]] = [[] for _ in range(self.n_rows)]

    @classmethod
    def detect(
        cls,
        x: np.ndarray,
        fs: float,
        method: Mapping,
        sigma_uV: np.ndarray | None = None,
    ) -> Detection:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x is (frames, rows), not of shape {x.shape}")
        return cls(method, fs, x.shape[1], sigma_uV).feed(x).finish()

    def feed(self, x: np.ndarray) -> Detector:
        y = np.asarray(x, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.n_rows:
            raise ValueError(f"a block is (frames, {self.n_rows}), not {y.shape}")
        y, self._state = self._filter.filter(y, self.fs, axis=0, zi=self._state)
        first = self._n
        self._n += len(y)
        if self.sigma is None:
            lo = max(self._start, first)
            lo += (self._start - lo) % self._stride
            hi = min(self._end, self._n)
            if lo < hi:
                self._head.append(
                    np.abs(y[lo - first : hi - first : self._stride]).astype(np.float32)
                )
            if self._n < self._end:
                return self
            self._freeze()
            y, first = y[self._end - first :], self._end
        self._scan(y, first)
        return self

    def finish(self) -> Detection:
        if self.sigma is None:
            raise MethodError(
                f"the sigma window ends at frame {self._end}, and the rows hold {self._n}: "
                "the window must fit inside them"
            )
        assert self.threshold is not None
        return Detection(
            frames=[
                np.concatenate(e) if e else np.zeros(0, np.int64) for e in self._events
            ],
            sigma_uV=self.sigma,
            threshold_uV=self.threshold,
            state_from_frame=int(self.state_from_frame),
        )

    def _freeze(self) -> None:
        head = np.sort(np.concatenate(self._head), axis=0).astype(np.float64)
        self._head = []
        n = len(head)
        # numpy's median of the float32 values, the middle pair averaged in float64
        median = head[n // 2] if n % 2 else (head[n // 2 - 1] + head[n // 2]) / 2.0
        self.sigma = median / SIGMA_DIVISOR
        self.threshold = -self._k * self.sigma

    def _scan(self, y: np.ndarray, first: int) -> None:
        if not len(y):
            return
        rows, at = np.nonzero((y < self.threshold).T)
        frames = at.astype(np.int64) + first
        bounds = np.searchsorted(rows, np.arange(self.n_rows + 1))
        for row in range(self.n_rows):
            candidates = frames[bounds[row] : bounds[row + 1]]
            if not len(candidates):
                continue
            kept = []
            quiet = self._quiet[row]
            for t in candidates.tolist():
                if t > quiet:
                    kept.append(t)
                    quiet = t + self._dead
            self._quiet[row] = quiet
            if kept:
                self._events[row].append(np.asarray(kept, dtype=np.int64))
