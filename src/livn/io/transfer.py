from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as _np

from livn.backend import backend
from livn.io.base import Component, _plain, static_field
from livn.utils import class_path, with_fields

_USES_JAX = "ax" in backend()

if _USES_JAX:
    import jax
    import jax.numpy as np
else:
    import numpy as np


@runtime_checkable
class Transfer(Protocol):
    """Channel to channel: `__call__(signal[C, T], *, dt, key=None) -> [C, T]`."""

    needs_dt: bool

    def __call__(self, signal, *, dt=None, key=None): ...

    def serialize(self) -> dict: ...

    def set_params(self, params: dict): ...


def stage_key(key, index: int):
    if key is None:
        return None
    if _USES_JAX:
        return jax.random.fold_in(key, index)
    if isinstance(key, _np.random.Generator):
        return key
    return _np.random.default_rng([int(key), index])


def _require_dt(dt, name):
    if dt is None:
        raise ValueError(f"{name} depends on the sampling interval: pass dt (ms)")
    return float(dt)


class ElectrodeRC(Component):
    """The electrode-electrolyte interface as a first-order RC low-pass."""

    log_r_seal: object
    c_pF: float = 100.0

    needs_dt = True

    def __init__(self, r_seal=50.0, c_pF: float = 100.0):
        self.log_r_seal = np.log(np.asarray(r_seal, dtype=np.float32))
        self.c_pF = float(c_pF)

    def serialize(self) -> dict:
        return {
            "cls": class_path(self),
            "r_seal": _plain(np.exp(self.log_r_seal)),
            "c_pF": float(self.c_pF),
        }

    def set_params(self, params: dict) -> ElectrodeRC:
        unknown = sorted(set(params) - {"r_seal", "log_r_seal", "c_pF"})
        if unknown:
            raise KeyError(
                f"{unknown} are not parameters of ElectrodeRC; they are "
                "['c_pF', 'log_r_seal', 'r_seal']"
            )
        fields = {}
        if "r_seal" in params:
            fields["log_r_seal"] = np.log(np.asarray(params["r_seal"], np.float32))
        if "log_r_seal" in params:
            fields["log_r_seal"] = np.asarray(params["log_r_seal"], np.float32)
        if "c_pF" in params:
            fields["c_pF"] = float(params["c_pF"])
        return with_fields(self, **fields)

    def __call__(self, signal, *, dt=None, key=None):
        dt = _require_dt(dt, "ElectrodeRC")
        x = np.asarray(signal)
        tau = np.exp(self.log_r_seal) * (self.c_pF * 1e-3)  # ms
        beta = np.broadcast_to(np.atleast_1d(np.exp(-dt / tau)), (x.shape[0],))
        if _USES_JAX:

            def channel(v, b):
                def step(prev, u):
                    out = b * prev + (1.0 - b) * u
                    return out, out

                _, y = jax.lax.scan(step, np.zeros((), v.dtype), v)
                return y

            return jax.vmap(channel)(x, beta)

        from scipy.signal import lfilter

        beta = _np.asarray(beta, dtype=float)
        return _np.stack(
            [
                lfilter([1.0 - b], [1.0, -b], row)
                for row, b in zip(_np.asarray(x), beta, strict=True)
            ]
        )


class FIR(Component):
    """A causal finite impulse response: `y[t] = sum_k taps[k] x[t - k]`."""

    taps: object
    per_channel: bool = static_field(False)

    needs_dt = False

    def __init__(self, taps, per_channel: bool = False):
        taps = np.asarray(taps, dtype=np.float32)
        expected = 2 if per_channel else 1
        if taps.ndim != expected:
            raise ValueError(
                f"{'per-channel ' if per_channel else ''}taps have {expected} "
                f"dimension(s) ({'[channels, K]' if per_channel else '[K]'}), "
                f"got shape {tuple(taps.shape)}"
            )
        self.taps = taps
        self.per_channel = bool(per_channel)

    @classmethod
    def identity(cls, n_taps: int, n_channels: int | None = None) -> FIR:
        taps = np.zeros((n_taps,) if n_channels is None else (n_channels, n_taps))
        taps = taps.at[..., 0].set(1.0) if _USES_JAX else _set_first(taps)
        return cls(taps, per_channel=n_channels is not None)

    def __call__(self, signal, *, dt=None, key=None):
        x = np.asarray(signal)
        if self.per_channel and self.taps.shape[0] != x.shape[0]:
            raise ValueError(
                f"{self.taps.shape[0]} per-channel responses for {x.shape[0]} channels"
            )
        if _USES_JAX:

            def channel(v, k):
                padded = np.pad(v, (k.shape[0] - 1, 0))
                return np.convolve(padded, k, mode="valid")

            if self.per_channel:
                return jax.vmap(channel)(x, self.taps)
            return jax.vmap(channel, in_axes=(0, None))(x, self.taps)

        from scipy.signal import lfilter

        x = _np.asarray(x)
        if self.per_channel:
            return _np.stack(
                [lfilter(k, [1.0], row) for row, k in zip(x, self.taps, strict=True)]
            )
        return lfilter(self.taps, [1.0], x, axis=-1)


def _set_first(taps):
    taps[..., 0] = 1.0
    return taps


class Bandpass(Component):
    """A Butterworth band-pass, or a high-pass when high is None."""

    low: float = static_field(300.0)
    high: float | None = static_field(3000.0)
    order: int = static_field(4)
    form: str = static_field("band")
    phase: str = static_field("causal")

    needs_dt = True

    @classmethod
    def from_filter(cls, spec) -> Bandpass:
        if spec.get("kind") != "butterworth_bandpass":
            raise ValueError(
                f"filter kind is 'butterworth_bandpass', not {spec.get('kind')!r}"
            )
        if spec.get("design") != "rbj_biquads":
            raise ValueError(
                f"filter design is 'rbj_biquads', not {spec.get('design')!r}"
            )
        if spec.get("causal") is not True:
            raise ValueError(
                "only a causal filter object describes an instrument's filter"
            )
        return cls(
            low=float(spec["low_cut_hz"]),
            high=float(spec["high_cut_hz"]),
            order=int(spec["order"]),
            form="cascade",
            phase="causal",
        )

    def sos(self, fs: float) -> _np.ndarray:
        """The second-order sections ``[b0, b1, b2, 1, a1, a2]`` at ``fs`` Hz."""
        fs = float(fs)
        nyquist = fs / 2.0
        high = None if self.high is None or self.high >= nyquist else float(self.high)
        if self.high is not None and high is None and self.form == "cascade":
            raise ValueError(
                f"a cascade's high cut of {self.high} Hz is not below Nyquist ({nyquist} Hz)"
            )
        if not 0.0 < self.low < (nyquist if high is None else high):
            raise ValueError(
                f"a band of [{self.low}, {self.high}] Hz needs "
                f"0 < low < high < Nyquist ({nyquist} Hz at {fs} Hz)"
            )
        if self.form == "band":
            from scipy.signal import butter

            if high is None:
                return butter(self.order, self.low, "highpass", fs=fs, output="sos")
            return butter(self.order, [self.low, high], "band", fs=fs, output="sos")
        if self.form == "cascade":
            sections = _butterworth(True, self.low, fs, int(self.order))
            if high is not None:
                sections += _butterworth(False, high, fs, int(self.order))
            return _np.asarray(sections, dtype=float)
        raise ValueError(f"form is 'band' or 'cascade', not {self.form!r}")

    def filter(self, x, fs: float, *, axis: int = -1, zi=None):
        from scipy.signal import sosfilt, sosfiltfilt

        x = _np.asarray(x, dtype=float)
        sos = self.sos(fs)
        if self.phase == "zero":
            if zi is not None:
                raise ValueError(
                    "a zero-phase filter has no state to carry across blocks"
                )
            return sosfiltfilt(sos, x, axis=axis)
        if self.phase != "causal":
            raise ValueError(f"phase is 'causal' or 'zero', not {self.phase!r}")
        if zi is None:
            return sosfilt(sos, x, axis=axis)
        return sosfilt(sos, x, axis=axis, zi=zi)

    def zeros_state(
        self, fs: float, shape: tuple[int, ...], axis: int = -1
    ) -> _np.ndarray:
        n = len(self.sos(fs))
        rest = list(shape)
        rest[axis] = 2
        return _np.zeros((n, *rest))

    def margin(self, fs: float) -> int:
        return int(_np.ceil(8 * fs / self.low))

    def fingerprint(self) -> str:
        high = "" if self.high is None else format(self.high, "g")
        core = f"({self.low:g},{high},{self.order})"
        if self.phase == "zero" and self.form == "band":
            return f"bp{core}"
        return f"bp-{self.form}-{self.phase}{core}"

    def __call__(self, signal, *, dt=None, key=None):
        fs = 1000.0 / _require_dt(dt, "Bandpass")
        if not _USES_JAX:
            return self.filter(signal, fs, axis=-1)
        if self.phase != "causal":
            raise ValueError(
                "a zero-phase Bandpass is not a recording stage under jax; "
                "use filter() on NumPy arrays"
            )
        y = np.asarray(signal)
        for b0, b1, b2, _a0, a1, a2 in self.sos(fs):
            b0, b1, b2, a1, a2 = (float(v) for v in (b0, b1, b2, a1, a2))

            def step(state, u, b0=b0, b1=b1, b2=b2, a1=a1, a2=a2):
                z1, z2 = state
                out = b0 * u + z1
                return (b1 * u - a1 * out + z2, b2 * u - a2 * out), out

            zero = np.zeros(y.shape[0], y.dtype)
            _, y = jax.lax.scan(step, (zero, zero), y.T)
            y = y.T
        return y


def _section(b0, b1, b2, a0, a1, a2) -> list[float]:
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def _butterworth(
    high_pass: bool, f0: float, fs: float, order: int
) -> list[list[float]]:
    if order < 1:
        raise ValueError(f"order is 1 or more, not {order}")
    w0 = 2.0 * math.pi * f0 / fs
    sin, cos = math.sin(w0), math.cos(w0)
    sections = []
    for k in range(1, order // 2 + 1):
        q = 1.0 / (2.0 * math.sin((2 * k - 1) * math.pi / (2.0 * order)))
        alpha = sin / (2.0 * q)
        if high_pass:
            b = ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0)
        else:
            b = ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0)
        sections.append(_section(*b, 1.0 + alpha, -2.0 * cos, 1.0 - alpha))
    if order % 2:
        t = math.tan(math.pi * f0 / fs)
        b = (1.0, -1.0, 0.0) if high_pass else (t, t, 0.0)
        sections.append(_section(*b, 1.0 + t, t - 1.0, 0.0))
    return sections


class Noise(Component):
    """Additive Gaussian noise of standard deviation `sigma_uV`."""

    sigma_uV: float = 0.0
    seed: int = static_field(0)

    needs_dt = False

    def __call__(self, signal, *, dt=None, key=None):
        if key is None:
            return signal
        x = np.asarray(signal)
        if _USES_JAX:
            return x + self.sigma_uV * jax.random.normal(key, x.shape, x.dtype)
        rng = (
            key
            if isinstance(key, _np.random.Generator)
            else _np.random.default_rng(key)
        )
        return x + rng.normal(0.0, float(self.sigma_uV), size=x.shape)

    def apply_seeded(self, x):
        rng = _np.random.default_rng(int(self.seed))
        x = _np.asarray(x)
        return x + rng.normal(0.0, float(self.sigma_uV), size=x.shape)


__all__ = ["FIR", "Bandpass", "ElectrodeRC", "Noise", "Transfer", "stage_key"]
