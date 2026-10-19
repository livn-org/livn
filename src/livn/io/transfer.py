from __future__ import annotations

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
    """A causal Butterworth band-pass."""

    low: float = static_field(300.0)
    high: float = static_field(3000.0)
    order: int = static_field(4)

    needs_dt = True

    def sos(self, dt: float):
        from scipy.signal import butter

        fs = 1000.0 / dt
        if not 0.0 < self.low < self.high < fs / 2.0:
            raise ValueError(
                f"a band of [{self.low}, {self.high}] Hz needs "
                f"0 < low < high < Nyquist ({fs / 2.0} Hz at dt={dt} ms)"
            )
        return butter(
            self.order, [self.low, self.high], btype="band", fs=fs, output="sos"
        )

    def __call__(self, signal, *, dt=None, key=None):
        sos = self.sos(_require_dt(dt, "Bandpass"))
        x = np.asarray(signal)
        if not _USES_JAX:
            from scipy.signal import sosfilt

            return sosfilt(sos, _np.asarray(x), axis=-1)

        y = x
        for b0, b1, b2, _a0, a1, a2 in sos:
            b0, b1, b2, a1, a2 = (float(v) for v in (b0, b1, b2, a1, a2))

            def step(state, u, b0=b0, b1=b1, b2=b2, a1=a1, a2=a2):
                z1, z2 = state
                out = b0 * u + z1
                return (b1 * u - a1 * out + z2, b2 * u - a2 * out), out

            zero = np.zeros(y.shape[0], y.dtype)
            _, y = jax.lax.scan(step, (zero, zero), y.T)
            y = y.T
        return y


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
