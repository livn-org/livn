from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import numpy as _np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from livn.utils import Jsonable

if TYPE_CHECKING:
    from livn.types import Array

_USES_JAX = False

if "ax" in os.environ.get("LIVN_BACKEND", ""):
    import jax.numpy as _jnp

    _USES_JAX = True


def _zeros(n_steps: int, n_channels: int):
    if _USES_JAX:
        return _jnp.zeros((n_steps, n_channels), dtype=_jnp.float32)
    return _np.zeros((n_steps, n_channels), dtype=_np.float32)


def _write(inputs, rows: slice, channel, value):
    """`inputs[rows, channel] = value`, on whichever array library is in play."""
    if _USES_JAX:
        return inputs.at[rows, channel].set(value)
    inputs[rows, channel] = value
    return inputs


class Policy(BaseModel, Jsonable):
    """Produces a channel-space action array, given an optional observation.

    Mirrors the RL convention `policy(observation) -> action`, with the action
    sitting upstream of the IO layer: it is what the array is told to do, not
    what any cell receives.
    """

    model_config = ConfigDict(extra="forbid")

    def __call__(self, observation: Any = None) -> "Array":
        raise NotImplementedError

    def serialize(self) -> dict:
        return self.model_dump()

    @classmethod
    def unserialize(cls, data: dict) -> "Policy":
        return cls(**{k: v for k, v in data.items() if k != "class"})


class ElectrodePolicy(Policy):
    """A policy that drives named channels of an array of known width."""

    n_channels: int | None = Field(default=None, gt=0)
    """Total channels on the array. `None` until an array is in hand."""

    channels: list[int] = []
    """Which of them this policy drives."""

    dt: float = Field(default=1.0, gt=0)
    """Timestep of the produced array, in ms."""

    @model_validator(mode="after")
    def _channels_fit_the_array(self):
        if self.n_channels is None or not self.channels:
            return self
        out_of_range = [c for c in self.channels if not 0 <= c < self.n_channels]
        if out_of_range:
            raise ValueError(
                f"channels {out_of_range} are outside a {self.n_channels}-channel "
                "array; a command written there would be silently dropped"
            )
        return self

    def for_array(
        self, n_channels: int, channels: list[int] | None = None, **overrides
    ):
        data = self.model_dump()
        data["n_channels"] = n_channels
        if channels is not None:
            data["channels"] = list(channels)
        data.update(overrides)
        return type(self)(**data)

    def _resolved(self) -> tuple[int, _np.ndarray]:
        if self.n_channels is None or not self.channels:
            raise ValueError(
                f"{type(self).__name__} has no array to drive -- set `n_channels` "
                "and `channels`, or call `.for_array(...)` once the geometry is "
                "known"
            )
        return self.n_channels, _np.asarray(self.channels)


class BiphasicPulsePolicy(ElectrodePolicy):
    """A charge-balanced biphasic pulse train.

    Each pulse is `phase_duration` of one polarity, `interphase_gap` of
    nothing, then `phase_duration` of the other so every pulse integrates to
    zero, which is what keeps a real electrode from polarising.
    """

    n_channels: int = Field(gt=0)
    channels: list[int]
    dt: float = Field(default=0.05, gt=0)

    amplitude: float = 1.5
    phase_duration: float = Field(default=0.2, gt=0)
    """Duration of each phase, in ms (default 200 us)."""

    interphase_gap: float = Field(default=0.05, ge=0)
    """Gap between the two phases, in ms (default 50 us)."""

    pulse_times: list[float] = [0.0]
    """Onset of each pulse, in ms."""

    cathodic_first: bool = True

    @field_validator("pulse_times")
    @classmethod
    def _at_least_one_pulse(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("a pulse train needs at least one pulse time")
        return v

    def __call__(self, observation: Any = None) -> "Array":
        n_channels, channels = self._resolved()
        pulse_times = _np.asarray(self.pulse_times, dtype=float)
        dt = self.dt

        single = self.phase_duration + self.interphase_gap + self.phase_duration
        n_steps = int(_np.ceil((pulse_times[-1] + single) / dt))
        inputs = _zeros(n_steps, n_channels)

        phase_steps = int(self.phase_duration / dt)
        gap_steps = int(self.interphase_gap / dt)
        first = -self.amplitude if self.cathodic_first else self.amplitude

        for onset in pulse_times:
            start = int(onset / dt)
            first_end = start + phase_steps
            inputs = _write(inputs, slice(start, first_end), channels, first)
            second_start = first_end + gap_steps
            second_end = second_start + phase_steps
            if second_end <= n_steps:
                inputs = _write(
                    inputs, slice(second_start, second_end), channels, -first
                )

        return inputs


class MonophasicPulsePolicy(ElectrodePolicy):
    """A rectangular pulse train of one polarity."""

    n_channels: int = Field(gt=0)
    channels: list[int]

    amplitude: float | list[float] = 1.5
    pulse_width: float = Field(default=1.0, gt=0)
    """Duration of each pulse, in ms."""

    pulse_times: list[float] = [0.0]
    """Onset of each pulse, in ms."""

    @field_validator("pulse_times")
    @classmethod
    def _at_least_one_pulse(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("a pulse train needs at least one pulse time")
        return v

    @model_validator(mode="after")
    def _amplitude_matches_the_channels(self):
        if isinstance(self.amplitude, list) and len(self.amplitude) != len(
            self.channels
        ):
            raise ValueError(
                f"{len(self.amplitude)} amplitudes for {len(self.channels)} "
                "channels; a per-channel amplitude names one value per channel, "
                "in the same order"
            )
        return self

    def __call__(self, observation: Any = None) -> "Array":
        n_channels, channels = self._resolved()
        pulse_times = _np.asarray(self.pulse_times, dtype=float)
        dt = self.dt

        amplitudes = _np.broadcast_to(
            _np.asarray(self.amplitude, dtype=_np.float32), channels.shape
        )

        n_steps = int(_np.ceil((pulse_times[-1] + self.pulse_width) / dt))
        inputs = _zeros(n_steps, n_channels)

        pulse_steps = int(self.pulse_width / dt)
        for onset in pulse_times:
            start = int(onset / dt)
            end = min(start + pulse_steps, n_steps)
            for channel, amplitude in zip(channels, amplitudes):
                if amplitude > 0.0:
                    inputs = _write(
                        inputs, slice(start, end), channel, float(amplitude)
                    )

        return inputs


class PulseSweepPolicy(ElectrodePolicy):
    """One pulse per trial, cycling through a set of amplitudes."""

    amplitudes: tuple[float, ...] = (300.0, 400.0, 500.0, 600.0)
    """The amplitudes to sweep, in the order they are first delivered."""

    repeats: int = Field(default=8, ge=1)
    """Trials per amplitude."""

    trial_ms: float = Field(default=2000.0, gt=0)
    """Spacing between pulses.

    Has to leave the network back at steady state before the next pulse, since
    a response is measured against the baseline immediately before it."""

    onset_ms: float = Field(default=1000.0, ge=0)
    """Where the pulse falls inside its trial, and so how much precedes it."""

    pulse_ms: float = Field(default=0.2, gt=0)
    """Total width of the biphasic pulse, half to each phase."""

    order: tuple[int, ...] = ()
    """Amplitude index per trial. Empty cycles through them in turn.

    A fixed pseudorandom sequence keeps adjacent trials from sharing an
    amplitude; cycling achieves the same and is reproducible."""

    start_ms: float = Field(default=0.0, ge=0)
    """Where the sweep begins, so a run can hold a free-running stretch first."""

    total_ms: float | None = Field(default=None, gt=0)
    """Length of the produced array. `None` ends it with the last trial."""

    dt: float = Field(default=0.1, gt=0)

    @field_validator("amplitudes")
    @classmethod
    def _at_least_one_amplitude(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v:
            raise ValueError("a sweep needs at least one amplitude")
        return v

    @model_validator(mode="after")
    def _the_pulse_fits_its_trial(self):
        if self.onset_ms >= self.trial_ms:
            raise ValueError(
                f"the pulse falls at {self.onset_ms:g} ms of a {self.trial_ms:g} ms "
                "trial, which is outside it"
            )
        if self.onset_ms + self.pulse_ms > self.trial_ms:
            raise ValueError("the pulse does not fit inside its trial")
        if self.order and len(self.order) != self.n_trials:
            raise ValueError(
                f"order names {len(self.order)} trials but the sweep runs "
                f"{self.n_trials}"
            )
        if self.order and max(self.order) >= len(self.amplitudes):
            raise ValueError(
                f"order names amplitude {max(self.order)} but only "
                f"{len(self.amplitudes)} were given"
            )
        return self

    @property
    def n_trials(self) -> int:
        return len(self.amplitudes) * self.repeats

    @property
    def duration_ms(self) -> float:
        """How long the sweep itself lasts, excluding `start_ms`."""
        return self.n_trials * self.trial_ms

    def schedule(self, start_ms: float | None = None) -> list[tuple[float, float]]:
        """`(pulse time, amplitude)` per trial, in order.

        Times are absolute, so `start_ms` is where the stimulated segment
        begins after whatever free-running stretch precedes it.
        """
        start = self.start_ms if start_ms is None else start_ms
        order = self.order or tuple(
            i % len(self.amplitudes) for i in range(self.n_trials)
        )
        return [
            (
                start + trial * self.trial_ms + self.onset_ms,
                float(self.amplitudes[index]),
            )
            for trial, index in enumerate(order)
        ]

    def __call__(self, observation: Any = None) -> "Array":
        n_channels, channels = self._resolved()
        dt = self.dt

        total = (
            self.start_ms + self.duration_ms if self.total_ms is None else self.total_ms
        )
        n_steps = int(round(total / dt))
        inputs = _zeros(n_steps, n_channels)

        width = max(1, int(round(self.pulse_ms / dt)))
        for at, amplitude in self.schedule():
            start = int(round(at / dt))
            end = start + width
            if end > n_steps:
                raise ValueError(
                    f"a pulse at {at:g} ms does not fit in a {total:g} ms run"
                )
            half = start + max(1, width // 2)
            for channel in channels:
                inputs = _write(inputs, slice(start, half), channel, -amplitude)
                inputs = _write(inputs, slice(half, end), channel, amplitude)

        return inputs
