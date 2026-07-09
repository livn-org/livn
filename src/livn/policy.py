from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import numpy as _np

from livn.utils import Jsonable

if TYPE_CHECKING:
    from livn.types import Array

_USES_JAX = False

if "ax" in os.environ.get("LIVN_BACKEND", ""):
    import jax.numpy as _jnp

    _USES_JAX = True


class Policy(Jsonable):
    """Produces a channel-space action array given an optional environment observation.

    Mirrors the RL convention `policy(observation) → action` with the action operating
    upstream of the IO layer (e.g. MEA channel inputs [timestep, n_channels]) to be
    passed through env.cell_stimulus() or wrapped as a Stimulus directly.
    """

    def __call__(self, observation: Any = None) -> "Array":
        raise NotImplementedError

    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def unserialize(cls, data: dict) -> "Policy":
        data = {k: v for k, v in data.items() if k != "class"}
        return cls(**data)


class BiphasicPulsePolicy(Policy):
    """Generates a charge-balanced biphasic waveform for MEA stimulation.

    Args:
        n_channels: Total number of MEA channels.
        channels: Indices of channels to stimulate.
        amplitude: Current amplitude in uA.
        phase_duration: Duration of each phase in ms (default: 0.2 ms = 200 us).
        interphase_gap: Gap between cathodic and anodic phases in ms (default: 0.05 ms = 50 us).
        pulse_times: Onset times for each pulse in ms (default: [0.0]).
        dt: Timestep in ms (default: 0.05 ms = 50 us).
        cathodic_first: If True, cathodic (negative) phase comes first.
    """

    def __init__(
        self,
        n_channels: int,
        channels: list[int] | _np.ndarray,
        amplitude: float = 1.5,
        phase_duration: float = 0.2,
        interphase_gap: float = 0.05,
        pulse_times: list[float] | None = None,
        dt: float = 0.05,
        cathodic_first: bool = True,
    ):
        self.n_channels = n_channels
        self.channels = _np.asarray(channels)
        self.amplitude = amplitude
        self.phase_duration = phase_duration
        self.interphase_gap = interphase_gap
        self.pulse_times = [0.0] if pulse_times is None else list(pulse_times)
        self.dt = dt
        self.cathodic_first = cathodic_first

    def __call__(self, observation: Any = None) -> _np.ndarray:
        """Returns a [timestep, n_channels] float32 array of electrode currents."""
        channels = self.channels
        pulse_times = _np.asarray(self.pulse_times)
        n_channels = self.n_channels
        amplitude = self.amplitude
        phase_duration = self.phase_duration
        interphase_gap = self.interphase_gap
        cathodic_first = self.cathodic_first
        dt = self.dt

        single_pulse_duration = phase_duration + interphase_gap + phase_duration
        total_duration = pulse_times[-1] + single_pulse_duration
        n_steps = int(_np.ceil(total_duration / dt))

        if _USES_JAX:
            inputs = _jnp.zeros((n_steps, n_channels), dtype=_jnp.float32)
        else:
            inputs = _np.zeros((n_steps, n_channels), dtype=_np.float32)

        phase1_steps = int(phase_duration / dt)
        gap_steps = int(interphase_gap / dt)
        phase1_amp = -amplitude if cathodic_first else amplitude
        phase2_amp = amplitude if cathodic_first else -amplitude

        for pulse_onset in pulse_times:
            onset_step = int(pulse_onset / dt)
            phase1_end = onset_step + phase1_steps
            if _USES_JAX:
                inputs = inputs.at[onset_step:phase1_end, channels].set(phase1_amp)
            else:
                inputs[onset_step:phase1_end, channels] = phase1_amp
            phase2_start = onset_step + phase1_steps + gap_steps
            phase2_end = phase2_start + int(phase_duration / dt)
            if phase2_end <= n_steps:
                if _USES_JAX:
                    inputs = inputs.at[phase2_start:phase2_end, channels].set(
                        phase2_amp
                    )
                else:
                    inputs[phase2_start:phase2_end, channels] = phase2_amp

        return inputs

    def serialize(self) -> dict:
        return {
            "n_channels": self.n_channels,
            "channels": self.channels.tolist(),
            "amplitude": self.amplitude,
            "phase_duration": self.phase_duration,
            "interphase_gap": self.interphase_gap,
            "pulse_times": self.pulse_times,
            "dt": self.dt,
            "cathodic_first": self.cathodic_first,
        }


class MonophasicPulsePolicy(Policy):
    """Generates a periodic rectangular pulse train for MEA stimulation.

    Args:
        n_channels: Total number of MEA channels.
        channels: Indices of channels to stimulate.
        amplitude: Current amplitude in uA. Scalar applies to all channels;
            array of length len(channels) sets a per-channel amplitude.
        pulse_width: Duration of each rectangular pulse in ms.
        pulse_times: Onset times for each pulse in ms (default: [0.0]).
        dt: Timestep in ms (default: 1.0 ms).
    """

    def __init__(
        self,
        n_channels: int,
        channels: list[int] | _np.ndarray,
        amplitude: float | list[float] | _np.ndarray = 1.5,
        pulse_width: float = 1.0,
        pulse_times: list[float] | None = None,
        dt: float = 1.0,
    ):
        self.n_channels = n_channels
        self.channels = _np.asarray(channels)
        self.amplitude = (
            amplitude
            if isinstance(amplitude, (int, float))
            else _np.asarray(amplitude, dtype=_np.float32).tolist()
        )
        self.pulse_width = pulse_width
        self.pulse_times = [0.0] if pulse_times is None else list(pulse_times)
        self.dt = dt

    def __call__(self, observation: Any = None) -> _np.ndarray:
        """Returns a [timestep, n_channels] float32 array of electrode currents."""
        channels = self.channels
        pulse_times = _np.asarray(self.pulse_times, dtype=float)
        n_channels = self.n_channels
        pulse_width = self.pulse_width
        dt = self.dt

        amplitudes = _np.broadcast_to(
            _np.asarray(self.amplitude, dtype=_np.float32), channels.shape
        ).copy()

        total_duration = pulse_times[-1] + pulse_width
        n_steps = int(_np.ceil(total_duration / dt))

        if _USES_JAX:
            inputs = _jnp.zeros((n_steps, n_channels), dtype=_jnp.float32)
        else:
            inputs = _np.zeros((n_steps, n_channels), dtype=_np.float32)

        pulse_steps = int(pulse_width / dt)
        for pulse_onset in pulse_times:
            onset_step = int(pulse_onset / dt)
            pulse_end = min(onset_step + pulse_steps, n_steps)
            for ch, amp in zip(channels, amplitudes):
                if amp > 0.0:
                    if _USES_JAX:
                        inputs = inputs.at[onset_step:pulse_end, ch].set(float(amp))
                    else:
                        inputs[onset_step:pulse_end, ch] = amp

        return inputs

    def serialize(self) -> dict:
        return {
            "n_channels": self.n_channels,
            "channels": self.channels.tolist(),
            "amplitude": self.amplitude
            if isinstance(self.amplitude, (int, float))
            else list(self.amplitude),
            "pulse_width": self.pulse_width,
            "pulse_times": self.pulse_times,
            "dt": self.dt,
        }
