from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in os.environ.get("LIVN_BACKEND", ""):
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


class Stimulus:
    def __init__(
        self,
        array: Float[Array, "timestep n_gids"] | None = None,
        dt: float = 1.0,
        gids: Int[Array, "n_gids"] | None = None,
        **meta_data,
    ):
        self.array = array
        if dt <= 0:
            raise ValueError("Stimulus dt must be positive")
        self.dt = dt
        self.gids = gids
        self.meta_data = meta_data

    @property
    def duration(self) -> float:
        if self.array is None:
            return 0.0
        return self.array.shape[0] * self.dt

    @property
    def input_mode(
        self,
    ) -> Literal[
        "extracellular", "current", "current_density", "conductance", "irradiance"
    ]:
        return self.meta_data.get("input_mode", "extracellular")

    def __iter__(self):
        yield from zip(self.gids, self.array.T)

    def __len__(self):
        return self.array.shape[-1]

    @classmethod
    def from_arg(cls, stimulus) -> "Stimulus":
        if isinstance(stimulus, cls):
            return stimulus

        if stimulus is None:
            return cls()

        if hasattr(stimulus, "shape"):
            return cls(stimulus)

        if isinstance(stimulus, (tuple, list)):
            return cls(*stimulus)

        if isinstance(stimulus, dict):
            return cls(**stimulus)

        raise ValueError("Invalid stimulus", stimulus)

    @classmethod
    def biphasic_pulse(
        cls,
        n_channels: int,
        channels: list[int] | np.ndarray,
        amplitude: float = 1.5,
        phase_duration: float = 0.2,
        interphase_gap: float = 0.05,
        pulse_times: list[float] | None = None,
        dt: float = 0.05,
        cathodic_first: bool = True,
    ) -> "Stimulus":
        """Generate charge-balanced biphasic waveform for MEA stimulation

        Args:
            n_channels: Total number of MEA channels (io.num_channels)
            channels: Indices of channels to stimulate
            amplitude: Current amplitude in uA
            phase_duration: Duration of each phase in ms (default: 0.2ms = 200us)
            interphase_gap: Gap between cathodic and anodic phases in ms (default: 0.05ms = 50us).
            pulse_times: Onset times for each pulse in ms, e.g. [0.0, 50.0] creates a paired-pulse at 50ms interval (default: [0.0])
            dt: Timestep (default: 0.05ms = 50us).
            cathodic_first: If True, cathodic (negative) phase first
        """
        if pulse_times is None:
            pulse_times = [0.0]

        channels = np.asarray(channels)
        pulse_times = np.asarray(pulse_times)

        single_pulse_duration = phase_duration + interphase_gap + phase_duration
        total_duration = pulse_times[-1] + single_pulse_duration

        n_steps = int(np.ceil(total_duration / dt))
        inputs = np.zeros((n_steps, n_channels), dtype=np.float32)

        phase1_steps = int(phase_duration / dt)
        gap_steps = int(interphase_gap / dt)
        phase2_steps = int(phase_duration / dt)

        phase1_amp = -amplitude if cathodic_first else amplitude
        phase2_amp = amplitude if cathodic_first else -amplitude

        for pulse_onset in pulse_times:
            onset_step = int(pulse_onset / dt)

            phase1_start = onset_step
            phase1_end = onset_step + phase1_steps
            if _USES_JAX:
                inputs = inputs.at[phase1_start:phase1_end, channels].set(phase1_amp)
            else:
                inputs[phase1_start:phase1_end, channels] = phase1_amp

            phase2_start = onset_step + phase1_steps + gap_steps
            phase2_end = phase2_start + phase2_steps
            if phase2_end <= n_steps:
                if _USES_JAX:
                    inputs = inputs.at[phase2_start:phase2_end, channels].set(
                        phase2_amp
                    )
                else:
                    inputs[phase2_start:phase2_end, channels] = phase2_amp

        return cls(
            array=inputs,
            dt=dt,
            # metadata
            kind="biphasic_pulse",
            pulse_times=pulse_times.tolist(),
            phase_duration=phase_duration,
            interphase_gap=interphase_gap,
            amplitude=amplitude,
            channels=channels.tolist(),
            cathodic_first=cathodic_first,
        )

    @classmethod
    def monophasic_pulse(
        cls,
        n_channels: int,
        channels: list[int] | np.ndarray,
        amplitude: float | list[float] | np.ndarray = 1.5,
        pulse_width: float = 1.0,
        pulse_times: list[float] | None = None,
        dt: float = 1.0,
    ) -> "Stimulus":
        """Generate a periodic rectangular pulse train for MEA stimulation

        Args:
            n_channels: Total number of MEA channels (io.num_channels)
            channels: Indices of channels to stimulate
            amplitude: Current amplitude in uA. Scalar applies to all channels;
                array of length len(channels) sets a per-channel amplitude
            pulse_width: Duration of each rectangular pulse
            pulse_times: Onset times for each pulse (default None = [0.0])
            dt: Timestep
        """
        if pulse_times is None:
            pulse_times = [0.0]

        channels = np.asarray(channels)
        pulse_times = np.asarray(pulse_times, dtype=float)
        amplitudes = np.broadcast_to(
            np.asarray(amplitude, dtype=np.float32), channels.shape
        ).copy()

        total_duration = pulse_times[-1] + pulse_width
        n_steps = int(np.ceil(total_duration / dt))
        inputs = np.zeros((n_steps, n_channels), dtype=np.float32)

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

        return cls(
            array=inputs,
            dt=dt,
            # metadata
            kind="monophasic_pulse",
            pulse_times=pulse_times.tolist(),
            pulse_width=pulse_width,
            amplitude=amplitudes.tolist(),
            channels=channels.tolist(),
        )

    @classmethod
    def from_conductance(
        cls,
        conductance: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **meta_data,
    ) -> "Stimulus":
        """Create stimulus from synaptic conductance values

        Args:
            conductance: in uS
            dt: Time step in ms
            gids: Neuron GIDs
            **meta_data: Additional metadata
        """
        meta_data["input_mode"] = "conductance"
        meta_data["units"] = "uS"
        return cls(array=conductance, dt=dt, gids=gids, **meta_data)

    @classmethod
    def from_current(
        cls,
        current: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **meta_data,
    ) -> "Stimulus":
        """Create stimulus from direct current injection

        Args:
            current: Current values in nA
            dt: Time step in ms
            gids: Neuron GIDs
            **meta_data: Additional metadata
        """
        meta_data["input_mode"] = "current"
        meta_data["units"] = "nA"
        return cls(array=current, dt=dt, gids=gids, **meta_data)

    @classmethod
    def from_current_density(
        cls,
        current_density: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **meta_data,
    ) -> "Stimulus":
        """Create stimulus from current density

        Args:
            current_density: Current density values in mA/cm2
            dt: Time step in ms
            gids: Neuron GIDs
            **meta_data: Additional metadata
        """
        meta_data["input_mode"] = "current_density"
        meta_data["units"] = "mA/cm2"
        return cls(array=current_density, dt=dt, gids=gids, **meta_data)

    @classmethod
    def from_extracellular(
        cls,
        voltage: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **meta_data,
    ) -> "Stimulus":
        meta_data["input_mode"] = "extracellular"
        meta_data["units"] = "mV"
        return cls(array=voltage, dt=dt, gids=gids, **meta_data)

    @classmethod
    def from_irradiance(
        cls,
        irradiance: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **meta_data,
    ) -> "Stimulus":
        """Optical stimulus as irradiance at each neuron (mW/mm^2).

        Args:
            irradiance: Light power density at each neuron, shape [timestep, n_gids].
        """
        meta_data["input_mode"] = "irradiance"
        meta_data["units"] = "mW/mm2"
        return cls(array=irradiance, dt=dt, gids=gids, **meta_data)

    @staticmethod
    def align_gids(
        stimulus: "Stimulus",
        all_gids: Int[Array, "n_total_gids"],
    ) -> "Stimulus":
        """Expand stimulus array to cover all_gids, zero-padding missing neurons"""
        if stimulus.array is None:
            return stimulus

        if stimulus.gids is None:
            assert stimulus.array.shape[-1] == len(all_gids), (
                f"Stimulus has {stimulus.array.shape[-1]} columns but system has "
                f"{len(all_gids)} neurons. Set gids= explicitly."
            )
            return stimulus

        gid_to_idx = {int(g): i for i, g in enumerate(all_gids)}
        n_timesteps = stimulus.array.shape[0]
        expanded = np.zeros((n_timesteps, len(all_gids)), dtype=stimulus.array.dtype)
        for col_idx, gid in enumerate(stimulus.gids):
            sys_idx = gid_to_idx.get(int(gid))
            if sys_idx is None:
                raise ValueError(
                    f"Stimulus targets GID {gid} which is not in the system"
                )
            if _USES_JAX:
                expanded = expanded.at[:, sys_idx].add(stimulus.array[:, col_idx])
            else:
                expanded[:, sys_idx] += stimulus.array[:, col_idx]

        return Stimulus(
            array=expanded, dt=stimulus.dt, gids=all_gids, **stimulus.meta_data
        )

    @staticmethod
    def resample(
        stimulus: "Stimulus",
        target_dt: float,
        duration: float,
    ) -> "Stimulus":
        """Resample stimulus to a common dt via linear interpolation"""
        if stimulus.array is None or np.isclose(stimulus.dt, target_dt):
            return stimulus

        n_target_steps = int(round(duration / target_dt))
        t_target = np.linspace(0.0, duration, n_target_steps, endpoint=False)
        t_src = np.arange(stimulus.array.shape[0]) * stimulus.dt
        resampled = np.stack(
            [
                np.interp(t_target, t_src, stimulus.array[:, col])
                for col in range(stimulus.array.shape[-1])
            ],
            axis=-1,
        )
        return Stimulus(
            array=resampled, dt=target_dt, gids=stimulus.gids, **stimulus.meta_data
        )

    def to_array(self, duration: float, dt: float):
        """Resample and pad/trim to simulation time grid.

        Returns an array with ``int(duration / dt) + 1`` rows, or None when no waveform data is present.
        Compatible with JAX tracers inside JIT.
        """
        if self.array is None:
            return None

        arr = np.asarray(self.array)
        expected_steps = int(duration / dt) + 1
        original_ndim = arr.ndim

        if original_ndim == 1:
            arr = arr[:, None]

        if not _USES_JAX:
            if arr.shape[0] != expected_steps or not np.isclose(self.dt, dt):
                time_src = np.arange(arr.shape[0]) * self.dt
                time_target = np.linspace(0.0, duration, expected_steps)
                arr = np.stack(
                    [
                        np.interp(time_target, time_src, arr[:, col])
                        for col in range(arr.shape[1])
                    ],
                    axis=1,
                )

        if arr.shape[0] < expected_steps:
            pad = np.zeros(
                (expected_steps - arr.shape[0], arr.shape[1]), dtype=arr.dtype
            )
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > expected_steps:
            arr = arr[:expected_steps]

        if original_ndim == 1:
            arr = arr[:, 0]

        return arr

    def tree_flatten(self):
        children = [self.array]
        aux = (self.dt, self.gids, self.meta_data)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, gids, meta_data = aux
        return cls(array=children[0], dt=dt, gids=gids, **meta_data)


def _register_pytree():
    """Register Stimulus as a JAX pytree if JAX is available."""
    try:
        import jax

        jax.tree_util.register_pytree_node(
            Stimulus,
            Stimulus.tree_flatten,
            Stimulus.tree_unflatten,
        )
    except ImportError:
        pass


_register_pytree()
