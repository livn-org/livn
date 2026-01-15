import numpy as np

from livn.types import Array, Float, Int


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
            inputs[phase1_start:phase1_end, channels] = phase1_amp

            phase2_start = onset_step + phase1_steps + gap_steps
            phase2_end = phase2_start + phase2_steps
            if phase2_end <= n_steps:
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
