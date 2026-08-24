import numpy as np
import pytest

from livn.policy import (
    BiphasicPulsePolicy,
    MonophasicPulsePolicy,
    PulseSweepPolicy,
)
from livn.stimulus import Stimulus


class TestStimulus:
    def test_the_array_is_kept_as_given_and_read_as_time_by_channel(self):
        array = np.zeros((100, 10), dtype=np.float32)
        stimulus = Stimulus(array, dt=0.1)

        assert stimulus.array is array, "the array was copied"
        assert len(stimulus) == 10, "len() is the channel count, not the samples"
        assert stimulus.duration == pytest.approx(10.0)

    def test_duration_follows_the_sample_rate(self):
        coarse = Stimulus(np.zeros((100, 10), dtype=np.float32), dt=0.1)
        fine = Stimulus(np.zeros((200, 5), dtype=np.float32), dt=0.05)

        assert coarse.duration == fine.duration == pytest.approx(10.0)

    @pytest.mark.parametrize("dt", [0.0, -1.0])
    def test_a_dt_that_is_not_a_duration_is_refused(self, dt):
        with pytest.raises(ValueError, match="dt must be positive"):
            Stimulus(np.zeros((1, 1)), dt=dt)

    def test_the_mode_the_units_and_anything_else_are_carried(self):
        stimulus = Stimulus(
            np.zeros((10, 4), dtype=np.float32),
            input_mode="current",
            units="nA",
            wavelength_nm=473.0,
        )

        assert stimulus.input_mode == "current"
        assert stimulus.units == "nA"
        assert stimulus.extra["wavelength_nm"] == 473.0

    def test_from_arg_accepts_each_form_a_caller_might_have(self):
        array = np.zeros((100, 10), dtype=np.float32)
        existing = Stimulus(array, dt=0.5)

        assert Stimulus.from_arg(None) is None
        assert Stimulus.from_arg(existing) is existing, "a Stimulus was rebuilt"
        assert Stimulus.from_arg(array).array is array
        assert Stimulus.from_arg((array, 0.5)).dt == 0.5
        assert Stimulus.from_arg({"array": array, "dt": 0.25}).dt == 0.25

    def test_from_arg_refuses_what_it_cannot_read(self):
        with pytest.raises(ValueError, match="Invalid stimulus"):
            Stimulus.from_arg("invalid")

    @pytest.mark.parametrize(
        ("constructor", "mode", "units"),
        [
            ("from_conductance", "conductance", "uS"),
            ("from_current", "current", "nA"),
            ("from_irradiance", "irradiance", "mW/mm2"),
        ],
    )
    def test_a_named_constructor_sets_the_mode_and_its_canonical_unit(
        self, constructor, mode, units
    ):
        array = np.random.rand(100, 5).astype(np.float32)
        stimulus = getattr(Stimulus, constructor)(array, dt=0.1)

        assert stimulus.input_mode == mode
        assert stimulus.units == units

    def test_extra_metadata_survives_a_named_constructor(self):
        stimulus = Stimulus.from_irradiance(
            np.random.rand(100, 3).astype(np.float32), dt=0.1, wavelength_nm=561.0
        )

        assert stimulus.extra["wavelength_nm"] == 561.0


class TestBiphasicPulse:
    def test_single_pulse_default(self):
        policy = BiphasicPulsePolicy(n_channels=64, channels=[0])
        arr = policy()

        assert arr is not None
        assert arr.shape[1] == 64
        assert policy.dt == 0.05
        assert policy.pulse_times == [0.0]
        assert policy.amplitude == 1.5
        assert policy.channels == [0]
        assert policy.cathodic_first is True

    def test_single_pulse_duration(self):
        policy = BiphasicPulsePolicy(
            n_channels=64,
            channels=[0],
            phase_duration=0.2,
            interphase_gap=0.05,
        )
        arr = policy()
        assert arr.shape[0] * policy.dt == pytest.approx(0.45, rel=0.1)

    def test_pulse_train(self):
        policy = BiphasicPulsePolicy(
            n_channels=64,
            channels=[0],
            pulse_times=[0, 10, 20, 30, 40],
        )
        assert len(policy.pulse_times) == 5
        arr = policy()
        assert arr.shape[0] * policy.dt == pytest.approx(40.45, rel=0.1)

    def test_cathodic_phase(self):
        policy = BiphasicPulsePolicy(
            n_channels=4,
            channels=[0],
            amplitude=1.0,
            dt=0.1,
            phase_duration=0.2,
            interphase_gap=0.1,
            cathodic_first=True,
        )
        arr = policy()
        first_phase_values = arr[0:2, 0]
        assert np.all(first_phase_values < 0)

        second_phase_start = int((0.2 + 0.1) / 0.1)
        second_phase_values = arr[second_phase_start : second_phase_start + 2, 0]
        assert np.all(second_phase_values > 0)

        policy2 = BiphasicPulsePolicy(
            n_channels=4,
            channels=[0],
            amplitude=1.0,
            dt=0.1,
            phase_duration=0.2,
            interphase_gap=0.1,
            cathodic_first=False,
        )
        arr2 = policy2()
        assert np.all(arr2[0:2, 0] > 0)
        assert np.all(arr2[second_phase_start : second_phase_start + 2, 0] < 0)

    def test_multiple_channels(self):
        arr = BiphasicPulsePolicy(
            n_channels=64,
            channels=[0, 1, 2, 3],
            amplitude=2.0,
        )()
        for c in range(4):
            assert np.any(arr[:, c] != 0)
        assert np.all(arr[:, 4] == 0)
        assert np.all(arr[:, 63] == 0)

    def test_amplitude_scaling(self):
        arr1 = BiphasicPulsePolicy(n_channels=4, channels=[0], amplitude=1.0)()
        arr2 = BiphasicPulsePolicy(n_channels=4, channels=[0], amplitude=2.0)()
        assert np.abs(arr2).max() == pytest.approx(2 * np.abs(arr1).max())

    def test_charge_balance(self):
        arr = BiphasicPulsePolicy(
            n_channels=4,
            channels=[0],
            amplitude=1.5,
            phase_duration=0.2,
            interphase_gap=0.05,
        )()
        assert np.sum(arr[:, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_numpy_array_channels(self):
        channels = np.array([0, 1, 2])
        policy = BiphasicPulsePolicy(n_channels=64, channels=channels)
        arr = policy()
        assert policy.channels == [0, 1, 2]
        for c in range(3):
            assert np.any(arr[:, c] != 0)

    def test_custom_dt(self):
        policy = BiphasicPulsePolicy(n_channels=4, channels=[0], dt=0.01)
        arr = policy()
        assert policy.dt == 0.01
        assert arr.shape[0] > 10

    def test_interphase_gap(self):
        policy = BiphasicPulsePolicy(
            n_channels=4,
            channels=[0],
            dt=0.05,
            phase_duration=0.2,
            interphase_gap=0.1,
        )
        arr = policy()
        gap_start = int(0.2 / 0.05)
        gap_end = int((0.2 + 0.1) / 0.05)
        assert np.all(arr[gap_start:gap_end, 0] == 0)

    def test_serialize_roundtrip(self):
        policy = BiphasicPulsePolicy(
            n_channels=8,
            channels=[2, 3],
            amplitude=2.0,
            phase_duration=0.3,
            interphase_gap=0.1,
            pulse_times=[0.0, 20.0],
            dt=0.05,
            cathodic_first=False,
        )
        restored = BiphasicPulsePolicy.from_json(policy.as_json())
        np.testing.assert_array_equal(policy(), restored())


class TestMonophasicPulse:
    def test_single_pulse_defaults(self):
        policy = MonophasicPulsePolicy(n_channels=64, channels=[0])
        arr = policy()

        assert arr is not None
        assert arr.shape[1] == 64
        assert policy.dt == 1.0
        assert policy.pulse_times == [0.0]
        assert policy.pulse_width == 1.0
        assert policy.channels == [0]

    def test_single_pulse_duration(self):
        policy = MonophasicPulsePolicy(
            n_channels=4, channels=[0], pulse_width=10.0, dt=1.0
        )
        arr = policy()
        assert arr.shape[0] * policy.dt == pytest.approx(10.0)

    def test_pulse_waveform_shape(self):
        arr = MonophasicPulsePolicy(
            n_channels=4,
            channels=[0],
            amplitude=2.0,
            pulse_width=5.0,
            pulse_times=[0.0],
            dt=1.0,
        )()
        assert np.all(arr[0:5, 0] == pytest.approx(2.0))
        assert np.all(arr[:, 1:] == 0)

    def test_pulse_train_multiple_times(self):
        policy = MonophasicPulsePolicy(
            n_channels=4,
            channels=[0],
            amplitude=1.0,
            pulse_width=5.0,
            pulse_times=[0.0, 25.0, 50.0],
            dt=1.0,
        )
        assert len(policy.pulse_times) == 3
        arr = policy()
        assert arr.shape[0] * policy.dt == pytest.approx(55.0)
        for onset in [0, 25, 50]:
            assert np.all(arr[onset : onset + 5, 0] == pytest.approx(1.0))
        assert np.all(arr[5:25, 0] == 0)

    def test_multiple_channels_scalar_amplitude(self):
        arr = MonophasicPulsePolicy(
            n_channels=64, channels=[0, 1, 2, 3], amplitude=1.5
        )()
        for c in range(4):
            assert np.any(arr[:, c] != 0)
        assert np.all(arr[:, 4] == 0)
        assert np.all(arr[:, 63] == 0)

    def test_per_channel_amplitude(self):
        arr = MonophasicPulsePolicy(
            n_channels=4,
            channels=[0, 1, 2],
            amplitude=[1.0, 2.0, 3.0],
            pulse_width=5.0,
            dt=1.0,
        )()
        assert arr[0, 0] == pytest.approx(1.0)
        assert arr[0, 1] == pytest.approx(2.0)
        assert arr[0, 2] == pytest.approx(3.0)

    def test_zero_amplitude_channel_suppressed(self):
        arr = MonophasicPulsePolicy(
            n_channels=4, channels=[0, 1], amplitude=[1.0, 0.0], pulse_width=5.0, dt=1.0
        )()
        assert np.any(arr[:, 0] != 0)
        assert np.all(arr[:, 1] == 0)

    def test_numpy_array_channels(self):
        channels = np.array([2, 5])
        policy = MonophasicPulsePolicy(n_channels=16, channels=channels)
        arr = policy()
        assert policy.channels == [2, 5]
        assert np.any(arr[:, 2] != 0)
        assert np.any(arr[:, 5] != 0)
        assert np.all(arr[:, 0] == 0)

    def test_custom_dt(self):
        policy = MonophasicPulsePolicy(
            n_channels=4, channels=[0], pulse_width=0.2, dt=0.05
        )
        arr = policy()
        assert policy.dt == 0.05
        assert np.sum(arr[:, 0] != 0) == 4

    def test_unstimulated_channels_zero(self):
        arr = MonophasicPulsePolicy(n_channels=16, channels=[3], amplitude=2.0)()
        for c in range(16):
            if c != 3:
                assert np.all(arr[:, c] == 0)

    def test_serialize_roundtrip(self):
        policy = MonophasicPulsePolicy(
            n_channels=4,
            channels=[0, 2],
            amplitude=[1.5, 3.0],
            pulse_width=5.0,
            pulse_times=[0.0, 30.0],
            dt=0.5,
        )
        restored = MonophasicPulsePolicy.from_json(policy.as_json())
        np.testing.assert_array_equal(policy(), restored())


class TestStimulusJax:
    @pytest.fixture(autouse=True)
    def requires_jax(self):
        jax = pytest.importorskip("jax")
        self.jnp = jax.numpy
        self.jax = jax

    def test_array_accepted(self):
        arr = self.jnp.ones((10, 3), dtype=self.jnp.float32)
        stim = Stimulus(arr, dt=0.1)
        assert stim.duration == pytest.approx(1.0)
        assert len(stim) == 3

    def test_to_array(self):
        arr = self.jnp.ones((5, 2), dtype=self.jnp.float32)
        stim = Stimulus(arr, dt=0.1)
        result = stim.to_array(duration=1.0, dt=0.1)
        assert result.shape == (11, 2)

    def test_align_gids(self):
        arr = self.jnp.ones((10, 3), dtype=self.jnp.float32)
        gids = self.jnp.array([0, 2, 4])
        stim = Stimulus(arr, dt=0.1, gids=gids)
        all_gids = self.jnp.array([0, 1, 2, 3, 4])
        result = Stimulus.align_gids(stim, all_gids)
        result_arr = np.asarray(result.array)
        assert result_arr.shape == (10, 5)
        np.testing.assert_allclose(result_arr[:, [0, 2, 4]], 1.0)
        np.testing.assert_allclose(result_arr[:, [1, 3]], 0.0)

    def test_resample(self):
        arr = self.jnp.ones((10, 2), dtype=self.jnp.float32)
        stim = Stimulus(arr, dt=0.1)
        result = Stimulus.resample(stim, target_dt=0.05, duration=1.0)
        assert result.array.shape[0] == 20
        np.testing.assert_allclose(np.asarray(result.array), 1.0, atol=1e-5)

    def _requires_jax_backend(self):
        from livn.stimulus import _USES_JAX

        if not _USES_JAX:
            pytest.skip("Requires JAX backend")

    @pytest.mark.traces
    def test_to_array_jit(self):
        self._requires_jax_backend()
        arr = self.jnp.ones((11, 3), dtype=self.jnp.float32)
        stim = Stimulus(arr, dt=0.1)

        @self.jax.jit
        def f(stim):
            return stim.to_array(duration=1.0, dt=0.1)

        result = f(stim)
        assert result.shape == (11, 3)
        np.testing.assert_allclose(np.asarray(result), 1.0)

    @pytest.mark.traces
    def test_to_array_pad_jit(self):
        self._requires_jax_backend()
        arr = self.jnp.ones((5, 2), dtype=self.jnp.float32)
        stim = Stimulus(arr, dt=0.1)

        @self.jax.jit
        def f(stim):
            return stim.to_array(duration=1.0, dt=0.1)

        result = f(stim)
        assert result.shape == (11, 2)
        np.testing.assert_allclose(np.asarray(result[:5]), 1.0)

    @pytest.mark.traces
    def test_to_array_trim_jit(self):
        self._requires_jax_backend()
        arr = self.jnp.ones((20, 2), dtype=self.jnp.float32)
        stim = Stimulus(arr, dt=0.1)

        @self.jax.jit
        def f(stim):
            return stim.to_array(duration=1.0, dt=0.1)

        result = f(stim)
        assert result.shape == (11, 2)

    @pytest.mark.traces
    def test_to_array_1d_jit(self):
        self._requires_jax_backend()
        arr = self.jnp.ones(11, dtype=self.jnp.float32)
        stim = Stimulus(arr, dt=0.1)

        @self.jax.jit
        def f(stim):
            return stim.to_array(duration=1.0, dt=0.1)

        result = f(stim)
        assert result.shape == (11,)
        np.testing.assert_allclose(np.asarray(result), 1.0)


class TestPulseSweep:
    def test_schedule_cycles_the_amplitudes(self):
        policy = PulseSweepPolicy(
            amplitudes=(300.0, 600.0), repeats=2, trial_ms=2000.0, onset_ms=1000.0
        )
        assert policy.n_trials == 4
        assert policy.duration_ms == 8000.0
        assert policy.schedule() == [
            (1000.0, 300.0),
            (3000.0, 600.0),
            (5000.0, 300.0),
            (7000.0, 600.0),
        ]

    def test_the_schedule_is_absolute_so_a_run_can_start_free_running(self):
        policy = PulseSweepPolicy(
            amplitudes=(1.0,), repeats=2, trial_ms=100.0, onset_ms=50.0
        )
        assert [t for t, _ in policy.schedule(1000.0)] == [1050.0, 1150.0]

        moved = policy.model_copy(update={"start_ms": 1000.0})
        assert moved.schedule() == policy.schedule(1000.0)

    def test_an_explicit_order_overrides_the_cycle(self):
        policy = PulseSweepPolicy(
            amplitudes=(1.0, 2.0),
            repeats=2,
            order=(1, 1, 0, 0),
            trial_ms=100.0,
            onset_ms=50.0,
        )
        assert [a for _, a in policy.schedule()] == [2.0, 2.0, 1.0, 1.0]

    def test_an_order_that_does_not_match_the_sweep_is_refused(self):
        with pytest.raises(ValueError, match="order names 3 trials"):
            PulseSweepPolicy(amplitudes=(1.0, 2.0), repeats=2, order=(0, 1, 0))
        with pytest.raises(ValueError, match="order names amplitude 5"):
            PulseSweepPolicy(amplitudes=(1.0, 2.0), repeats=2, order=(5, 0, 1, 0))

    def test_a_pulse_that_does_not_fit_its_trial_is_refused(self):
        with pytest.raises(ValueError, match="outside it"):
            PulseSweepPolicy(trial_ms=1000.0, onset_ms=1000.0)
        with pytest.raises(ValueError, match="does not fit inside its trial"):
            PulseSweepPolicy(trial_ms=1000.0, onset_ms=999.9, pulse_ms=1.0)
        with pytest.raises(ValueError, match="at least one amplitude"):
            PulseSweepPolicy(amplitudes=())

    def test_the_array_carries_each_pulse_at_its_own_amplitude(self):
        policy = PulseSweepPolicy(
            amplitudes=(300.0, 600.0),
            repeats=1,
            trial_ms=100.0,
            onset_ms=50.0,
            pulse_ms=0.2,
            dt=0.1,
        ).for_array(8, [3])
        arr = policy()

        assert arr.shape == (2000, 8)
        assert sorted(set(np.nonzero(arr)[1].tolist())) == [3]
        assert sorted(set(np.abs(arr[arr != 0]).tolist())) == [300.0, 600.0]
        assert np.sum(arr[:, 3]) == pytest.approx(0.0, abs=1e-3)
        assert arr[500, 3] == pytest.approx(-300.0)
        assert arr[501, 3] == pytest.approx(300.0)

    def test_a_sweep_without_an_array_says_so_rather_than_guessing(self):
        policy = PulseSweepPolicy(amplitudes=(1.0,))
        assert policy.schedule()
        with pytest.raises(ValueError, match="no array to drive"):
            policy()

    def test_a_pulse_past_the_end_of_the_run_is_refused(self):
        policy = PulseSweepPolicy(
            amplitudes=(1.0,), repeats=4, trial_ms=100.0, onset_ms=50.0
        ).for_array(4, [0], total_ms=120.0)
        with pytest.raises(ValueError, match="does not fit in a 120 ms run"):
            policy()

    def test_channels_outside_the_array_are_refused(self):
        with pytest.raises(ValueError, match="outside a 4-channel array"):
            PulseSweepPolicy(amplitudes=(1.0,)).for_array(4, [9])

    def test_serialize_roundtrip(self):
        policy = PulseSweepPolicy(
            amplitudes=(300.0, 600.0),
            repeats=2,
            trial_ms=200.0,
            onset_ms=100.0,
            order=(1, 0, 0, 1),
            start_ms=50.0,
            dt=0.1,
        ).for_array(8, [2])
        restored = PulseSweepPolicy.from_json(policy.as_json())
        assert restored.schedule() == policy.schedule()
        np.testing.assert_array_equal(policy(), restored())
