import numpy as np
import pytest

from livn.stimulus import Stimulus


class TestStimulus:
    def test_init_default(self):
        stim = Stimulus()
        assert stim.array is None
        assert stim.dt == 1.0
        assert stim.gids is None

    def test_init_with_array(self):
        arr = np.zeros((100, 10), dtype=np.float32)
        stim = Stimulus(arr, dt=0.1)
        assert stim.array is arr
        assert stim.dt == 0.1
        assert len(stim) == 10

    def test_init_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            Stimulus(dt=-1.0)

    def test_init_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            Stimulus(dt=0.0)

    def test_init_with_metadata(self):
        stim = Stimulus(foo="bar", baz=123)
        assert stim.meta_data["foo"] == "bar"
        assert stim.meta_data["baz"] == 123

    def test_duration_empty(self):
        stim = Stimulus()
        assert stim.duration == 0.0

    def test_duration_with_array(self):
        arr = np.zeros((100, 10), dtype=np.float32)
        stim = Stimulus(arr, dt=0.1)
        assert stim.duration == pytest.approx(10.0)

    def test_duration_with_different_dt(self):
        arr = np.zeros((200, 5), dtype=np.float32)
        stim = Stimulus(arr, dt=0.05)
        assert stim.duration == pytest.approx(10.0)  # 200 * 0.05

    def test_len(self):
        arr = np.zeros((100, 64), dtype=np.float32)
        stim = Stimulus(arr)
        assert len(stim) == 64

    def test_from_arg_stimulus(self):
        original = Stimulus(dt=0.5)
        result = Stimulus.from_arg(original)
        assert result is original

    def test_from_arg_none(self):
        result = Stimulus.from_arg(None)
        assert isinstance(result, Stimulus)
        assert result.array is None

    def test_from_arg_array(self):
        arr = np.zeros((100, 10), dtype=np.float32)
        result = Stimulus.from_arg(arr)
        assert isinstance(result, Stimulus)
        assert result.array is arr

    def test_from_arg_tuple(self):
        arr = np.zeros((100, 10), dtype=np.float32)
        result = Stimulus.from_arg((arr, 0.5))
        assert result.array is arr
        assert result.dt == 0.5

    def test_from_arg_dict(self):
        arr = np.zeros((100, 10), dtype=np.float32)
        result = Stimulus.from_arg({"array": arr, "dt": 0.25})
        assert result.array is arr
        assert result.dt == 0.25

    def test_from_arg_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid stimulus"):
            Stimulus.from_arg("invalid")


class TestBiphasicPulse:
    def test_single_pulse_default(self):
        stim = Stimulus.biphasic_pulse(n_channels=64, channels=[0])

        assert stim.array is not None
        assert stim.array.shape[1] == 64
        assert stim.dt == 0.05

        assert stim.meta_data["kind"] == "biphasic_pulse"
        assert stim.meta_data["pulse_times"] == [0.0]
        assert stim.meta_data["amplitude"] == 1.5
        assert stim.meta_data["channels"] == [0]
        assert stim.meta_data["cathodic_first"] is True

    def test_single_pulse_duration(self):
        """Test single pulse duration calculation."""
        stim = Stimulus.biphasic_pulse(
            n_channels=64,
            channels=[0],
            phase_duration=0.2,
            interphase_gap=0.05,
        )
        assert stim.duration == pytest.approx(0.45, rel=0.1)

    def test_pulse_train(self):
        stim = Stimulus.biphasic_pulse(
            n_channels=64,
            channels=[0],
            pulse_times=[0, 10, 20, 30, 40],
        )

        assert len(stim.meta_data["pulse_times"]) == 5
        assert stim.duration == pytest.approx(40.45, rel=0.1)

    def test_cathodic_phase(self):
        stim = Stimulus.biphasic_pulse(
            n_channels=4,
            channels=[0],
            amplitude=1.0,
            dt=0.1,
            phase_duration=0.2,
            interphase_gap=0.1,
            cathodic_first=True,
        )

        arr = stim.array
        first_phase_values = arr[0:2, 0]
        assert np.all(first_phase_values < 0)

        second_phase_start = int((0.2 + 0.1) / 0.1)
        second_phase_values = arr[second_phase_start : second_phase_start + 2, 0]
        assert np.all(second_phase_values > 0)

        stim = Stimulus.biphasic_pulse(
            n_channels=4,
            channels=[0],
            amplitude=1.0,
            dt=0.1,
            phase_duration=0.2,
            interphase_gap=0.1,
            cathodic_first=False,
        )

        arr = stim.array
        first_phase_values = arr[0:2, 0]
        assert np.all(first_phase_values > 0)

        second_phase_start = int((0.2 + 0.1) / 0.1)
        second_phase_values = arr[second_phase_start : second_phase_start + 2, 0]
        assert np.all(second_phase_values < 0)

    def test_multiple_channels(self):
        stim = Stimulus.biphasic_pulse(
            n_channels=64,
            channels=[0, 1, 2, 3],
            amplitude=2.0,
        )

        arr = stim.array
        assert np.any(arr[:, 0] != 0)
        assert np.any(arr[:, 1] != 0)
        assert np.any(arr[:, 2] != 0)
        assert np.any(arr[:, 3] != 0)

        assert np.all(arr[:, 4] == 0)
        assert np.all(arr[:, 63] == 0)

    def test_amplitude_scaling(self):
        stim1 = Stimulus.biphasic_pulse(n_channels=4, channels=[0], amplitude=1.0)
        stim2 = Stimulus.biphasic_pulse(n_channels=4, channels=[0], amplitude=2.0)

        max1 = np.abs(stim1.array).max()
        max2 = np.abs(stim2.array).max()

        assert max2 == pytest.approx(2 * max1)

    def test_charge_balance(self):
        stim = Stimulus.biphasic_pulse(
            n_channels=4,
            channels=[0],
            amplitude=1.5,
            phase_duration=0.2,
            interphase_gap=0.05,
        )
        total_charge = np.sum(stim.array[:, 0])
        assert total_charge == pytest.approx(0.0, abs=1e-6)

    def test_numpy_array_channels(self):
        channels = np.array([0, 1, 2])
        stim = Stimulus.biphasic_pulse(n_channels=64, channels=channels)

        assert stim.meta_data["channels"] == [0, 1, 2]
        assert np.any(stim.array[:, 0] != 0)
        assert np.any(stim.array[:, 1] != 0)
        assert np.any(stim.array[:, 2] != 0)

    def test_custom_dt(self):
        stim = Stimulus.biphasic_pulse(
            n_channels=4,
            channels=[0],
            dt=0.01,
        )
        assert stim.dt == 0.01
        assert stim.array.shape[0] > 10

    def test_interphase_gap(self):
        stim = Stimulus.biphasic_pulse(
            n_channels=4,
            channels=[0],
            dt=0.05,
            phase_duration=0.2,
            interphase_gap=0.1,  # 2 steps at dt=0.05
        )

        arr = stim.array
        gap_start = int(0.2 / 0.05)
        gap_end = int((0.2 + 0.1) / 0.05)
        gap_values = arr[gap_start:gap_end, 0]
        assert np.all(gap_values == 0)


class TestMonophasicPulse:
    def test_single_pulse_defaults(self):
        stim = Stimulus.monophasic_pulse(n_channels=64, channels=[0])

        assert stim.array is not None
        assert stim.array.shape[1] == 64
        assert stim.dt == 1.0
        assert stim.meta_data["kind"] == "monophasic_pulse"
        assert stim.meta_data["pulse_times"] == [0.0]
        assert stim.meta_data["pulse_width"] == 1.0
        assert stim.meta_data["channels"] == [0]

    def test_single_pulse_duration(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=4,
            channels=[0],
            pulse_width=10.0,
            dt=1.0,
        )
        assert stim.duration == pytest.approx(10.0)

    def test_pulse_waveform_shape(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=4,
            channels=[0],
            amplitude=2.0,
            pulse_width=5.0,
            pulse_times=[0.0],
            dt=1.0,
        )
        arr = stim.array
        assert np.all(arr[0:5, 0] == pytest.approx(2.0))
        assert np.all(arr[:, 1:] == 0)

    def test_pulse_train_multiple_times(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=4,
            channels=[0],
            amplitude=1.0,
            pulse_width=5.0,
            pulse_times=[0.0, 25.0, 50.0],
            dt=1.0,
        )
        assert len(stim.meta_data["pulse_times"]) == 3
        assert stim.duration == pytest.approx(55.0)
        arr = stim.array
        for onset in [0, 25, 50]:
            assert np.all(arr[onset : onset + 5, 0] == pytest.approx(1.0))
        # gaps between pulses should be zero
        assert np.all(arr[5:25, 0] == 0)

    def test_multiple_channels_scalar_amplitude(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=64,
            channels=[0, 1, 2, 3],
            amplitude=1.5,
        )
        arr = stim.array
        for c in range(4):
            assert np.any(arr[:, c] != 0)
        assert np.all(arr[:, 4] == 0)
        assert np.all(arr[:, 63] == 0)

    def test_per_channel_amplitude(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=4,
            channels=[0, 1, 2],
            amplitude=[1.0, 2.0, 3.0],
            pulse_width=5.0,
            dt=1.0,
        )
        arr = stim.array
        assert arr[0, 0] == pytest.approx(1.0)
        assert arr[0, 1] == pytest.approx(2.0)
        assert arr[0, 2] == pytest.approx(3.0)
        assert stim.meta_data["amplitude"] == pytest.approx([1.0, 2.0, 3.0])

    def test_zero_amplitude_channel_suppressed(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=4,
            channels=[0, 1],
            amplitude=[1.0, 0.0],
            pulse_width=5.0,
            dt=1.0,
        )
        arr = stim.array
        assert np.any(arr[:, 0] != 0)
        assert np.all(arr[:, 1] == 0)

    def test_numpy_array_channels(self):
        channels = np.array([2, 5])
        stim = Stimulus.monophasic_pulse(n_channels=16, channels=channels)
        assert stim.meta_data["channels"] == [2, 5]
        assert np.any(stim.array[:, 2] != 0)
        assert np.any(stim.array[:, 5] != 0)
        assert np.all(stim.array[:, 0] == 0)

    def test_custom_dt(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=4,
            channels=[0],
            pulse_width=0.2,
            dt=0.05,
        )
        assert stim.dt == 0.05
        # 0.2 ms / 0.05 ms = 4
        assert np.sum(stim.array[:, 0] != 0) == 4

    def test_unstimulated_channels_zero(self):
        stim = Stimulus.monophasic_pulse(
            n_channels=16,
            channels=[3],
            amplitude=2.0,
        )
        arr = stim.array
        for c in range(16):
            if c != 3:
                assert np.all(arr[:, c] == 0)
