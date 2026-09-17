import numpy as np
import pytest

from livn.encoding import ElectrodeStimulus
from livn.policy import BiphasicPulsePolicy, PulseSweepPolicy


class _Array:
    def __init__(self, n_channels=8, strongest=5):
        self.io = _Channels(n_channels)
        self._strongest = strongest
        self._n_channels = n_channels

    def channel_reach(self, *args, **kwargs):
        reach = np.zeros((self._n_channels, 3))
        reach[self._strongest] = 1.0
        return reach


class _Channels:
    def __init__(self, n_channels):
        self.channel_ids = list(range(n_channels))


class _NoArray:
    """A graph that bundles no array at all."""

    class io:
        @property
        def channel_ids(self):
            raise NotImplementedError

    io = io()

    def channel_reach(self, *args, **kwargs):
        raise NotImplementedError


def sweep(**overrides):
    settings = {
        "amplitudes": (1.0,),
        "repeats": 1,
        "trial_ms": 100.0,
        "onset_ms": 50.0,
        "pulse_ms": 0.2,
        "dt": 0.1,
    }
    settings.update(overrides)
    return PulseSweepPolicy(**settings)


def test_an_explicit_channel_is_the_one_driven():
    encoding = ElectrodeStimulus(channel=2)
    resolved = encoding(_Array(), 100.0, sweep())

    assert resolved.channels == [2]
    assert resolved.n_channels == 8


def test_a_list_of_channels_drives_all_of_them():
    encoding = ElectrodeStimulus(channel=[1, 4, 6])
    policy = sweep(amplitudes=((10.0, 20.0, 30.0),))
    resolved = encoding(_Array(), 100.0, policy)

    assert resolved.channels == [1, 4, 6]
    assert sorted(set(np.nonzero(resolved())[1].tolist())) == [1, 4, 6]


def test_a_policy_that_names_its_own_channels_keeps_them():
    policy = sweep(channels=[3, 7])
    resolved = ElectrodeStimulus()(_Array(strongest=5), 100.0, policy)

    assert resolved.channels == [3, 7]


def test_a_policy_that_names_none_takes_the_strongest_coupling():
    resolved = ElectrodeStimulus()(_Array(strongest=5), 100.0, sweep())

    assert resolved.channels == [5]


def test_an_explicit_channel_overrules_the_policy():
    policy = sweep(channels=[3, 7])
    resolved = ElectrodeStimulus(channel=0)(_Array(), 100.0, policy)

    assert resolved.channels == [0]


def test_the_policy_is_stretched_to_fill_the_run():
    resolved = ElectrodeStimulus(channel=1)(_Array(), 250.0, sweep())

    assert resolved.total_ms == 250.0
    assert len(resolved()) == 2500


def test_a_policy_without_a_length_of_its_own_is_left_alone():
    policy = BiphasicPulsePolicy(n_channels=8, channels=[1], pulse_times=[0.0])
    resolved = ElectrodeStimulus()(_Array(), 100.0, policy)

    assert resolved.channels == [1]
    assert "total_ms" not in type(resolved).model_fields


def test_the_resolved_policy_is_what_was_delivered():
    encoding = ElectrodeStimulus(channel=2)
    assert encoding.resolved is None

    resolved = encoding(_Array(), 100.0, sweep())
    assert encoding.resolved is resolved


def test_an_encoding_with_no_policy_says_so():
    with pytest.raises(ValueError, match="no policy to deliver"):
        ElectrodeStimulus()(_Array(), 100.0, None)


def test_a_run_with_no_array_says_so():
    with pytest.raises(ValueError, match="no array to stimulate through"):
        ElectrodeStimulus()(_NoArray(), 100.0, sweep())
