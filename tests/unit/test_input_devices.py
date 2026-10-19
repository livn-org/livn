import json

import numpy as np
import pytest

from livn.conditions import Conditions
from livn.io import (
    MEA,
    Chain,
    ComposedIO,
    ElectrodeRC,
    ExtracellularPotential,
    InputDevices,
    LightArtifact,
    PatternedLight,
    Sum,
)
from livn.io.recording import electrode_light
from livn.policy import Policy
from livn.stimulus import ComposedStimulus, Stimulus

DT = 0.1
ELECTRODES = [[0, 0.0, 0.0, 0.0], [1, 300.0, 0.0, 0.0], [2, 900.0, 0.0, 0.0]]
CELLS = np.asarray([[0, 10.0, 0.0, 5.0], [1, 290.0, 5.0, 5.0], [2, 600.0, 0.0, 5.0]])


def _lasers(**kwargs):
    # two lasers on overlapping spots
    first = PatternedLight([[0, 0.0, 0.0, 80.0], [1, 300.0, 0.0, 80.0]], 2.0)
    second = PatternedLight([[0, 50.0, 0.0, 80.0]], 1.5)
    return ComposedIO(
        inputs=[first, second], outputs=MEA(ELECTRODES, output_radius=2000.0), **kwargs
    )


def _commands(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(size=(n, 3)) > 0.6).astype(np.float32)


def test_the_devices_share_the_medium_and_concatenate_their_channels():
    io = _lasers()

    assert isinstance(io.inputs, InputDevices)
    assert io.stimulating_device() is io.inputs
    assert io.inputs.sizes == (2, 1) and io.inputs.num_channels == 3
    np.testing.assert_array_equal(io.inputs.channel_ids, [0, 1, 2])
    assert all(d.medium is io.medium for d in io.inputs.devices)
    assert io.input_units == "normalized"
    with pytest.raises(ValueError, match=r"devices of \[2, 1\] channels"):
        io.inputs.split(np.zeros((5, 2)))


def test_cell_stimulus_is_the_composite_of_each_devices_stimulus():
    io = _lasers()
    commands = _commands()

    stimulus = io.cell_stimulus(CELLS, commands, dt=DT)

    first = io.inputs.devices[0]
    assert isinstance(stimulus, ComposedStimulus)
    assert len(stimulus.parts) == 2
    want = first.cell_stimulus(CELLS, commands[:, :2], dt=DT)
    np.testing.assert_array_equal(stimulus.parts[0].array, want.array)
    np.testing.assert_array_equal(stimulus.parts[0].gids, want.gids)


def test_two_lasers_on_one_spot_are_one_laser_with_the_summed_commands():
    spot = [[0, 0.0, 0.0, 80.0]]
    io = ComposedIO(
        inputs=[PatternedLight(spot, 2.0), PatternedLight(spot, 2.0)],
        outputs=MEA(ELECTRODES),
    )
    one = PatternedLight(spot, 2.0)
    a, b = _commands(seed=1)[:, :1], _commands(seed=2)[:, :1]

    together = io.cell_stimulus(CELLS, np.concatenate([a, b], axis=1), dt=DT)
    summed = one.cell_stimulus(CELLS, a + b, dt=DT)

    light = together.groups()["irradiance"]
    np.testing.assert_array_equal(light.gids, summed.gids)
    np.testing.assert_allclose(light.array, summed.array, rtol=1e-6, atol=1e-7)
    assert light.extra["wavelength"] == summed.extra["wavelength"]


def test_lasers_of_two_wavelengths_cannot_drive_one_opsin():
    io = ComposedIO(
        inputs=[
            PatternedLight([[0, 0.0, 0.0, 80.0]], wavelength=470.0),
            PatternedLight([[0, 0.0, 0.0, 80.0]], wavelength=590.0),
        ],
        outputs=MEA(ELECTRODES),
    )
    with pytest.raises(ValueError, match="470 and 590 nm"):
        io.cell_stimulus(CELLS, np.ones((10, 2)), dt=DT).groups()


def test_the_artifact_sees_the_light_of_every_laser():
    artifact = LightArtifact([30.0, -10.0], taus=(0.5, 5.0))
    io = _lasers(recording=Chain(artifact=artifact, rc=ElectrodeRC(20.0)))
    commands = _commands()
    first, second = io.inputs.devices
    coordinates = io.outputs.electrode_coordinates

    got = io.observe(Conditions(inputs=commands, dt=DT))

    light = electrode_light(commands[:, :2], first.reach(coordinates)) + (
        electrode_light(commands[:, 2:], second.reach(coordinates))
    )
    want = ElectrodeRC(20.0)(artifact.response(light, dt=DT), dt=DT)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert np.abs(got).max() > 1.0


def test_only_light_sources_light_the_electrodes():
    mea = MEA(ELECTRODES, output_radius=2000.0)
    light = PatternedLight([[0, 0.0, 0.0, 80.0]], 2.0)
    artifact = LightArtifact([30.0], taus=(0.5,))
    io = ComposedIO(
        inputs=["outputs", light], outputs=mea, recording=Chain(artifact=artifact)
    )
    commands = np.zeros((100, 4), np.float32)
    commands[20:60, 3] = 1.0
    commands[:, :3] = 5.0  # the electrodes' own stimulation is no light

    got = io.observe(Conditions(inputs=commands, dt=DT))

    reach = light.reach(io.outputs.electrode_coordinates)
    want = artifact.response(electrode_light(commands[:, 3:], reach), dt=DT)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert io.inputs.devices[0] is io.outputs


def test_a_composition_with_several_devices_round_trips():
    io = _lasers(
        recording=Chain(
            Sum(extracellular=ExtracellularPotential(), artifact=LightArtifact()),
            rc=ElectrodeRC(20.0),
        )
    )
    document = json.loads(json.dumps(io.serialize()))

    again = ComposedIO(**document)
    assert isinstance(document["inputs"], list) and len(document["inputs"]) == 2
    assert again.serialize() == io.serialize()
    assert all(d.medium is again.medium for d in again.inputs.devices)
    shared = ComposedIO(
        inputs=["outputs", PatternedLight([[0, 0.0, 0.0, 5.0]])], outputs=MEA()
    )
    back = ComposedIO(**json.loads(json.dumps(shared.serialize())))
    assert shared.serialize()["inputs"][0] == "outputs"
    assert back.inputs.devices[0] is back.outputs


def test_a_device_is_addressed_by_its_position():
    io = _lasers()

    io.set_params({"inputs-1-irradiance": 4.0, "medium-optical-width": 7.0})
    assert io.inputs.devices[1].irradiance == 4.0
    assert io.inputs.devices[0].medium.optical.width == 7.0
    with pytest.raises(KeyError, match="medium is shared"):
        io.set_params({"inputs-0-medium-optical-width": 1.0})


class _Commands(Policy):
    dt: float = DT
    seed: int = 0

    def __call__(self, observation=None):
        return _commands(seed=self.seed)


class _Env:
    def __init__(self, io):
        self.io = io

    def cell_stimulus(self, channel_inputs, dt=1.0):
        return self.io.cell_stimulus(CELLS, channel_inputs, dt=dt)


def test_a_policy_through_several_devices_is_one_deferred_part_per_device():
    io = _lasers()
    policy = _Commands()

    stimulus = Stimulus.from_policy(policy, _Env(io), 20.0)

    assert isinstance(stimulus, ComposedStimulus) and stimulus.deferred
    whole = io.cell_stimulus(CELLS, policy.window(0.0, 5.0, DT), dt=DT)
    for part, want in zip(stimulus.parts, whole.parts, strict=True):
        assert part.deferred
        np.testing.assert_array_equal(part.window(0.0, 5.0), want.array)
