import dataclasses
import json

import numpy as np
import pytest

from livn.backend import backend
from livn.conditions import Conditions
from livn.io import (
    FIR,
    MEA,
    Bandpass,
    Chain,
    ComposedIO,
    ElectrodeRC,
    ExtracellularPotential,
    LightArtifact,
    Noise,
    PatternedLight,
    Sum,
)
from livn.io.recording import is_stack, stack
from livn.io.transfer import stage_key
from livn.types import Env
from livn.utils import import_instance

DT = 0.1
ELECTRODES = [[0, 0.0, 0.0, 0.0], [1, 400.0, 0.0, 0.0], [2, 1000.0, 0.0, 0.0]]
CELLS = np.asarray(
    [[0, 50.0, 30.0, 5.0], [1, 500.0, -20.0, 5.0], [2, 900.0, 10.0, 5.0]]
)
N = 200


def _currents(seed=1):
    return np.random.default_rng(seed).normal(size=(len(CELLS), N)) * 1e-3


class _Env:
    potential_recording = Env.potential_recording
    fill_conditions = Env.fill_conditions
    recording_rows = Env.recording_rows
    placed_rows = Env.placed_rows
    recording_distances = Env.recording_distances
    membrane_current_recording_dt = DT

    def __init__(self, io, sections=None):
        self.io = io
        self.sections = sections

    def recording_coordinates(self):
        return CELLS

    def recording_sections(self):
        return self.sections


class Offset:
    def __init__(self, value):
        self.value = value

    def __call__(self, *, io, dt=None):
        return np.full((io.num_channels, N), self.value)


class Gain:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, signal, *, dt):
        return self.factor * signal


class Everything:
    def __init__(self):
        self.seen = None

    def __call__(self, **fields):
        self.seen = fields
        return np.zeros((3, N))


class Misspelt:
    def __call__(self, *, temprature):
        return temprature


class RigConditions(Conditions):
    temperature: object = None


class Drift:
    """A source driven by temperature, a field livn does not know about."""

    def __init__(self, gain):
        self.gain = gain

    def __call__(self, *, temperature, io):
        t = np.asarray(temperature)
        return self.gain * np.broadcast_to(t - t[0], (io.num_channels, len(t)))


def _stack_io(*transfers, **kwargs):
    return ComposedIO(
        outputs=[MEA(ELECTRODES, output_radius=2000.0), *transfers], **kwargs
    )


def _today(io, distances, currents, dt, key):
    signal = io.outputs.potential_recording(distances, currents, dt=dt)
    for k, transfer in enumerate(io.transfers):
        signal = transfer(signal, dt=dt, key=stage_key(key, k))
    return signal


def _key(seed):
    if seed is None:
        return None
    if "ax" in backend():
        import jax

        return jax.random.PRNGKey(seed)
    return seed


@pytest.mark.parametrize("seed", [None, 7])
def test_a_plain_stack_is_bit_identical_to_the_device_and_its_transfers(seed):
    key = _key(seed)
    io = _stack_io(ElectrodeRC(20.0), Bandpass(300.0, 3000.0), Noise(2.0))
    distances, currents = io.distances(CELLS), _currents()

    got = io.potential_recording(distances, currents, dt=DT, key=key)

    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(_today(io, distances, currents, DT, key))
    )
    env = _Env(io)
    through_env = env.potential_recording(
        conditions=Conditions(membrane_currents_uA=currents), key=key
    )
    np.testing.assert_array_equal(np.asarray(through_env), np.asarray(got))
    assert is_stack(io.recording)
    assert io.recording.names == ("extracellular", "electrode_rc", "bandpass", "noise")


def test_a_bare_device_records_through_its_own_projection():
    mea = MEA(ELECTRODES, output_radius=2000.0)
    distances, currents = mea.distances(CELLS), _currents()

    got = _Env(mea).potential_recording(
        conditions=Conditions(membrane_currents_uA=currents, gids=CELLS[:, 0])
    )

    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(mea.potential_recording(distances, currents))
    )


def test_a_plain_stack_serializes_as_it_always_did():
    io = _stack_io(ElectrodeRC(20.0), Noise(2.0, seed=3))
    document = io.serialize()

    assert set(document) == {"inputs", "outputs", "medium"}
    assert [d["cls"] for d in document["outputs"]] == [
        "livn.io.MEA",
        "livn.io.ElectrodeRC",
        "livn.io.Noise",
    ]
    assert ComposedIO(**json.loads(json.dumps(document))).serialize() == document
    assert set(ComposedIO(outputs=MEA(ELECTRODES)).serialize()) == {
        "inputs",
        "outputs",
        "medium",
    }


def test_repeated_stages_of_a_stack_are_numbered():
    io = _stack_io(Noise(1.0), Noise(2.0))

    assert io.recording.names == ("extracellular", "noise", "noise_2")
    assert io.outputs.detection_noise_uV == pytest.approx(5**0.5)
    io.set_params({"recording-noise_2-sigma_uV": 3.0})
    assert io.transfers[1].sigma_uV == 3.0


# injection


def test_a_block_gets_the_fields_it_names():
    io = _stack_io()
    recording = Chain(Offset(2.0), gain=Gain(3.0))

    got = io.observe(Conditions(dt=DT), recording)

    np.testing.assert_array_equal(got, np.full((3, N), 6.0))
    assert recording.requires() == {"dt"}
    assert recording.reads_io()


def test_a_required_field_that_is_unset_names_the_block_and_the_field():
    with pytest.raises(ValueError, match=r"'gain' \(Gain\) needs `dt`"):
        Chain(Offset(1.0), gain=Gain(2.0))(Conditions(), io=_stack_io())


def test_an_optional_field_takes_the_block_default():
    # Offset's dt has a default, so it runs without one
    out = Chain(Offset(1.0))(Conditions(), io=_stack_io())
    np.testing.assert_array_equal(out, np.ones((3, N)))


def test_a_block_that_names_io_gets_the_io_it_records_through():
    io = _stack_io()
    currents = _currents()
    conditions = Conditions(membrane_currents_uA=currents, neuron_coordinates=CELLS)

    with pytest.raises(
        ValueError, match=r"'extracellular' \(ExtracellularPotential\) needs the io"
    ):
        Chain(extracellular=ExtracellularPotential())(conditions)
    np.testing.assert_array_equal(
        np.asarray(Chain(extracellular=ExtracellularPotential())(conditions, io=io)),
        np.asarray(io.outputs.potential_recording(io.distances(CELLS), currents)),
    )
    # `io` is not a field, so no Conditions subclass needs one
    Chain(extracellular=ExtracellularPotential()).check(Conditions)
    assert "io" not in Chain(extracellular=ExtracellularPotential()).requires()


def test_a_misspelt_name_is_caught_before_anything_runs():
    recording = Chain(Sum(extracellular=ExtracellularPotential(), drift=Misspelt()))

    with pytest.raises(TypeError, match=r"'drift' \(Misspelt\) asks for `temprature`"):
        recording.check(Conditions)
    with pytest.raises(TypeError, match="temprature"):
        recording.check(RigConditions)
    # on every call too, even when nothing is set
    with pytest.raises(TypeError, match="temprature"):
        recording(Conditions())


def test_var_keywords_receive_every_set_field():
    block = Everything()
    Chain(block)(Conditions(dt=DT, segment=3))

    assert block.seen == {"dt": DT, "segment": 3}


def test_a_user_subclass_drives_a_user_block():
    io = _stack_io(ElectrodeRC(20.0))
    env = _Env(io)
    temperature = 36.0 + np.linspace(0.0, 0.5, N)
    recording = Chain(
        Sum(extracellular=ExtracellularPotential(), drift=Drift(4.0)),
        rc=ElectrodeRC(20.0),
    )
    conditions = RigConditions(
        membrane_currents_uA=_currents(), temperature=temperature
    )

    got = env.potential_recording(recording, conditions)

    neural = env.potential_recording(conditions=conditions)
    drift = ElectrodeRC(20.0)(
        4.0 * np.broadcast_to(temperature - temperature[0], (3, N)), dt=DT
    )
    np.testing.assert_allclose(got, neural + drift, rtol=1e-5, atol=1e-6)
    with pytest.raises(TypeError, match="temperature"):
        env.potential_recording(recording, Conditions(membrane_currents_uA=_currents()))


# Sum and Chain


def test_a_sum_adds_its_parts_and_refuses_parts_of_another_shape():
    io = _stack_io()

    got = io.observe(Conditions(dt=DT), Sum(a=Offset(1.0), b=Offset(2.0)))
    np.testing.assert_array_equal(got, np.full((3, N), 3.0))

    class Short(Offset):
        def __call__(self, *, io):
            return np.zeros((2, N))

    with pytest.raises(ValueError, match="one shape"):
        io.observe(Conditions(dt=DT), Sum(a=Offset(1.0), b=Short(0.0)))


def test_a_chain_runs_its_transfers_in_order_and_nests():
    io = _stack_io()
    inner = Chain(Offset(1.0), double=Gain(2.0))
    outer = Chain(Sum(inner=inner, one=Offset(1.0)), triple=Gain(3.0))

    np.testing.assert_array_equal(
        io.observe(Conditions(dt=DT), outer), np.full((3, N), 9.0)
    )
    assert outer["double"] is inner["double"]
    assert "offset" in outer and "one" in outer


def test_sources_and_transfers_stay_in_their_places():
    with pytest.raises(TypeError, match="transfer"):
        Sum(a=Offset(1.0), b=Gain(2.0))
    with pytest.raises(TypeError, match="head of a Chain"):
        Chain(Gain(2.0))
    with pytest.raises(TypeError, match="source"):
        Chain(a=Offset(1.0), b=Offset(2.0))
    with pytest.raises(TypeError, match="source"):
        ComposedIO(outputs=[MEA(ELECTRODES), LightArtifact(n_channels=3)])
    with pytest.raises(TypeError, match="not both"):
        ComposedIO(
            outputs=[MEA(ELECTRODES), Noise(1.0)],
            recording=Chain(ExtracellularPotential()),
        )


def test_names_default_to_the_class_and_duplicates_must_be_named():
    chain = Chain(ExtracellularPotential(), ElectrodeRC(), FIR([1.0]), Bandpass())
    assert chain.names == ("extracellular", "electrode_rc", "fir", "bandpass")

    with pytest.raises(ValueError, match="two blocks are named 'light_artifact'"):
        Sum(LightArtifact(), LightArtifact())
    with pytest.raises(ValueError, match="two blocks are named 'noise'"):
        Chain(ExtracellularPotential(), Noise(1.0), Noise(2.0))
    # a nested block shares the namespace
    with pytest.raises(ValueError, match="two blocks are named 'rc'"):
        Chain(
            Sum(extracellular=ExtracellularPotential(), rc=Offset(1.0)),
            rc=ElectrodeRC(),
        )
    with pytest.raises(ValueError, match="cannot name a block"):
        Chain(kwargs=ExtracellularPotential())
    named = Chain(ExtracellularPotential(), a=Noise(1.0), b=Noise(2.0))
    assert named["b"].sigma_uV == 2.0
    with pytest.raises(KeyError, match="names no block"):
        named["c"]


def test_a_composition_round_trips_through_its_document():
    recording = Chain(
        Sum(
            extracellular=ExtracellularPotential(),
            artifact=LightArtifact([1.0, -0.5], taus=(0.5, 5.0)),
        ),
        rc=ElectrodeRC(20.0),
        noise=Noise(2.0, seed=3),
    )
    document = json.loads(json.dumps(recording.serialize()))

    again = import_instance(document)
    assert document["cls"] == "livn.io.Chain"
    assert list(document["kwargs"]) == ["sum", "rc", "noise"]
    assert again.serialize() == recording.serialize()
    assert again.names == recording.names
    assert again["artifact"].taus == (0.5, 5.0)
    with pytest.raises(TypeError, match="serialize"):
        Chain(Offset(1.0)).serialize()


def test_set_params_reaches_nested_blocks_by_name():
    recording = Chain(
        Sum(
            extracellular=ExtracellularPotential(),
            artifact=LightArtifact(n_channels=2, taus=(1.0,)),
        ),
        rc=ElectrodeRC(20.0),
    )

    changed = recording.set_params(
        {"artifact-weights": np.ones((2, 1)), "rc-r_seal": 5.0}
    )
    np.testing.assert_array_equal(changed["artifact"].weights, np.ones((2, 1)))
    np.testing.assert_array_equal(recording["artifact"].weights, np.zeros((2, 1)))
    with pytest.raises(KeyError, match="'<block>-<parameter>'"):
        recording.set_params({"weights": 1.0})


# the Env and IO fill in what they own


def test_the_env_fills_what_it_owns_and_refuses_other_coordinates():
    light = PatternedLight([[0, 0.0, 0.0, 60.0]], irradiance=2.0)
    io = ComposedIO(inputs=light, outputs=MEA(ELECTRODES, output_radius=2000.0))
    env = _Env(io)
    recording = Chain(Sum(extracellular=ExtracellularPotential(), all=Everything()))
    env.potential_recording(recording, Conditions(membrane_currents_uA=_currents()))
    seen = recording["all"].seen

    assert set(seen) == {"membrane_currents_uA", "dt", "neuron_coordinates", "gids"}
    assert seen["dt"] == DT
    np.testing.assert_array_equal(seen["neuron_coordinates"], CELLS)
    np.testing.assert_array_equal(seen["gids"], CELLS[:, 0])
    with pytest.raises(ValueError, match="neuron_coordinates disagrees with the env"):
        env.potential_recording(
            conditions=Conditions(
                membrane_currents_uA=_currents(), neuron_coordinates=CELLS + 1.0
            )
        )
    # a composition that reads no coordinates gets none computed
    filled = env.fill_conditions(Conditions(), Chain(Offset(1.0)))
    assert filled.neuron_coordinates is None and filled.dt == DT


def test_the_env_projects_the_rows_its_gids_name():
    io = _stack_io()
    env = _Env(io)
    currents = _currents()[[2, 0]]

    got = env.potential_recording(
        conditions=Conditions(membrane_currents_uA=currents, gids=[2, 0])
    )

    want = io.outputs.potential_recording(io.distances(CELLS[[2, 0]]), currents)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_rows_pair_by_section_and_a_section_without_a_coordinate_is_left_out():
    from livn.types import place_rows

    owners, names = [0, 0, 1], ["soma", "dend", "soma"]
    # gid 1 records a dend the model places nowhere
    picked, kept, unplaced = place_rows(
        owners, names, [1, 0, 1, 0], ["soma", "dend", "dend", "soma"]
    )
    assert (picked, kept, unplaced) == ([2, 1, 0], [0, 1, 3], ["dend"])
    # in order without names, and a gid with more rows than coordinates refuses
    assert place_rows(owners, None, [0, 1, 0], None)[:2] == ([0, 2, 1], [0, 1, 2])
    with pytest.raises(ValueError, match="more membrane-current rows"):
        place_rows(owners, None, [1, 1], None)


def test_the_env_leaves_out_rows_recorded_at_a_section_it_cannot_place():
    io = _stack_io()
    env = _Env(io, sections=np.asarray(["soma", "soma", "soma"]))
    currents = _currents()
    extra = np.random.default_rng(3).normal(size=(1, N)) * 1e-3

    got = env.potential_recording(
        conditions=Conditions(
            membrane_currents_uA=np.concatenate([currents, extra]),
            gids=[0, 1, 2, 1],
            sections=["soma", "soma", "soma", "dend"],
        )
    )

    # the dend row of gid 1 has no coordinate: the projection is the soma rows' alone
    want = io.outputs.potential_recording(io.distances(CELLS), currents)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_sections_are_a_static_tuple():
    conditions = Conditions(sections=np.asarray(["soma", "dend"]))
    assert conditions.sections == ("soma", "dend")


def test_without_currents_the_neural_block_says_so_and_the_empty_dish_records():
    io = _stack_io()
    env = _Env(io)

    with pytest.raises(ValueError, match="needs `membrane_currents_uA`"):
        env.potential_recording()
    np.testing.assert_array_equal(
        env.potential_recording(Chain(Offset(1.5))), np.full((3, N), 1.5)
    )


def test_stack_is_the_shorthand_of_a_composed_io():
    assert is_stack(stack(ElectrodeRC(), Noise(1.0)))
    assert not is_stack(Chain(extracellular=ExtracellularPotential(), rc=ElectrodeRC()))
    assert not is_stack(Chain(Sum(extracellular=ExtracellularPotential())))


# conditions


def test_conditions_are_frozen_and_replaced():
    conditions = Conditions(dt=0.1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        conditions.dt = 0.2
    again = conditions.replace(segment=2)
    assert (conditions.segment, again.segment, again.dt) == (None, 2, 0.1)
    assert again.as_dict() == {"dt": 0.1, "segment": 2}
    with pytest.raises(TypeError, match="not fields of Conditions"):
        conditions.replace(temperature=1.0)
    rig = RigConditions(temperature=np.ones(3), dt=np.float64(0.25))
    assert dataclasses.is_dataclass(rig) and type(rig.dt) is float
    assert "temperature" in RigConditions.field_names()
