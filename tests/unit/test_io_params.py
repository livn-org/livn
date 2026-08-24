from __future__ import annotations

import pytest

pytest.importorskip("livn")

from livn.io import MEA, ComposedIO
from livn.types import Env


def _mea():
    return MEA(electrode_coordinates=[[0, 0.0, 0.0, 5.0], [1, 200.0, 0.0, 5.0]])


class _Env:
    set_params = Env.set_params

    def __init__(self, io=None):
        self.io = io
        self.weights = None
        self.noise = None

    def set_weights(self, weights):
        self.weights = weights

    def set_noise(self, noise):
        self.noise = noise


def test_each_level_is_addressed_by_its_own_prefix():
    mea = _mea()
    mea.set_params({"volume_conductor-stimulation_gain": 12.5, "input_radius": 80.0})

    assert mea.volume_conductor.stimulation_gain == 12.5
    assert mea.input_radius == 80.0


def test_a_bare_child_name_is_not_resolved_against_the_child():
    with pytest.raises(KeyError, match="not a parameter of MEA"):
        _mea().set_params({"stimulation_gain": 12.5})


def test_setting_a_parameter_drops_the_coupling_it_had_cached():
    mea = _mea()
    mea._cell_induction = "stale"
    mea.cell_measurement = "stale"

    mea.set_params({"volume_conductor-stimulation_gain": 2.0})

    assert mea._cell_induction is None
    assert mea.cell_measurement is None


def test_a_name_nothing_has_is_refused_at_the_level_that_was_addressed():
    with pytest.raises(KeyError, match="not a parameter of MEA"):
        _mea().set_params({"tissue_resistivity_ohm_m": 3.5})

    with pytest.raises(KeyError, match="not parameters of PointSourceModel"):
        _mea().set_params({"volume_conductor-tissue_resistivity_ohm_m": 3.5})


def test_the_env_routes_by_prefix_and_owns_none_of_it():
    env = _Env(io=_mea())
    env.io._cell_induction = "stale"

    env.set_params(
        {
            "io-volume_conductor-stimulation_gain": 3.0,
            "noise-g_e0": 0.03,
            "EXC_EXC-dend-AMPA-weight": 1.5,
        }
    )

    assert env.io.volume_conductor.stimulation_gain == 3.0
    assert env.io._cell_induction is None
    assert env.noise == {"g_e0": 0.03}
    assert env.weights == {"EXC_EXC-dend-AMPA-weight": 1.5}


def test_an_env_without_an_array_says_so():
    with pytest.raises(
        ValueError, match=r"no io on this env.*io-volume_conductor-stimulation_gain"
    ):
        _Env(io=None).set_params({"io-volume_conductor-stimulation_gain": 1.0})


def test_a_composed_array_addresses_one_half_at_a_time():
    composed = ComposedIO(
        inputs=MEA(electrode_coordinates=[[0, 0.0, 0.0, 5.0]]),
        outputs=MEA(electrode_coordinates=[[0, 0.0, 0.0, 5.0]]),
    )
    before = composed.outputs.volume_conductor.stimulation_gain

    composed.set_params({"inputs-volume_conductor-stimulation_gain": 3.0})

    assert composed.inputs.volume_conductor.stimulation_gain == 3.0
    assert composed.outputs.volume_conductor.stimulation_gain == before

    with pytest.raises(KeyError, match="not a parameter of ComposedIO"):
        composed.set_params({"nope": 1.0})


def test_the_shipped_conductor_round_trips():
    mea = _mea()
    mea.set_params({"volume_conductor-stimulation_gain": 9.5})

    spec = mea.serialize()
    assert spec["volume_conductor"]["stimulation_gain"] == 9.5

    back = MEA(
        electrode_coordinates=spec["electrode_coordinates"],
        input_radius=spec["input_radius"],
        output_radius=spec["output_radius"],
        volume_conductor=spec["volume_conductor"],
    )
    assert back.volume_conductor.stimulation_gain == 9.5
