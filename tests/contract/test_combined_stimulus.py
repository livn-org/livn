from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.stimulus import Stimulus
from testing import livn_test_env

BACKEND = os.environ.get("LIVN_BACKEND", "")

pytestmark = pytest.mark.skipif(
    BACKEND not in ("neuron", "native", "diffrax", "brian2"),
    reason="needs a simulating backend",
)

CELLS = 4
DT = 0.1
DURATION = 30.0
N_STEPS = round(DURATION / DT)
# These cells are close to threshold (a 2 pA step moves them by a millivolt and
# 4 pA by four), so additivity holds for responses of tens of uV; the default
# RhO3c (1 pS) under 1 mW/mm2 gives about that
CURRENT = 1e-4  # nA
LIGHT = 1.0  # mW/mm2

_OPEN_ENVS = []


@pytest.fixture(autouse=True)
def _close_envs():
    """One NEURON env per test, closed after it; leaked ones wedge later psolves."""
    yield
    while _OPEN_ENVS:
        with contextlib.suppress(Exception):
            _OPEN_ENVS.pop().close()


def _env(input_mode=("current", "irradiance")):
    # only the diffrax cell is compiled for its modes; the others carry an
    # opsin and deliver any mode
    env = livn_test_env(model=ReducedCalciumSomaDendrite(input_mode=input_mode))
    env.selection(CELLS)
    env.init()
    _OPEN_ENVS.append(env)
    return env.record_voltage()


def _gids(env):
    return np.asarray(env.simulated_gids()).astype(int)


def _pulse(env, amplitude, start=50, stop=250):
    values = np.zeros((N_STEPS, len(_gids(env))), dtype=np.float32)
    values[start:stop] = amplitude
    return values


def _current(env, amplitude):
    return Stimulus.from_current(_pulse(env, amplitude), dt=DT, gids=_gids(env))


def _light(env, amplitude):
    return Stimulus.from_irradiance(_pulse(env, amplitude), dt=DT, gids=_gids(env))


def _voltage(env, stimulus):
    if BACKEND != "diffrax":
        env.clear(reseed=False)
    run = env.run(DURATION, stimulus=stimulus, dt=DT if BACKEND == "diffrax" else None)
    return np.asarray(run.voltage, dtype=np.float64)


@pytest.mark.skipif(BACKEND == "brian2", reason="brian2 delivers one mode per run")
def test_a_combined_run_with_one_mode_dark_is_the_other_mode_alone():
    both = _env()
    if BACKEND == "diffrax":
        current_only, light_only = _env("current"), _env("irradiance")
    else:
        current_only = light_only = both

    combined = _voltage(both, _current(both, CURRENT) + _light(both, 0.0))
    alone = _voltage(current_only, _current(current_only, CURRENT))
    np.testing.assert_array_equal(combined, alone)

    combined = _voltage(both, _current(both, 0.0) + _light(both, LIGHT))
    alone = _voltage(light_only, _light(light_only, LIGHT))
    np.testing.assert_array_equal(combined, alone)


@pytest.mark.skipif(BACKEND == "brian2", reason="brian2 delivers one mode per run")
def test_below_threshold_a_combined_run_is_the_sum_of_its_parts():
    env = _env()
    rest = _voltage(env, None)
    by_current = _voltage(env, _current(env, CURRENT)) - rest
    by_light = _voltage(env, _light(env, LIGHT)) - rest
    together = _voltage(env, _current(env, CURRENT) + _light(env, LIGHT)) - rest

    scales = [np.abs(by_current).max(), np.abs(by_light).max()]
    assert scales[0] > 0.01, "the current did not move the membrane"
    assert scales[1] > 0.001, "the light did not move the membrane"
    assert np.abs(together).max() < 1.0, "the cells left the linear range"
    # what remains is the interaction: the light's driving force moves with V
    residual = np.abs(together - (by_current + by_light)).max()
    assert residual < 0.05 * min(scales), (
        f"the combined response differs from the sum by {residual:.3g} mV, on "
        f"responses of {scales[0]:.3g} (current) and {scales[1]:.3g} mV (light)"
    )


@pytest.mark.skipif(BACKEND == "brian2", reason="brian2 delivers one mode per run")
def test_two_parts_of_one_mode_are_the_summed_stimulus():
    env = _env()
    gids = _gids(env)
    early = Stimulus.from_current(_pulse(env, CURRENT, 50, 150), dt=DT, gids=gids)
    late = Stimulus.from_current(_pulse(env, CURRENT, 100, 250), dt=DT, gids=gids)
    summed = Stimulus.from_current(
        _pulse(env, CURRENT, 50, 150) + _pulse(env, CURRENT, 100, 250),
        dt=DT,
        gids=gids,
    )

    np.testing.assert_array_equal(_voltage(env, early + late), _voltage(env, summed))


@pytest.mark.skipif(BACKEND == "brian2", reason="brian2 delivers one mode per run")
def test_light_of_two_wavelengths_is_refused():
    env = _env("irradiance" if BACKEND == "diffrax" else ("current", "irradiance"))
    gids = _gids(env)
    blue = Stimulus.from_irradiance(
        _pulse(env, LIGHT), dt=DT, gids=gids, wavelength=470.0
    )
    amber = Stimulus.from_irradiance(
        _pulse(env, LIGHT), dt=DT, gids=gids, wavelength=590.0
    )
    with pytest.raises(ValueError, match="470 and 590 nm"):
        env.run(DURATION, stimulus=blue + amber)


@pytest.mark.skipif(BACKEND != "diffrax", reason="the diffrax module declares modes")
def test_diffrax_refuses_two_modes_on_a_single_mode_module():
    env = _env("current")
    with pytest.raises(ValueError, match="input_mode=\\('current', 'irradiance'\\)"):
        env.run(
            DURATION,
            stimulus=_current(env, CURRENT) + _light(env, LIGHT),
            dt=DT,
        )


@pytest.mark.skipif(BACKEND != "brian2", reason="brian2 only")
def test_brian2_says_it_delivers_one_mode_per_run():
    env = livn_test_env()
    env.init()
    gids = np.asarray(env.system.gids).astype(int)
    values = np.zeros((N_STEPS, len(gids)), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="one stimulus mode per run"):
        env.run(
            DURATION,
            stimulus=Stimulus.from_current(values, dt=DT, gids=gids)
            + Stimulus.from_extracellular(values, dt=DT, gids=gids),
        )
