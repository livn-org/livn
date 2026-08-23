import os

import numpy as np
import pytest

from testing import (
    backend_supports,
    livn_test_mea,
    livn_test_system,
)

pytestmark = [
    pytest.mark.skipif(
        "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
    ),
    pytest.mark.needs("simulation"),
]


def _run(env, duration=20, amplitude=200.0):
    env.record_spikes()
    env.record_voltage()
    inputs = np.zeros([duration, env.io.num_channels])
    inputs[duration // 3 : 2 * duration // 3, :] = amplitude
    return env.run(duration, stimulus=env.cell_stimulus(inputs))


def _driven_env():
    from livn.env import Env

    return Env(livn_test_system(), io=livn_test_mea())


def _explicit_selection(env, count=3):
    population = env.active_populations()[0]
    gids = sorted(
        int(g) for g in np.asarray(env.system.coordinate_array(population))[:, 0]
    )
    return {population: gids[:count]}, gids[:count]


def _simulated(env):
    return sorted(int(g) for cells in env.cells.values() for g in cells)


def test_a_selection_instantiates_only_the_cells_it_names():
    env = _driven_env()
    spec, wanted = _explicit_selection(env)
    env.selection(spec)
    env.init()

    assert _simulated(env) == wanted
    assert len(wanted) < len(env.system.gids)


@pytest.mark.slow
def test_selecting_everything_is_the_same_as_not_selecting():
    plain = _driven_env()
    plain.init()
    reference = _run(plain)
    reference_ids = np.asarray(reference.voltage_ids)
    reference_v = np.asarray(reference.voltage)

    env = _driven_env()
    env.selection(1.0)
    env.init()
    selected = _run(env)

    np.testing.assert_array_equal(np.asarray(selected.voltage_ids), reference_ids)
    np.testing.assert_allclose(
        np.asarray(selected.voltage), reference_v, rtol=1e-5, atol=1e-6
    )


def test_a_recording_is_addressed_by_gid_not_by_position():
    env = _driven_env()
    env.selection(_explicit_selection(env)[0])
    env.init()

    run = _run(env)
    simulated = set(_simulated(env))

    recorded = {int(g) for g in np.asarray(run.voltage_ids)}
    assert recorded, "nothing was recorded, so the test asserts nothing"
    assert recorded <= simulated, (
        f"recorded gids {sorted(recorded - simulated)} were never instantiated; "
        "these look like module indices reported as gids"
    )


def test_an_explicit_gid_list_is_honoured_exactly():
    env = _driven_env()
    population = env.active_populations()[0]
    available = sorted(
        int(g) for g in np.asarray(env.system.coordinate_array(population))[:, 0]
    )
    wanted = available[1:4]

    env.selection({population: wanted})
    env.init()

    assert _simulated(env) == wanted


def test_the_stimulus_still_covers_the_whole_graph():
    env = _driven_env()
    env.selection(_explicit_selection(env)[0])
    env.init()

    inputs = np.zeros([10, env.io.num_channels])
    inputs[2:5, :] = 200.0
    stimulus = env.cell_stimulus(inputs)

    covered = len(np.unique(np.asarray(stimulus.gids)))
    assert covered == len(env.system.gids), (
        f"the command covers {covered} cells but the graph has {len(env.system.gids)}"
    )


def test_selection_is_refused_after_init():
    env = _driven_env()
    env.init()

    with pytest.raises(RuntimeError, match="before init"):
        env.selection(3)


def test_a_backend_that_simulates_can_also_select():
    assert backend_supports("simulation")

    env = _driven_env()
    spec, wanted = _explicit_selection(env)
    env.selection(spec)
    env.init()

    assert _simulated(env) == wanted, (
        "this backend declares Capability.SELECTION but did not restrict itself"
    )


@pytest.mark.needs("differentiable")
def test_a_selection_survives_a_pytree_round_trip():
    import jax

    env = _driven_env()
    spec, wanted = _explicit_selection(env)
    env.selection(spec)
    env.init()

    leaves, structure = jax.tree_util.tree_flatten(env)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert restored.num_cells == env.num_cells == len(wanted)
    np.testing.assert_array_equal(
        np.asarray(restored.simulated_gids(everywhere=True)),
        np.asarray(env.simulated_gids(everywhere=True)),
    )
    assert sorted(int(g) for g in restored.cells.gids) == wanted
