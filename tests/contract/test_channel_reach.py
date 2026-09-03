from __future__ import annotations

import os

import numpy as np
import pytest

from testing import livn_test_env

CELLS = 12

pytestmark = pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") not in ("neuron", "native"),
    reason="builds a network, which resolves the backend at import time",
)


def _env(electrodes=None):
    pytest.importorskip("livn")
    from livn.io import MEA
    from livn.system import System

    xyz = np.asarray(System(os.environ["LIVN_TEST_SYSTEM"]).neuron_coordinates)[:, 1:]
    if electrodes is None:
        near = [xyz[:, 0].min() - 50.0, xyz[:, 1].mean(), xyz[:, 2].mean()]
        far = [xyz[:, 0].max() + 5000.0, xyz[:, 1].mean(), xyz[:, 2].mean()]
        electrodes = [[0, *near], [1, *far]]

    env = livn_test_env(io=MEA(electrodes, input_radius=1e9, output_radius=1e9))
    env.selection(CELLS)
    env.init()
    return env


def test_reach_is_one_row_per_channel_and_one_column_per_section():
    env = _env()
    coordinates = env.stimulus_coordinates()
    reach = env.channel_reach(coordinates)

    assert reach.shape == (len(env.io.channel_ids), len(coordinates))
    assert np.all(reach > 0), (
        "the conductor has no cutoff, so every section is reached to some "
        "degree -- a zero here would mean the falloff had gained one"
    )


def test_the_best_coupled_channel_is_the_one_in_the_tissue():
    env = _env()
    reach = env.channel_reach()

    assert int(reach.sum(axis=1).argmax()) == 0, "the near electrode did not win"
    assert reach[0].sum() > reach[1].sum()
    assert reach[0].max() > reach[1].max()


def test_reach_describes_the_cells_that_were_built_not_the_graph():
    env = _env()

    built = env.stimulus_coordinates()
    graph = env.stimulus_coordinates(simulated_only=False)

    assert len(built) < len(graph), "the selection left nothing out"
    assert {int(g) for g in built[:, 0]} == {
        int(g) for g in env.simulated_gids(everywhere=True)
    }
    assert env.channel_reach().shape[1] == len(built)


def test_the_best_coupled_channel_is_an_index_a_policy_can_drive():
    from livn.policy import BiphasicPulsePolicy

    env = _env()
    channel = int(env.channel_reach().sum(axis=1).argmax())

    policy = BiphasicPulsePolicy(
        n_channels=len(env.io.channel_ids), channels=[channel], amplitude=1.0
    )
    command = np.asarray(policy())
    assert command.shape[-1] == len(env.io.channel_ids)
    assert np.abs(command[:, channel]).max() > 0
    assert not np.abs(np.delete(command, channel, axis=1)).any()


def test_a_command_drives_the_cells_the_reach_says_it_will():
    env = _env()
    coordinates = env.stimulus_coordinates()
    reach_per_channel = env.channel_reach(coordinates)
    channel = int(reach_per_channel.sum(axis=1).argmax())
    reach = reach_per_channel[channel]

    command = np.zeros((2, len(env.io.channel_ids)), dtype=np.float64)
    command[:, channel] = 1.0
    stimulus = env.cell_stimulus(command, dt=0.1)

    from livn.io import section_labels

    wide = stimulus.expand(*section_labels(env.stimulus_coordinates(False)))
    columns = {
        (int(g), int(s)): i
        for i, (g, s) in enumerate(zip(wide.gids, wide.sections, strict=False))
    }
    rows = [columns[key] for key in zip(*section_labels(coordinates), strict=False)]
    delivered = np.asarray(wide.array)[0, rows]

    assert np.argmax(delivered) == np.argmax(reach)
    assert np.allclose(
        delivered / delivered.max(), reach / reach.max(), rtol=1e-5, atol=1e-6
    )
