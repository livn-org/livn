from __future__ import annotations

import os

import numpy as np
import pytest

from testing import livn_test_env

pytestmark = pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") != "neuron",
    reason="the extracellular block belongs to the neuron backend",
)

DT = 0.1
CELLS = 20
STEP_MS = 50.0


def _env():
    pytest.importorskip("livn")
    env = livn_test_env()
    env.selection(CELLS)
    env.init()
    env.record_spikes()
    return env


def _pulse(gids, n_steps=20):
    from livn.stimulus import Stimulus

    values = np.zeros((n_steps, len(gids)), dtype=np.float32)
    values[5:7, :] = 50.0
    return Stimulus(values, dt=DT, gids=gids)


def _block(env):
    return getattr(env, "_stim_block", None) if hasattr(env, "_stim_block") else None


def test_installing_a_stimulus_again_reuses_its_rows():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    heights = []
    for _ in range(4):
        env.run(STEP_MS, stimulus=_pulse(gids))
        heights.append(env._stim_block.shape[0])

    assert len(set(heights)) == 1, (
        f"the block grew a row per install: {heights}. Four calls on the same "
        "sections must occupy the same four rows they did on the first."
    )
    assert env._stim_block.shape[0] == len(env._stim_segments)
    assert len(set(map(id, env._stim_segments))) == len(env._stim_segments)


def test_the_block_carries_the_command_precision_not_double_it():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    env.run(STEP_MS, stimulus=_pulse(gids))

    assert env._stim_block.dtype == np.float32


def test_a_later_shorter_stimulus_does_not_truncate_an_earlier_one():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    env.run(STEP_MS, stimulus=_pulse(gids, n_steps=400))
    wide = env._stim_block.shape[1]

    env.run(STEP_MS, stimulus=_pulse(gids, n_steps=2))
    assert env._stim_block.shape[1] >= wide


def test_the_pulse_lands_at_the_offset_the_run_had_reached():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    env.run(STEP_MS)
    env.run(STEP_MS, stimulus=_pulse(gids))

    driven = np.flatnonzero(np.abs(env._stim_block).max(axis=0) > 0)
    assert driven.size, "nothing was written into the block"
    assert driven.min() * DT == pytest.approx(STEP_MS + 5 * DT, abs=DT)
