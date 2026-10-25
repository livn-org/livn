from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

from testing import livn_test_env

pytestmark = pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") not in ("neuron", "native"),
    reason="the extracellular block belongs to the neuron backend",
)

DT = 0.1
CELLS = 20
STEP = 50.0


_OPEN_ENVS = []


@pytest.fixture(autouse=True)
def _close_envs():
    """A NEURON env per test, closed after it; leaked ones wedge later psolves."""
    yield
    while _OPEN_ENVS:
        with contextlib.suppress(Exception):
            _OPEN_ENVS.pop().close()


def _env():
    pytest.importorskip("livn")
    env = livn_test_env()
    env.selection(CELLS)
    env.init()
    env = env.record_spikes()
    _OPEN_ENVS.append(env)
    return env


def _pulse(gids, n_steps=20):
    from livn.stimulus import Stimulus

    values = np.zeros((n_steps, len(gids)), dtype=np.float32)
    values[5:7, :] = 50.0
    return Stimulus(values, dt=DT, gids=gids)


def _block(env):
    return getattr(env, "_stim_block", None) if hasattr(env, "_stim_block") else None


def _start(env):
    return getattr(env, "_stim_block_start", 0)


def test_installing_a_stimulus_again_reuses_its_rows():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    heights = []
    for _ in range(4):
        env.run(STEP, stimulus=_pulse(gids))
        heights.append(env._stim_block.shape[0])

    assert len(set(heights)) == 1, (
        f"the block grew a row per install: {heights}. Four calls on the same "
        "sections must occupy the same four rows they did on the first."
    )
    assert env._stim_block.shape[0] == len(env._stim_rows)
    assert len(set(env._stim_rows.values())) == len(env._stim_rows)


def test_the_block_carries_the_command_precision_not_double_it():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    env.run(STEP, stimulus=_pulse(gids))

    assert env._stim_block.dtype == np.float32


def test_a_later_shorter_stimulus_does_not_truncate_an_earlier_one():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    # 100 ms of command into a 50 ms run: its second half is still to come
    env.run(STEP, stimulus=_pulse(gids, n_steps=1000))
    end = _start(env) + env._stim_block.shape[1]

    env.run(STEP, stimulus=_pulse(gids, n_steps=2))
    assert _start(env) + env._stim_block.shape[1] >= end


def test_the_pulse_lands_at_the_offset_the_run_had_reached():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    env.run(STEP)
    env.run(STEP, stimulus=_pulse(gids))

    driven = np.flatnonzero(np.abs(env._stim_block).max(axis=0) > 0)
    assert driven.size, "nothing was written into the block"
    assert (_start(env) + driven.min()) * DT == pytest.approx(STEP + 5 * DT, abs=DT)


@pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") != "native",
    reason="the neuron backend still holds a block's whole history",
)
def test_the_block_holds_what_is_to_come_not_the_whole_history():
    env = _env()
    gids = np.asarray(env.active_neuron_coordinates())[:, 0].astype(int)

    widths = []
    for _ in range(20):
        env.run(STEP, stimulus=_pulse(gids, n_steps=round(STEP / DT)))
        widths.append(env._stim_block.shape[1])

    # each run's command is one run long: the block must not grow with the runs before it
    assert max(widths) <= 2 * round(STEP / DT), widths
