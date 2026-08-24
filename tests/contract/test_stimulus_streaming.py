from __future__ import annotations

import contextlib
import itertools
import os

import numpy as np
import pytest

from livn.io import MAX_STIMULUS_GB_ENV
from livn.stimulus import STIMULUS_CHUNK_MB_ENV, Stimulus
from testing import livn_test_env

DT = 0.1
CELLS = 20

neuron_only = pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") != "neuron",
    reason="builds a network, which resolves the backend at import time",
)


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
    from livn.io import MEA
    from livn.system import System

    xyz = np.asarray(System(os.environ["LIVN_TEST_SYSTEM"]).neuron_coordinates)[:, 1:]
    centre = [xyz[:, 0].min() - 100.0, xyz[:, 1].mean(), xyz[:, 2].mean()]
    env = livn_test_env(io=MEA([[0, *centre]], input_radius=1e9, output_radius=1e9))
    env.selection(CELLS)
    env.init()
    _OPEN_ENVS.append(env)
    return env


def _sweep(env, repeats=2, trial_ms=200.0):
    from livn.policy import PulseSweepPolicy

    return PulseSweepPolicy(
        n_channels=len(env.io.channel_ids),
        channels=[0],
        amplitudes=(150.0, 300.0),
        repeats=repeats,
        trial_ms=trial_ms,
        onset_ms=50.0,
        dt=DT,
    )


def _width(env) -> int:
    probe = np.zeros((1, len(env.io.channel_ids)), dtype=np.float32)
    return env.cell_stimulus(probe, dt=DT).width


def test_a_deferred_stimulus_renders_what_holding_it_would_have():
    held = np.arange(40, dtype=np.float64).reshape(10, 4)
    gids = np.array([0, 0, 1, 1])
    sections = np.array([0, 1, 0, 1])

    deferred = Stimulus(
        dt=DT,
        gids=gids,
        sections=sections,
        source=lambda a, b: held[round(a / DT) : round(b / DT)],
        extent=10 * DT,
    )

    assert deferred.deferred
    assert deferred.width == 4
    assert deferred.duration == pytest.approx(10 * DT)
    assert np.array_equal(deferred.window(0.0, 3 * DT), held[:3])
    assert np.array_equal(deferred.window(3 * DT, 10 * DT), held[3:])
    assert np.array_equal(deferred.array, held)
    assert list(deferred.columns()) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_a_source_that_changes_its_columns_is_refused():
    deferred = Stimulus(
        dt=DT,
        gids=np.array([0, 1, 2, 3]),
        source=lambda a, b: np.zeros((2, 3)),
        extent=1.0,
    )

    with pytest.raises(ValueError, match="same columns in the same order"):
        deferred.window(0.0, 2 * DT)


def test_a_deferred_stimulus_has_to_say_what_it_holds_and_how_long_it_is():
    with pytest.raises(ValueError, match="must name the `gids`"):
        Stimulus(source=lambda a, b: np.zeros((1, 1)), extent=1.0)

    with pytest.raises(ValueError, match="how long it is"):
        Stimulus(source=lambda a, b: np.zeros((1, 1)), gids=np.array([0]))

    with pytest.raises(ValueError, match="not both"):
        Stimulus(
            np.zeros((2, 1)), gids=np.array([0]), source=lambda a, b: None, extent=1.0
        )

    with pytest.raises(ValueError, match="either an `array` or a `source`"):
        Stimulus()


@pytest.mark.parametrize("given", ["lots", "0", "-4"])
def test_a_window_budget_that_cannot_size_a_window_says_so(monkeypatch, given):
    from livn.stimulus import chunk_bytes

    monkeypatch.setenv(STIMULUS_CHUNK_MB_ENV, given)
    with pytest.raises(ValueError, match=STIMULUS_CHUNK_MB_ENV):
        chunk_bytes()


@neuron_only
def test_a_policy_handed_to_run_is_deferred_rather_than_held():
    env = _env()
    sweep = _sweep(env)

    env.run(sweep.duration_ms, stimulus=sweep)

    assert env._stim_block is None, "the command was held after all"
    assert len(env._stim_streams) == 1


@neuron_only
def test_streaming_a_command_simulates_identically_to_holding_it():
    spikes = []
    for stream in (True, False):
        env = _env()
        env.record_spikes()
        sweep = _sweep(env)

        stimulus = sweep
        if not stream:
            stimulus = env.cell_stimulus(
                sweep.window(0.0, sweep.duration_ms, DT), dt=DT
            )
            assert not stimulus.deferred

        run = env.run(sweep.duration_ms, stimulus=stimulus)
        assert (env._stim_block is None) is stream
        times = run.spike_times
        spikes.append(np.asarray([] if times is None else times))
        env.clear()

    streamed, held = spikes
    assert len(streamed) == len(held), (
        f"{len(streamed)} spikes streamed against {len(held)} held: pulling the "
        "command a window at a time changed the simulation rather than only how "
        "it is stored"
    )
    assert np.allclose(streamed, held)


@neuron_only
def test_a_protocol_too_long_to_hold_is_still_delivered(monkeypatch):
    env = _env()
    sweep = _sweep(env, repeats=4, trial_ms=400.0)

    n_steps = round(sweep.duration_ms / DT)
    held_bytes = n_steps * _width(env) * 4
    monkeypatch.setenv(MAX_STIMULUS_GB_ENV, str(held_bytes / 2**30 / 2))
    monkeypatch.setenv(STIMULUS_CHUNK_MB_ENV, str(held_bytes / 2**20 / 8))

    with pytest.raises(MemoryError):
        env.cell_stimulus(sweep.window(0.0, sweep.duration_ms, DT), dt=DT)

    env.record_spikes()
    env.run(sweep.duration_ms, stimulus=sweep)

    stream = env._stim_streams[0]
    assert stream["chunk_steps"] < n_steps, "the window is the whole protocol"

    peaks, at = [], 0.0
    step_ms = stream["chunk_steps"] * DT
    while at < sweep.duration_ms:
        window = np.asarray(
            stream["stimulus"].window(at, min(at + step_ms, sweep.duration_ms))
        )
        peaks.append(np.abs(window).max(axis=1))
        at += step_ms

    peak = np.concatenate(peaks)
    assert len(peak) == n_steps, "the windows do not tile the protocol"
    driven = np.flatnonzero(peak > 0)
    starts = [driven[0]] + [b for a, b in itertools.pairwise(driven) if b - a > 1]
    assert len(starts) == sweep.n_trials


@neuron_only
def test_windows_do_not_accumulate_over_a_protocol_delivered_in_pieces():
    env = _env()
    sweep = _sweep(env)

    env.run(sweep.duration_ms, stimulus=sweep)
    assert len(env._stim_streams) == 1

    env.run(sweep.duration_ms, stimulus=sweep)
    assert len(env._stim_streams) == 1, "the finished command is still held"
    assert env._stim_streams[0]["start_step"] == round(sweep.duration_ms / DT)


@neuron_only
def test_a_run_handed_no_stimulus_is_not_stimulated_by_the_last_one():
    env = _env()
    sweep = _sweep(env)

    env.run(sweep.duration_ms, stimulus=sweep)
    for clamp in env._stim_clamps:
        clamp.amp = 12.0

    env.run(50.0)

    assert not env._stim_streams, "the finished command is still installed"
    assert all(clamp.amp == 0.0 for clamp in env._stim_clamps)


@neuron_only
def test_a_streamed_command_stops_when_it_ends():
    env = _env()
    sweep = _sweep(env)
    quiet = 100.0

    env.run(sweep.duration_ms + quiet, stimulus=sweep)

    stream = env._stim_streams[0]
    last = stream["start_step"] + stream["n_steps"] - 1
    col = np.zeros(len(env._stim_rows))

    env._accumulate_stim_stream(stream, last + 1, col)
    assert not col.any(), "the command kept driving after it was over"
