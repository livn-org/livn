from __future__ import annotations

import contextlib
import itertools
import os

import numpy as np
import pytest

from livn.io import MAX_STIMULUS_GB_ENV, calculate_cell_stimulus
from livn.stimulus import STIMULUS_CHUNK_MB_ENV, Stimulus
from testing import livn_test_env

DT = 0.1
CELLS = 20

neuron_only = pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") not in ("neuron", "native"),
    reason="builds a network, which resolves the backend at import time",
)
neuron_clamps = pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") != "neuron",
    reason="reaches into the NEURON backend's IClamp drives",
)


_OPEN_ENVS = []


@pytest.fixture(autouse=True)
def _close_envs():
    """A NEURON env per test, closed after it; leaked ones wedge later psolves."""
    yield
    while _OPEN_ENVS:
        with contextlib.suppress(Exception):
            _OPEN_ENVS.pop().close()


def _env(reach=1e9):
    pytest.importorskip("livn")
    from livn.io import MEA
    from livn.system import System

    xyz = np.asarray(System(os.environ["LIVN_TEST_SYSTEM"]).neuron_coordinates)[:, 1:]
    centre = [xyz[:, 0].min() - 100.0, xyz[:, 1].mean(), xyz[:, 2].mean()]
    env = livn_test_env(io=MEA([[0, *centre]], input_radius=reach, output_radius=reach))
    env.selection(CELLS)
    env.init()
    _OPEN_ENVS.append(env)
    return env


def _command(env, n_steps=40):
    inputs = np.zeros((n_steps, len(env.io.channel_ids)), dtype=np.float32)
    inputs[10:12, 0] = 60.0
    return inputs


def _induction(n_gids, reached):
    return np.stack(
        [
            np.zeros(n_gids),
            np.arange(n_gids),
            np.r_[np.ones(reached), np.zeros(n_gids - reached)],
        ],
        axis=1,
    )


def test_an_accidental_expansion_is_refused_before_it_allocates():
    induction = _induction(2_600, reached=125)
    command = np.ones((1, 1_170_000, 1), dtype=np.float32)

    with pytest.raises(MemoryError) as raised:
        calculate_cell_stimulus(command, induction, n_gids=2_600)

    message = str(raised.value)
    assert "1,170,000" in message, "the time axis"
    assert "2,600" in message, "the cell axis"
    assert "125" in message, "how many of those cells are even reachable"
    assert "Policy" in message, "the way out"
    assert STIMULUS_CHUNK_MB_ENV in message, "what sizes a window, having taken it"
    assert MAX_STIMULUS_GB_ENV in message, "the override"
    assert "{" not in message, "an env var name is named, not left as a placeholder"


def test_the_ceiling_is_five_gigabytes_and_liftable(monkeypatch):
    induction = _induction(2_600, reached=125)
    over = np.ones((1, 700_000, 1), dtype=np.float32)
    under = np.ones((1, 400_000, 1), dtype=np.float32)

    calculate_cell_stimulus(under, induction, n_gids=2_600)
    with pytest.raises(MemoryError):
        calculate_cell_stimulus(over, induction, n_gids=2_600)

    monkeypatch.setenv(MAX_STIMULUS_GB_ENV, "16")
    calculate_cell_stimulus(over, induction, n_gids=2_600)

    monkeypatch.setenv(MAX_STIMULUS_GB_ENV, "0")
    calculate_cell_stimulus(over, induction, n_gids=2_600)


def test_a_ceiling_that_is_not_a_number_says_so(monkeypatch):
    monkeypatch.setenv(MAX_STIMULUS_GB_ENV, "lots")
    with pytest.raises(ValueError, match="not a number of gigabytes"):
        calculate_cell_stimulus(
            np.ones((1, 2, 1), dtype=np.float32), _induction(4, 2), n_gids=4
        )


def test_widening_puts_each_column_back_on_its_own_compartment():
    gids = np.array([0, 0, 1, 1])
    sections = np.array([0, 1, 0, 1])
    narrow = Stimulus(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dt=DT,
        gids=np.array([0, 1]),
        sections=np.array([1, 0]),
    )

    wide = narrow.expand(gids, sections)

    assert np.array_equal(
        wide.array, np.array([[0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
    )
    assert np.array_equal(wide.gids, gids)
    assert np.array_equal(wide.sections, sections)


def test_widening_something_already_wide_is_a_no_op():
    gids, sections = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    wide = Stimulus(np.ones((2, 4)), dt=DT, gids=gids, sections=sections)

    assert wide.expand(gids, sections) is wide


def test_a_stimulus_that_does_not_say_which_columns_it_holds_cannot_be_widened():
    with pytest.raises(ValueError, match="does not say which columns"):
        Stimulus(np.ones((2, 4)), dt=DT).expand(np.array([0, 0, 1, 1]))


@neuron_only
def test_the_narrow_stimulus_is_the_wide_one_with_its_zeros_removed():
    from livn.io import coupled_sections, section_labels

    env = _env()
    coordinates = env.system.transform_coordinates(
        env.model.stimulus_coordinates, populations=env.active_populations()
    )
    command = _command(env)

    narrow = env.cell_stimulus(command, dt=DT)
    wide = narrow.expand(*section_labels(coordinates))

    _, _, keep = coupled_sections(coordinates, env.io._cell_induction)
    assert narrow.array.shape[-1] == len(keep)
    assert wide.array.shape[-1] == len(coordinates)

    assert np.allclose(np.asarray(wide.array)[:, keep], np.asarray(narrow.array))
    dropped = np.setdiff1d(np.arange(len(coordinates)), keep)
    assert np.all(np.asarray(wide.array)[:, dropped] == 0.0)


@neuron_only
def test_narrowing_does_not_change_what_the_network_does():
    from livn.io import section_labels

    spikes = []
    for widen in (False, True):
        env = _env()
        env.record_spikes()
        coordinates = env.system.transform_coordinates(
            env.model.stimulus_coordinates, populations=env.active_populations()
        )
        stimulus = env.cell_stimulus(_command(env), dt=DT)
        if widen:
            stimulus = stimulus.expand(*section_labels(coordinates))
            assert stimulus.array.shape[-1] == len(coordinates)

        run = env.run(30.0, stimulus=stimulus)
        times = run.spike_times
        spikes.append(np.asarray([] if times is None else times))
        env.clear()

    narrow, wide = spikes
    assert len(narrow) == len(wide), (
        f"{len(narrow)} spikes narrow against {len(wide)} wide: the restriction "
        "changed the simulation rather than only its representation"
    )
    assert np.allclose(narrow, wide)


def _drive(env, n_steps):
    peak = np.zeros(n_steps)

    block = env._stim_block
    if block is not None:
        width = min(n_steps, block.shape[1])
        peak[:width] = np.abs(block[:, :width]).max(axis=0)

    for stream in getattr(env, "_stim_streams", []):
        stimulus = stream["stimulus"]
        rendered = np.asarray(stimulus.window(0.0, stimulus.duration))
        rendered = np.abs(rendered[:, stream["columns"]]).max(axis=1)
        lo = stream["start_step"]
        hi = min(n_steps, lo + len(rendered))
        if hi > lo:
            peak[lo:hi] = np.maximum(peak[lo:hi], rendered[: hi - lo])

    return peak


@neuron_only
def test_a_policy_is_delivered_over_the_run_it_is_handed_to():
    from livn.policy import PulseSweepPolicy

    env = _env()
    env.record_spikes()
    sweep = PulseSweepPolicy(
        n_channels=len(env.io.channel_ids),
        channels=[0],
        amplitudes=(125.0, 250.0),
        repeats=2,
        trial_ms=200.0,
        onset_ms=50.0,
        dt=DT,
    )

    quiet = 100.0
    env.run(quiet)
    env.run(sweep.duration_ms, stimulus=sweep)

    driven = np.flatnonzero(_drive(env, int((quiet + sweep.duration_ms) / DT) + 1))
    starts = [driven[0]] + [b for a, b in itertools.pairwise(driven) if b - a > 1]

    assert len(starts) == sweep.n_trials, "a pulse went missing"
    for step, (at, _amplitude) in zip(starts, sweep.schedule(), strict=False):
        assert step * DT == pytest.approx(quiet + at, abs=DT)


@neuron_only
def test_a_run_too_short_for_its_policy_is_refused():
    from livn.policy import PulseSweepPolicy

    env = _env()
    sweep = PulseSweepPolicy(
        n_channels=len(env.io.channel_ids),
        channels=[0],
        amplitudes=(300.0,),
        repeats=4,
        trial_ms=200.0,
        onset_ms=50.0,
        dt=DT,
    )

    with pytest.raises(ValueError, match="cannot deliver"):
        env.run(100.0, stimulus=sweep)


def test_a_policy_handed_straight_to_stimulus_says_where_it_belongs():
    from livn.policy import PulseSweepPolicy

    sweep = PulseSweepPolicy(n_channels=4, channels=[0], amplitudes=(1.0,))
    with pytest.raises(ValueError, match="commands channels, not cells"):
        Stimulus.from_arg(sweep)
