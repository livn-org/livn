from __future__ import annotations

import numpy as np
import pytest

from livn.decoding import stimulus_response

N_UNITS = 64
WINDOW_MS = 5_000.0
ONSET_MS = 1_000.0
BASELINE_HZ = 4.0
PITCH_UM = 200.0
SPEED_UM_PER_MS = 300.0


def _positions(n=N_UNITS, side=8, first=0):
    return {
        first + i: (float((i % side) * PITCH_UM), float((i // side) * PITCH_UM))
        for i in range(n)
    }


def _window(seed, positions, *, evoked_at=None, travelling=True, jitter=1.0):
    rng = np.random.default_rng(seed)
    it, tt = [], []
    for channel in sorted(positions):
        n = rng.poisson(BASELINE_HZ * WINDOW_MS / 1000.0)
        it += [channel] * n
        tt += list(rng.uniform(0.0, WINDOW_MS, n))
        if evoked_at is None:
            continue
        x, y = positions[channel]
        delay = float(np.hypot(x, y)) / SPEED_UM_PER_MS if travelling else 0.0
        it += [channel] * 3
        tt += list(ONSET_MS + evoked_at + delay + rng.normal(0.0, jitter, 3))

    it, tt = np.asarray(it), np.asarray(tt)
    order = np.argsort(tt, kind="stable")
    return it[order], tt[order]


def _regular(positions):
    step = 1000.0 / BASELINE_HZ
    grid = np.arange(0.0, WINDOW_MS, step)
    it = np.repeat(np.arange(N_UNITS), grid.size)
    tt = np.concatenate([grid + c * step / N_UNITS for c in range(N_UNITS)])
    order = np.argsort(tt, kind="stable")
    return it[order], tt[order]


def _measure(**kwargs):
    it, tt = _window(0, _positions(), **kwargs)
    return stimulus_response(it, tt, N_UNITS, ONSET_MS)


def test_a_window_with_no_response_says_so():
    r = _measure()

    assert r["response_gain"] == pytest.approx(1.0, abs=0.4)
    assert abs(r["evoked_rate_hz"]) < 2.0
    assert r["response_probability"] < 0.3
    assert np.isnan(r["response_latency_ms"])


def test_a_response_is_recovered_where_it_was_planted():
    r = _measure(evoked_at=40.0, travelling=False)

    assert abs(r["response_latency_ms"] - 40.0) <= 10.0
    assert r["response_gain"] > 2.0
    assert r["evoked_rate_hz"] > 3.0
    assert r["response_probability"] > 0.6
    assert 0.0 < r["response_duration_ms"] <= 50.0


def test_duration_is_measured_from_the_response_not_the_pulse():
    late = _measure(evoked_at=200.0, travelling=False)

    assert abs(late["response_latency_ms"] - 200.0) <= 10.0
    assert late["response_duration_ms"] > 0.0

    it, tt = _regular(_positions())
    steady = stimulus_response(it, tt, N_UNITS, ONSET_MS)
    assert np.isnan(steady["response_duration_ms"])
    assert np.isnan(steady["response_latency_ms"])


def test_a_window_that_cannot_hold_the_comparison_measures_nothing():
    it, tt = _window(0, _positions(), evoked_at=40.0)

    assert stimulus_response(it, tt, N_UNITS, 500.0, pre_ms=1000.0) == {}
    assert (
        stimulus_response(np.array([], dtype=int), np.array([]), N_UNITS, ONSET_MS)
        == {}
    )


def test_the_second_array_of_a_file_is_measured_like_the_first():
    first = _positions()
    second = _positions(first=N_UNITS)
    assert max(second) == 2 * N_UNITS - 1

    measured = {}
    for label, positions in (("first", first), ("second", second)):
        it, tt = _window(0, positions, evoked_at=40.0, travelling=True)
        measured[label] = stimulus_response(it, tt, N_UNITS, ONSET_MS)

    a, b = measured["first"], measured["second"]
    assert a and b

    for name in ("response_gain", "response_probability", "response_latency_ms"):
        assert a[name] == pytest.approx(b[name], rel=1e-9)

    assert b["response_probability"] > 0.6


def test_a_silent_electrode_at_the_top_of_the_block_still_measures():
    positions = _positions(first=N_UNITS)
    it, tt = _window(0, positions, evoked_at=40.0, travelling=True)

    dead = max(positions)
    keep = it != dead
    it, tt = it[keep], tt[keep]
    assert it.max() < dead, "this fixture needs the top channel silent"

    r = stimulus_response(it, tt, N_UNITS, ONSET_MS)

    assert r, "a dead top electrode must not silence the whole measurement"
    assert r["response_probability"] > 0.5
    assert r["response_probability"] <= (N_UNITS - 1) / N_UNITS
