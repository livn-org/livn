from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from livn.decoding import (
    ExtracellularRecruitment,
    RecruitmentCurve,
    recruitment_threshold,
)
from livn.policy import PulseSweepPolicy
from livn.run import Run

N_CHANNELS = 64
TRIAL_MS = 2_000.0
ONSET_MS = 1_000.0
BASELINE_HZ = 5.0


def _env():
    return SimpleNamespace(
        comm=None, system=SimpleNamespace(gids=list(range(N_CHANNELS))), io=None
    )


def _swept(recruitment: dict[float, float], repeats=4, seed=0):
    policy = PulseSweepPolicy(
        n_channels=N_CHANNELS,
        channels=[0],
        amplitudes=tuple(sorted(recruitment)),
        repeats=repeats,
        trial_ms=TRIAL_MS,
        onset_ms=ONSET_MS,
    )
    schedule = policy.schedule()

    rng = np.random.default_rng(seed)
    it, tt = [], []
    for trial in range(policy.n_trials):
        n = rng.poisson(BASELINE_HZ * TRIAL_MS / 1000.0 * N_CHANNELS)
        it += list(rng.integers(0, N_CHANNELS, n))
        tt += list(rng.uniform(trial * TRIAL_MS, (trial + 1) * TRIAL_MS, n))

    for at, amplitude in schedule:
        answering = rng.choice(
            N_CHANNELS, size=int(recruitment[amplitude] * N_CHANNELS), replace=False
        )
        for channel in answering:
            it += [int(channel)] * 4
            tt += list(at + rng.uniform(5.0, 40.0, 4))

    order = np.argsort(tt, kind="stable")
    run = Run(duration=policy.duration_ms).add_spikes(
        np.asarray(it, dtype=np.int64)[order], np.asarray(tt, dtype=float)[order]
    )
    return run, policy


def _read(recruitment, **kwargs):
    run, policy = _swept(recruitment)
    return RecruitmentCurve(
        duration=int(policy.duration_ms), schedule=policy.schedule(), **kwargs
    )(run, _env())


def test_the_crossing_is_bracketed_where_it_was_planted():
    out = _read({300.0: 0.02, 400.0: 0.05, 500.0: 0.80, 600.0: 0.95})

    assert out["censored"] is None
    assert (out["below_mv"], out["above_mv"]) == (400.0, 500.0)
    assert "crossing_mv" not in out


def test_a_network_that_never_answers_is_reported_as_a_bound():
    out = _read(dict.fromkeys((300.0, 400.0, 500.0, 600.0), 0.0))

    assert out["censored"] == "above"
    assert out["above_mv"] is None
    assert out["highest_tested_mv"] == 600.0
    assert out["recruited"] == 0.5


def test_a_network_already_recruited_at_the_lowest_is_bounded_the_other_way():
    out = _read(dict.fromkeys((300.0, 400.0, 500.0, 600.0), 0.95))

    assert out["censored"] == "below"
    assert out["below_mv"] is None
    assert out["above_mv"] == 300.0


def test_the_decoding_and_the_free_function_read_the_same_curve():
    out = _read({300.0: 0.02, 400.0: 0.05, 500.0: 0.80, 600.0: 0.95})
    curve = out["curve"]

    direct = recruitment_threshold(curve)
    assert direct
    for name, value in direct.items():
        assert out[name] == value


def test_the_recruited_fraction_is_a_parameter():
    recruitment = {300.0: 0.02, 400.0: 0.35, 500.0: 0.80, 600.0: 0.95}

    assert _read(recruitment)["above_mv"] == 500.0
    assert _read(recruitment, recruited=0.3)["above_mv"] == 400.0


def test_one_amplitude_brackets_nothing():
    out = _read({500.0: 0.9})

    assert out["curve"], "the curve is still measured"
    assert "censored" not in out, "but a single amplitude cannot bracket a crossing"


def test_a_pulse_with_no_room_for_its_baseline_counts_as_no_recruitment():
    run, policy = _swept({300.0: 0.9, 600.0: 0.9}, repeats=1)
    out = RecruitmentCurve(
        duration=int(policy.duration_ms),
        schedule=policy.schedule(),
        pre_ms=ONSET_MS + 500.0,
    )(run, _env())

    assert set(out["curve"]) == {300.0, 600.0}, "every amplitude still on the curve"
    assert out["curve"][300.0] == 0.0, "the unmeasurable trial reads as no answer"
    assert out["curve"][600.0] > 0.5, "and the measurable one is unaffected"


def test_an_empty_run_is_a_curve_of_zeros_not_an_absent_measurement():
    policy = PulseSweepPolicy(
        n_channels=N_CHANNELS,
        channels=[0],
        amplitudes=(300.0, 600.0),
        repeats=2,
        trial_ms=TRIAL_MS,
        onset_ms=ONSET_MS,
    )
    run = Run(duration=policy.duration_ms).add_spikes(
        np.array([], dtype=np.int64), np.array([], dtype=float)
    )

    out = RecruitmentCurve(
        duration=int(policy.duration_ms), schedule=policy.schedule()
    )(run, _env())

    assert out["curve"] == {300.0: 0.0, 600.0: 0.0}
    assert out["censored"] == "above"


@pytest.mark.parametrize("repeats", [1, 3])
def test_pulses_sharing_an_amplitude_are_pooled_by_their_median(repeats):
    run, policy = _swept({300.0: 0.1, 600.0: 0.9}, repeats=repeats)
    out = RecruitmentCurve(
        duration=int(policy.duration_ms), schedule=policy.schedule()
    )(run, _env())

    assert len(out["curve"]) == 2
    assert out["curve"][600.0] > out["curve"][300.0]


def _one_cell_env():
    return SimpleNamespace(
        comm=None,
        system=SimpleNamespace(gids=[0]),
        io=None,
        voltage_recording_dt=0.025,
    )


def _driven(answered: dict[float, bool], peaks: dict[float, float] | None = None):
    """A run in which one cell answered exactly the amplitudes named."""
    policy = PulseSweepPolicy(
        n_channels=1,
        channels=[0],
        amplitudes=tuple(sorted(answered)),
        repeats=1,
        trial_ms=TRIAL_MS,
        onset_ms=ONSET_MS,
    )
    it, tt = [], []
    for at, amplitude in policy.schedule():
        if answered[amplitude]:
            it += [0] * 3
            tt += [at + 2.0, at + 4.0, at + 6.0]

    run = Run(duration=policy.duration_ms).add_spikes(
        np.asarray(it, dtype=np.int64), np.asarray(tt, dtype=float)
    )
    if peaks is not None:
        dt = 0.025
        trace = np.full(int(policy.duration_ms / dt), -65.0)
        for at, amplitude in policy.schedule():
            trace[int(at / dt) : int((at + 10.0) / dt)] = peaks[amplitude]
        run = run.add_voltage(
            np.asarray([0], dtype=np.int64), trace[None, :], sections=["soma"]
        )
    return run, policy


def _range(answered, peaks=None, **kwargs):
    run, policy = _driven(answered, peaks)
    return ExtracellularRecruitment(
        duration=int(policy.duration_ms),
        schedule=policy.schedule(),
        units=1,
        **kwargs,
    )(run, _one_cell_env())


def test_both_ends_of_the_recruitable_range_are_reported():
    out = _range({10.0: False, 20.0: True, 40.0: True, 80.0: False})

    assert out["threshold_ua"] == 20.0
    assert out["block_ua"] == 80.0, "the amplitude it stopped answering at"
    assert out["band"] == pytest.approx(4.0)


def test_a_cell_that_never_stops_answering_has_no_upper_edge():
    out = _range({10.0: False, 20.0: True, 40.0: True, 80.0: True})

    assert out["threshold_ua"] == 20.0
    assert out["block_ua"] is None
    assert out["band"] == float("inf")


def test_a_monotonic_reading_would_have_missed_the_block():
    blocked = _range({10.0: False, 20.0: True, 40.0: True, 80.0: False})
    open_ended = _range({10.0: False, 20.0: True, 40.0: True, 80.0: True})

    assert blocked["threshold_ua"] == open_ended["threshold_ua"]
    assert blocked["band"] != open_ended["band"]


def test_a_peak_no_sodium_current_could_reach_closes_the_range():
    answered = {10.0: False, 20.0: True, 40.0: True, 80.0: True}
    peaks = {10.0: -40.0, 20.0: 30.0, 40.0: 45.0, 80.0: 180.0}

    out = _range(answered, peaks, ceiling_mv=60.6)

    assert out["threshold_ua"] == 20.0
    assert out["block_ua"] is None, "it answered every amplitude"
    assert out["ceiling_ua"] == 80.0, "but not as a spike"
    assert out["band"] == pytest.approx(4.0)


def test_a_cell_that_never_answers_has_no_threshold():
    out = _range(dict.fromkeys((10.0, 20.0, 40.0, 80.0), False))

    assert out["threshold_ua"] is None
    assert np.isnan(out["band"])
