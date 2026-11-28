from __future__ import annotations

import contextlib
import gc

import numpy as np
import pytest

from livn.stimulus import Stimulus
from testing import livn_test_env

pytestmark = pytest.mark.needs("simulation")

DT = 0.1
LEAD = 100.0
PULSE = 50.0
AFTER = 50.0
AMPLITUDE = 0.5  # nA


def _pulse(gids, duration: float, onset: float) -> Stimulus:
    values = np.zeros((round(duration / DT), len(gids)))
    first = round(onset / DT)
    values[first : first + round(PULSE / DT)] = AMPLITUDE
    return Stimulus.from_current(values, dt=DT, gids=gids)


def _spikes(split: bool):
    gc.collect()
    env = livn_test_env().init()
    try:
        env.apply_model_defaults()
        env.record_spikes()
        gids = np.asarray(env.active_gids())
        if split:
            lead = env.run(LEAD)
            run = env.run(PULSE + AFTER, stimulus=_pulse(gids, PULSE + AFTER, 0.0))
            ids = np.concatenate([lead.spike_ids, run.spike_ids])
            times = np.concatenate([lead.spike_times, run.spike_times + LEAD])
        else:
            total = LEAD + PULSE + AFTER
            run = env.run(total, stimulus=_pulse(gids, total, LEAD))
            ids, times = run.spike_ids, run.spike_times
    finally:
        with contextlib.suppress(Exception):
            env.close()
    ids, times = np.asarray(ids), np.asarray(times, dtype=np.float64)
    order = np.lexsort((times, ids))
    return ids[order], times[order]


def test_a_current_pulse_in_a_continued_run_matches_one_run():
    ids, times = _spikes(split=False)
    driven = int(np.count_nonzero(times >= LEAD))
    assert driven >= 10, f"the pulse barely drives the cells ({driven} spikes)"

    split_ids, split_times = _spikes(split=True)
    assert split_ids.size == ids.size, (
        f"{split_ids.size} spikes in the continued run against {ids.size} in one run"
    )
    np.testing.assert_array_equal(split_ids, ids)
    np.testing.assert_allclose(split_times, times, atol=1e-6)
