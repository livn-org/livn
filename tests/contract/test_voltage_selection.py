from __future__ import annotations

import os

import numpy as np
import pytest

from testing import livn_test_env

DT = 0.05
CELLS = 12
DURATION = 20.0

pytestmark = pytest.mark.skipif(
    os.environ.get("LIVN_BACKEND") != "neuron",
    reason="builds a network, which resolves the backend at import time",
)


def _env():
    pytest.importorskip("livn")
    env = livn_test_env()
    env.selection(CELLS)
    env.init()
    return env


def _traces(run):
    iv = np.asarray(run.voltage_ids if run.voltage_ids is not None else [])
    vv = np.asarray(run.voltage if run.voltage is not None else [])
    return iv.astype(np.int64), vv


def test_the_whole_simulation_is_known_without_a_collective():
    env = _env()
    mine = env.simulated_gids()
    everywhere = env.simulated_gids(everywhere=True)

    assert len(everywhere) == CELLS
    assert set(mine.tolist()) <= set(everywhere.tolist())
    assert set(everywhere.tolist()) < set(int(g) for g in env.active_gids()), (
        "the selection left nothing out, so this graph cannot show the "
        "difference between what the graph describes and what was built"
    )
    if env.comm is None or env.comm.size == 1:
        assert np.array_equal(mine, everywhere)


def test_recording_named_cells_records_those_and_no_others():
    env = _env()
    wanted = sorted(env.simulated_gids(everywhere=True))[:3]

    env.record_voltage(gids=wanted, dt=DT)
    iv, vv = _traces(env.run(DURATION, root_only=False))

    assert sorted(set(iv.tolist())) == wanted
    assert len(vv) >= len(wanted), "a two-compartment cell contributes two rows"


def test_a_narrowed_recording_is_the_wide_one_with_the_other_cells_dropped():
    wide_env = _env()
    wide_env.record_voltage(dt=DT)
    wide_iv, wide_vv = _traces(wide_env.run(DURATION, root_only=False))

    wanted = sorted(wide_env.simulated_gids(everywhere=True))[:3]

    narrow_env = _env()
    narrow_env.record_voltage(gids=wanted, dt=DT)
    narrow_iv, narrow_vv = _traces(narrow_env.run(DURATION, root_only=False))

    assert len(narrow_iv) < len(wide_iv), "nothing was narrowed"
    keep = np.isin(wide_iv, wanted)
    assert np.array_equal(narrow_iv, wide_iv[keep])
    assert np.allclose(narrow_vv, wide_vv[keep])


def test_a_gid_with_no_cell_is_refused_rather_than_silently_dropped():
    env = _env()
    absent = max(int(g) for g in env.system.gids) + 1000

    with pytest.raises(ValueError, match="have no cell in this simulation"):
        env.record_voltage(gids=[absent])


def test_a_gid_the_selection_left_out_says_that_is_why():
    env = _env()
    left_out = sorted(
        set(int(g) for g in env.system.gids)
        - set(env.simulated_gids(everywhere=True).tolist())
    )
    if not left_out:
        pytest.skip("this graph is smaller than the selection")

    with pytest.raises(ValueError) as raised:
        env.record_voltage(gids=[left_out[0]])

    assert "no cell in this simulation" in str(raised.value)
    assert f"{len(env.simulated_gids(everywhere=True))} gids do" in str(raised.value)


def test_recording_no_cells_at_all_is_refused():
    env = _env()

    with pytest.raises(ValueError, match="Pass `gids=None`"):
        env.record_voltage(gids=[])


def test_a_decoding_can_ask_for_the_cells_and_the_resolution_it_needs():
    from livn.decoding import GatherAndMerge

    env = _env()
    wanted = sorted(env.simulated_gids(everywhere=True))[:2]

    merged = env(
        GatherAndMerge(
            duration=int(DURATION),
            membrane_currents=False,
            voltage_gids=wanted,
            voltage_dt=DT,
        )
    )

    iv = np.asarray(merged.voltage_ids).astype(np.int64)
    assert sorted(set(iv.tolist())) == wanted
    assert float(env.voltage_recording_dt) == pytest.approx(DT)
