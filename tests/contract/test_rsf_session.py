from __future__ import annotations

import contextlib
import json
import os

import numpy as np
import pytest

from testing import livn_test_env

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVN_BACKEND"), reason="no simulation backend selected"
)

DURATION = 100
AMPLITUDE = 1.5


@pytest.fixture(scope="module")
def recorded():
    pytest.importorskip("zarr", minversion="3")
    env = livn_test_env()
    env.init()
    env.record_spikes()
    env.record_voltage()
    env.record_membrane_current()

    inputs = np.zeros([DURATION, env.io.num_channels])
    inputs[10:30, :] = AMPLITUDE
    run = env.run(DURATION, env.cell_stimulus(inputs), dt=0.025)
    yield env, run
    with contextlib.suppress(Exception):
        env.close()


@pytest.fixture(scope="module")
def session(recorded, tmp_path_factory):
    from livn.rsf import store

    env, run = recorded
    path = str(tmp_path_factory.mktemp("rsf") / "session-1")
    store(run, path, env, voltage=True, currents=True)
    return path


def test_a_saved_run_reads_back(recorded, session):
    from livn.rsf import DEFAULT_GAIN_UV, load

    env, run = recorded
    loaded = load(session)
    back = loaded.run

    expected = np.asarray(env.potential_recording(run.current, run.current_ids))
    np.testing.assert_allclose(
        back.potential.values, expected, atol=DEFAULT_GAIN_UV / 2 + 1e-6
    )

    dt = back.potential.dt

    ours = sorted(
        zip(
            np.asarray(run.spike_ids).tolist(),
            np.asarray(run.spike_times).tolist(),
            strict=True,
        )
    )
    theirs = sorted(
        zip(back.spike_ids.tolist(), back.spike_times.tolist(), strict=True)
    )
    assert len(theirs) == len(ours)
    for (gid, t), (gid_back, t_back) in zip(ours, theirs, strict=True):
        assert gid == gid_back
        assert t - dt - 1e-6 <= t_back <= t + 1e-6

    for name in ("voltage", "current"):
        np.testing.assert_array_equal(
            back.values(name), np.asarray(run.values(name), dtype=np.float32)
        )
        if run.sections(name) is not None:
            np.testing.assert_array_equal(back.sections(name), run.sections(name))

    def rendered(io):
        return json.dumps(
            io.serialize(), default=lambda x: np.asarray(x).tolist(), sort_keys=True
        )

    assert rendered(loaded.io) == rendered(env.io)


def test_the_record_names_the_backend(recorded, session):
    from livn.backend import backend

    env, _ = recorded
    with open(os.path.join(session, "traces.zarr", "rsf", "simulation.json")) as f:
        simulation = json.load(f)
    assert simulation["simulator"]["name"] == f"livn_{backend()}"
    assert simulation["sources"]["ids"] == [
        int(g) for g in env.simulated_gids(everywhere=True)
    ]
