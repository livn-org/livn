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
    env = env.record_spikes()
    env = env.record_voltage()
    env = env.record_membrane_current()

    inputs = np.zeros([DURATION, env.io.num_channels])
    inputs[10:30, :] = AMPLITUDE
    run = env.run(DURATION, env.cell_stimulus(inputs), dt=0.025)
    yield env, run
    with contextlib.suppress(Exception):
        env.close()


@pytest.fixture(scope="module")
def session(recorded, tmp_path_factory):
    from livn.data.rsf import store

    env, run = recorded
    path = str(tmp_path_factory.mktemp("rsf") / "session-1")
    store(run, path, env, voltage=True, currents=True)
    return path


def test_a_saved_run_reads_back(recorded, session):
    from livn.data.rsf import DEFAULT_GAIN_UV, load

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


def test_the_session_names_the_sections_in_its_traces(recorded, session):
    from livn.data.rsf import load

    _, run = recorded
    simulation = load(session).simulation
    recorded_sections = set(np.asarray(run.current_sections).tolist())
    contributing = set(simulation["trace_sections"])
    assert contributing
    # every recorded section is in the traces or said to be left out of them
    assert contributing | set(simulation.get("unplaced_sections", [])) == (
        recorded_sections
    )


def test_the_delivered_input_is_stored_as_a_series(recorded, tmp_path_factory):
    from livn.data.rsf import RsfRecording, store

    env, run = recorded
    inputs = np.zeros([DURATION, env.io.num_channels], dtype=np.float32)
    inputs[10:30, :] = AMPLITUDE
    path = str(tmp_path_factory.mktemp("rsf") / "session-inputs")
    store(run, path, env, inputs=inputs, inputs_dt=1.0)

    recording = RsfRecording(path)
    n = recording.num_samples(0)
    delivered = recording.inputs(0, 0, n)
    assert delivered.shape == (n, env.io.num_channels)
    # frame i is at i * dt ms and holds the 1 ms row it falls in
    rows = np.minimum(
        np.floor(np.arange(n) * 1000.0 / recording.fs + 1e-9).astype(int), DURATION - 1
    )
    np.testing.assert_array_equal(delivered, inputs[rows])
    assert delivered.max() == np.float32(AMPLITUDE)


def test_a_run_goes_through_the_arrow_layout(recorded, tmp_path_factory):
    pytest.importorskip("datasets")
    from livn.data import arrow
    from livn.data import open as open_recording

    env, run = recorded
    root = str(tmp_path_factory.mktemp("arrow") / "dataset")
    arrow.store(run, root, env=env, ground_truth=("spikes",))
    recording = open_recording(root)

    n_cells = len(np.unique(np.asarray(run.spike_ids)))
    spikes = recording.spikes(0)
    assert len(spikes.ids) == len(run.spike_ids)
    assert len(np.unique(spikes.ids)) == n_cells
    window = recording.read(0, 100, 200)
    assert window.shape == (200, env.io.num_channels)
    assert np.isfinite(window).all()
    assert recording.io.num_channels == env.io.num_channels
