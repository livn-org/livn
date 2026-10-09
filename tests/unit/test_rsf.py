import json
import os

import numpy as np
import pytest

from livn.io import MEA, ComposedIO
from livn.run import Run, Series

pytest.importorskip("zarr", minversion="3")

from livn import rsf
from livn.rsf import RSFError, load, store

DT = 0.1
N_FRAMES = 50


class _System:
    name = "toy"

    def serialize(self):
        return {"total_cells": 3}


class _Env:
    seed = 11
    noise_stream = 2
    comm = None
    model = None

    def __init__(self, io=None):
        self.system = _System()
        self.io = io or MEA(
            [[0, 0.0, 0.0, 0.0], [1, 100.0, 0.0, 0.0], [2, 200.0, 0.0, 0.0]],
            output_radius=500.0,
        )

    def simulated_gids(self, everywhere=False):
        return np.array([0, 1, 2])

    def active_neuron_coordinates(self):
        return np.array(
            [[0, 10.0, 5.0, 1.0], [1, 110.0, 5.0, 1.0], [2, 210.0, 5.0, 1.0]]
        )

    def recording_coordinates(self, simulated_only=False):
        return np.array(
            [
                [0, 10.0, 5.0, 1.0],
                [0, 20.0, 5.0, 1.0],
                [1, 110.0, 5.0, 1.0],
                [1, 120.0, 5.0, 1.0],
                [2, 210.0, 5.0, 1.0],
                [2, 220.0, 5.0, 1.0],
            ]
        )


_Env.__module__ = "livn.backend.native.env"


def _run(t0=0.0, rank_order=False):
    rng = np.random.default_rng(0)
    ids = np.array([0, 0, 1, 1, 2, 2])
    sections = np.array(["soma", "dend"] * 3)
    current = rng.normal(scale=1e-3, size=(6, N_FRAMES)).astype(np.float32)
    voltage = rng.normal(-65.0, 1.0, size=(6, N_FRAMES)).astype(np.float32)
    if rank_order:
        order = np.array([4, 5, 0, 1, 2, 3])
        ids, sections, current, voltage = (
            ids[order],
            sections[order],
            current[order],
            voltage[order],
        )
    return (
        Run(t0=t0, duration=N_FRAMES * DT)
        .add_spikes(np.array([1, 0, 2, 1]), np.array([0.05, 0.31, 0.31, 4.95]))
        .add_voltage(ids, voltage, dt=DT, sections=sections)
        .add_current(ids, current, dt=DT, sections=sections)
    )


def _attrs(path):
    with open(os.path.join(path, "zarr.json")) as f:
        return json.load(f)["attributes"]


def _doc(path):
    with open(path) as f:
        return json.load(f)


def test_a_saved_run_reads_back(tmp_path):
    run, env = _run(), _Env()
    session = str(tmp_path / "session")

    assert store(run, session, env, voltage=True, currents=True) == session
    loaded = load(session)
    back = loaded.run

    assert back.t0 == 0.0 and back.duration == pytest.approx(N_FRAMES * DT)
    expected = np.asarray(rsf._project(run, env, env.io))
    np.testing.assert_allclose(
        back.potential.values, expected, atol=rsf.DEFAULT_GAIN_UV / 2
    )
    np.testing.assert_array_equal(back.potential.ids, [0, 1, 2])

    np.testing.assert_array_equal(back.voltage, run.voltage)
    np.testing.assert_array_equal(back.voltage_sections, run.voltage_sections)
    np.testing.assert_array_equal(back.current, run.current)
    np.testing.assert_array_equal(back.current_ids, run.current_ids)

    # spike times come back on the frame grid, ordered by frame and then by unit
    np.testing.assert_array_equal(back.spike_ids, [1, 0, 2, 1])
    np.testing.assert_allclose(back.spike_times, [0.0, 0.3, 0.3, 4.9])

    assert isinstance(loaded.io, MEA)
    np.testing.assert_allclose(
        loaded.io.electrode_coordinates, env.io.electrode_coordinates
    )
    assert (
        loaded.io.serialize()["volume_conductor"]
        == env.io.serialize()["volume_conductor"]
    )

    # what the format records beside the run comes back beside it
    assert loaded.record["space"] == "channel"
    assert loaded.simulation["seed"] == env.seed
    assert loaded.simulation["sources"]["ids"] == [0, 1, 2]


def test_the_documents_say_what_ran(tmp_path):
    session = str(tmp_path / "session")
    store(
        _run(t0=20.0), session, _Env(), stimulus={"cls": "a.Policy", "kwargs": {"x": 1}}
    )

    rsf = os.path.join(session, "traces.zarr", "rsf")
    group = _attrs(rsf)
    record = _attrs(os.path.join(session, "traces.zarr"))["annotations"][
        "rsf_provenance"
    ]
    simulation = _doc(os.path.join(rsf, "simulation.json"))

    assert group["backend_id"] == "livn_native"
    assert group["producer"] == {
        "name": "livn",
        "version": group["producer"]["version"],
        "component": "rsf",
    }
    # a system livn cannot identify has no spec, and is saved all the same
    assert record["env"]["system"] == {
        "simulation": {"simulator": "livn_native", "spec": None}
    }
    assert simulation["regenerate"]["system_spec"]["kwargs"] == {"total_cells": 3}
    assert (
        record["first_sample_tick"] == 200
    )  # the simulation clock, one tick per frame
    assert simulation["sources"]["ids"] == [0, 1, 2]
    # several recording coordinates per gid: the soma stands for the source
    assert simulation["sources"]["coordinates_um"][0] == [10.0, 5.0, 1.0]
    assert simulation["stimulus"] == {"cls": "a.Policy", "kwargs": {"x": 1}}
    assert simulation["regenerate"]["noise_stream"] == 2
    assert _doc(os.path.join(rsf, "stores.json")) == [
        {"path": "ground_truth/spikes.zarr", "space": "source", "written_by": "livn"}
    ]

    assert load(session).run.t0 == pytest.approx(20.0)


def test_rows_in_rank_order_project_like_rows_in_coordinate_order(tmp_path):
    env = _Env()
    # the sum runs in another order, so under jax's float32 it agrees to float32 precision
    np.testing.assert_allclose(
        rsf._project(_run(rank_order=True), env, env.io),
        rsf._project(_run(), env, env.io),
        rtol=1e-5,
    )


def test_withholding_records_its_stages(tmp_path):
    session = str(tmp_path / "session")
    store(
        _run(),
        session,
        _Env(),
        withhold=[
            {
                "mask": {
                    "fields": ["electrode_positions", "source_positions", "channel:1"]
                }
            },
            {"noise": {"distribution": "gaussian", "params": {"sigma_uv": 5.0}}},
        ],
    )

    traces = os.path.join(session, "traces.zarr")
    attrs = _attrs(traces)
    rows = attrs["annotations"]["rsf_provenance"]["rows"]
    assert sorted(rows) == ["0", "2"]
    stages = [next(iter(s["transform"])) for s in rows["0"]]
    assert stages == ["observe", "mask", "noise"]
    assert isinstance(
        rows["0"][2]["transform"]["noise"]["seed"], int
    )  # drawn and written

    loaded = load(session)
    assert loaded.io is None
    np.testing.assert_array_equal(loaded.run.potential.ids, [0, 2])
    assert [next(iter(s["transform"])) for s in loaded.record["rows"]["0"]] == stages


def test_the_traces_never_clip(tmp_path):
    with pytest.raises(RSFError, match="pass gain_uv >="):
        store(_run(), str(tmp_path / "session"), _Env(), gain_uv=1e-6)
    assert not os.path.exists(tmp_path / "session")


def test_a_run_without_a_channel_signal_is_refused(tmp_path):
    run = _run().drop_current()
    with pytest.raises(RSFError, match="record_membrane_current"):
        store(run, str(tmp_path / "session"), _Env())


def test_a_spike_outside_the_window_is_refused(tmp_path):
    run = _run().drop_spikes().add_spikes(np.array([0]), np.array([5.2]))
    with pytest.raises(RSFError, match="outside"):
        store(run, str(tmp_path / "session"), _Env())


def test_a_spike_at_the_window_end_lands_on_the_last_frame(tmp_path):
    session = str(tmp_path / "session")
    end = N_FRAMES * DT + 5e-11
    run = _run().drop_spikes().add_spikes(np.array([0]), np.array([end]))
    store(run, session, _Env())
    np.testing.assert_allclose(load(session).run.spike_times, [(N_FRAMES - 1) * DT])


def test_ground_truth_shares_the_frame_axis(tmp_path):
    run = _run()
    slow = run.add_voltage(
        run.voltage_ids, run.voltage[:, ::2], dt=2 * DT, sections=run.voltage_sections
    )
    with pytest.raises(RSFError, match="dt="):
        store(slow, str(tmp_path / "session"), _Env(), voltage=True)


def test_multi_compartment_currents_need_section_names(tmp_path):
    run = _run()
    unnamed = run.add_current(run.current_ids, run.current, dt=DT)
    with pytest.raises(RSFError, match="currents=False"):
        store(unnamed, str(tmp_path / "a"), _Env(), currents=True)
    store(unnamed, str(tmp_path / "b"), _Env(), voltage=True)


def test_a_location_is_written_once(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "x").write_text("")
    with pytest.raises(RSFError, match="not empty"):
        store(_run(), str(session), _Env())


def test_a_case_twin_is_refused(tmp_path):
    (tmp_path / "Session").mkdir()
    if os.path.exists(tmp_path / "session"):
        pytest.skip("case-insensitive file system")
    with pytest.raises(RSFError, match="case"):
        store(_run(), str(tmp_path / "session"), _Env())


@pytest.mark.parametrize(
    "item",
    [
        {"mask": {"fields": ["impedance"]}},
        {"mask": {"fields": ["channel:9"]}},
        {"noise": {"distribution": "uniform", "params": {"sigma_uv": 1.0}}},
        {"filter": {}},
    ],
)
def test_unknown_withholding_is_refused(tmp_path, item):
    with pytest.raises(RSFError):
        store(_run(), str(tmp_path / "session"), _Env(), withhold=[item])


def test_only_a_point_source_mea_is_an_operator(tmp_path):
    from livn.io import LightArray

    with pytest.raises(RSFError, match="LightArray"):
        store(
            _run(), str(tmp_path / "session"), _Env(io=LightArray([[0, 0.0, 0.0, 0.0]]))
        )


def test_a_composed_io_records_through_its_outputs(tmp_path):
    from livn.io import LightArray

    session = str(tmp_path / "session")
    mea = _Env().io
    store(_run(), session, _Env(io=ComposedIO(LightArray([[0, 0.0, 0.0, 0.0]]), mea)))
    observe = _doc(os.path.join(session, "traces.zarr", "rsf", "observe.json"))
    assert "stimulation_gain" not in observe and "input_radius_um" not in observe


def test_an_incomplete_session_is_not_read(tmp_path):
    session = str(tmp_path / "session")
    store(_run(), session, _Env())
    os.remove(os.path.join(session, "traces.zarr", "rsf", "zarr.json"))
    with pytest.raises(RSFError, match="not complete"):
        load(session)


def test_a_loaded_run_stores_again_without_projecting(tmp_path):
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    store(_run(), first, _Env())
    back = load(first).run
    store(back, second, _Env())

    np.testing.assert_allclose(load(second).run.potential.values, back.potential.values)


def test_off_the_root_rank_nothing_is_written(tmp_path):
    assert store(None, str(tmp_path / "session"), _Env()) is None
    assert not os.path.exists(tmp_path / "session")


def test_no_document_holds_nan(tmp_path):
    with pytest.raises(RSFError, match="NaN"):
        store(
            _run(),
            str(tmp_path / "session"),
            _Env(),
            stimulus={"amplitude": float("nan")},
        )
    assert not os.path.exists(tmp_path / "session")


def test_the_signal_is_not_a_cell_channel():
    potential = Series(ids=np.array([0, 1]), values=np.ones((2, 10)), dt=DT)
    run = Run(duration=1.0, potential=potential).add_spikes(
        np.array([0]), np.array([0.5])
    )
    assert run.select(gids=[5]).potential is potential
    assert run.drop_spikes().potential is potential


def test_zarr_2_is_refused_with_the_install_hint(tmp_path, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "zarr", types.SimpleNamespace(__version__="2.18.7")
    )
    with pytest.raises(ImportError, match="zarr>=3"):
        store(_run(), str(tmp_path / "session"), _Env())
    assert not os.path.exists(tmp_path / "session")
