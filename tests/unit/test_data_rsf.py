import json
import os

import numpy as np
import pytest

from livn.io import MEA, ComposedIO
from livn.run import Run, Series

pytest.importorskip("zarr", minversion="3")

from livn.data import Missing, rsf
from livn.data.rsf import RSFError, load, store

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
    expected = np.asarray(rsf._project(run, env, env.io)[0])
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
    assert loaded.io.serialize()["medium"] == env.io.serialize()["medium"]

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
        rsf._project(_run(rank_order=True), env, env.io)[0],
        rsf._project(_run(), env, env.io)[0],
        rtol=1e-5,
    )


def _noisy_env(seed=7):
    from livn.io import Noise

    return _Env(io=ComposedIO(None, [_Env().io, Noise(5.0, seed=seed)]))


def test_withholding_records_its_stages(tmp_path):
    session = str(tmp_path / "session")
    store(
        _run(),
        session,
        _noisy_env(),
        withhold=[
            {
                "mask": {
                    "fields": ["electrode_positions", "source_positions", "channel:1"]
                }
            },
        ],
    )

    traces = os.path.join(session, "traces.zarr")
    attrs = _attrs(traces)
    rows = attrs["annotations"]["rsf_provenance"]["rows"]
    assert sorted(rows) == ["0", "2"]
    stages = [next(iter(s["transform"])) for s in rows["0"]]
    assert stages == ["observe", "noise", "mask"]

    assert rows["0"][1]["transform"]["noise"]["seed"] == 7
    assert rows["0"][1]["transform"]["noise"]["params"] == {"sigma_uv": 5.0}

    loaded = load(session)
    assert loaded.io is None
    np.testing.assert_array_equal(loaded.run.potential.ids, [0, 2])
    assert [next(iter(s["transform"])) for s in loaded.record["rows"]["0"]] == stages


def test_the_traces_never_clip(tmp_path):
    with pytest.raises(RSFError, match="pass gain_uV >="):
        store(_run(), str(tmp_path / "session"), _Env(), gain_uV=1e-6)
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
        {"noise": {"distribution": "uniform", "params": {"sigma_uV": 1.0}}},
        {"filter": {}},
    ],
)
def test_unknown_withholding_is_refused(tmp_path, item):
    with pytest.raises(RSFError):
        store(_run(), str(tmp_path / "session"), _Env(), withhold=[item])


def test_the_stack_noise_is_applied_with_its_seed_and_regenerates(tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    store(_run(), str(a), _noisy_env(7))
    store(_run(), str(b), _noisy_env(7))
    store(_run(), str(c), _noisy_env(8))
    quiet = tmp_path / "quiet"
    store(_run(), str(quiet), _Env())

    def traces(path):
        return load(str(path)).run.potential.values

    np.testing.assert_array_equal(traces(a), traces(b))
    assert not np.array_equal(traces(a), traces(c))
    spread = (traces(a) - traces(quiet)).std()
    assert spread == pytest.approx(5.0, rel=0.2)


def test_a_stack_is_stored_and_read_back_whole(tmp_path):
    from livn.io import FIR, ElectrodeRC, Noise

    mea = _Env().io
    io = ComposedIO(None, [mea, ElectrodeRC(20.0), FIR([1.0, 0.5]), Noise(2.0, seed=3)])
    session = str(tmp_path / "session")
    store(_run(), session, _Env(io=io))

    observe = _doc(os.path.join(session, "traces.zarr", "rsf", "observe.json"))
    assert observe["kind"] == rsf.LIVN_IO
    loaded = load(session)
    assert loaded.io.serialize() == io.serialize()

    rows = loaded.record["rows"]["0"]
    assert [next(iter(s["transform"])) for s in rows] == [
        "observe",
        "transfer",
        "transfer",
        "noise",
    ]


def test_a_composition_is_stored_with_its_artifact_and_read_back_whole(tmp_path):
    from livn.conditions import Conditions
    from livn.io import (
        Chain,
        ElectrodeRC,
        ExtracellularPotential,
        LightArtifact,
        PatternedLight,
        Sum,
    )
    from livn.io.recording import electrode_light

    light = PatternedLight([[0, 0.0, 0.0, 60.0]], irradiance=2.0)
    artifact = LightArtifact([50.0], taus=(0.5,))
    io = ComposedIO(
        light,
        _Env().io,
        recording=Chain(
            Sum(extracellular=ExtracellularPotential(), artifact=artifact),
            rc=ElectrodeRC(20.0),
        ),
    )
    inputs = np.zeros((N_FRAMES, 1), dtype=np.float32)
    inputs[10:30] = 1.0
    lit, dark = str(tmp_path / "lit"), str(tmp_path / "dark")
    store(_run(), lit, _Env(io=io), inputs=inputs, inputs_dt=DT, currents=True)
    # the same cells without the light: the plain stack, in the RSF document
    plain = ComposedIO(light, [_Env().io, ElectrodeRC(20.0)])
    store(_run(), dark, _Env(io=plain))

    observe = _doc(os.path.join(lit, "traces.zarr", "rsf", "observe.json"))
    assert observe["kind"] == rsf.LIVN_IO
    assert observe["livn"]["recording"]["cls"] == "livn.io.Chain"
    assert observe["livn"]["inputs"]["cls"] == "livn.io.PatternedLight"
    assert (
        "recording"
        not in _doc(os.path.join(dark, "traces.zarr", "rsf", "observe.json"))["livn"]
    )
    loaded = load(lit)
    assert loaded.io.serialize() == io.serialize()

    # the light on electrode 0 is in its trace; without the inputs it is not
    gap = loaded.run.potential.values - load(dark).run.potential.values
    assert np.abs(gap[0]).max() > 10.0
    reach = light.reach(np.asarray(_Env().io.electrode_coordinates))
    want = ElectrodeRC(20.0)(
        artifact.response(electrode_light(inputs, reach), dt=DT), dt=DT
    )
    np.testing.assert_allclose(gap, want, atol=2 * rsf.DEFAULT_GAIN_UV)

    # the session's conditions project it again
    conditions = loaded.conditions()
    assert isinstance(conditions, Conditions) and conditions.dt == pytest.approx(DT)
    np.testing.assert_array_equal(conditions.inputs, inputs)
    again = loaded.io.observe(
        conditions.replace(neuron_coordinates=_Env().recording_coordinates())
    )
    np.testing.assert_allclose(
        again, loaded.run.potential.values, atol=rsf.DEFAULT_GAIN_UV
    )


def test_a_composition_that_reads_the_inputs_needs_them(tmp_path):
    from livn.io import (
        Chain,
        ExtracellularPotential,
        LightArtifact,
        PatternedLight,
        Sum,
    )

    io = ComposedIO(
        PatternedLight([[0, 0.0, 0.0, 60.0]]),
        _Env().io,
        recording=Chain(
            Sum(
                extracellular=ExtracellularPotential(),
                artifact=LightArtifact([1.0], taus=(0.5,)),
            )
        ),
    )
    with pytest.raises(ValueError, match="needs `inputs`"):
        store(_run(), str(tmp_path / "s"), _Env(io=io))


def test_a_recording_passed_to_store_is_the_one_stored(tmp_path):
    from livn.io import Chain, ElectrodeRC, ExtracellularPotential

    session = str(tmp_path / "s")
    store(
        _run(),
        session,
        _Env(),
        recording=Chain(extracellular=ExtracellularPotential(), rc=ElectrodeRC(20.0)),
    )

    observe = _doc(os.path.join(session, "traces.zarr", "rsf", "observe.json"))
    assert observe["kind"] == rsf.LIVN_IO
    assert list(observe["livn"]["recording"]["kwargs"]) == ["extracellular", "rc"]
    assert load(session).io.recording.names == ("extracellular", "rc")


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
    node = os.path.join(session, "traces.zarr", "rsf", "zarr.json")
    with open(node) as f:
        meta = json.load(f)
    del meta["attributes"]["finished_at"]
    with open(node, "w") as f:
        json.dump(meta, f)
    with pytest.raises(RSFError, match="not complete"):
        load(session)
    os.remove(node)
    with pytest.raises(RSFError, match="not an RSF session"):
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


class _AxonEnv(_Env):
    __module__ = "livn.backend.native.env"

    """Three sections per cell, a backend may record fewer of them"""

    def recording_sections(self):
        return np.array(["soma", "dend", "ais0"] * 3)

    def recording_coordinates(self, simulated_only=False):
        return np.array(
            [
                [g, x + dx, 5.0, 1.0]
                for g, x in ((0, 10.0), (1, 110.0), (2, 210.0))
                for dx in (0.0, 10.0, -40.0)
            ]
        )


def _currents(sections):
    ids = np.repeat([0, 1, 2], len(sections))
    names = np.tile(sections, 3)
    # a row's values depend on its gid and section, not on where it sits
    values = np.stack(
        [
            np.random.default_rng([int(g), sum(map(ord, str(s)))]).normal(
                scale=1e-3, size=N_FRAMES
            )
            for g, s in zip(ids, names, strict=True)
        ]
    ).astype(np.float32)
    return (
        Run(t0=0.0, duration=N_FRAMES * DT).add_current(
            ids, values, dt=DT, sections=names
        ),
        {(int(g), str(s)): v for g, s, v in zip(ids, names, values, strict=True)},
    )


def test_current_rows_pair_with_coordinates_by_section_name():
    env = _AxonEnv()
    run, rows = _currents(["ais0", "soma"])

    uv, sections = rsf._project(run, env, env.io)

    # each row at its own section's coordinate, not the first ones in order
    coordinates = env.recording_coordinates()
    expected = 0.0
    for (gid, name), values in rows.items():
        at = coordinates[gid * 3 + ["soma", "dend", "ais0"].index(name)][None]
        expected = expected + np.asarray(
            env.io.potential_recording(env.io.distances(at), values[None])
        )
    np.testing.assert_allclose(uv, expected, rtol=1e-4, atol=1e-6)
    assert sections == {"contributing": ["ais0", "soma"], "unplaced": []}

    # the order of the rows does not matter, only their names
    other, _ = _currents(["soma", "ais0"])
    np.testing.assert_allclose(
        rsf._project(other, env, env.io)[0], uv, rtol=1e-4, atol=1e-6
    )


def test_a_section_without_a_coordinate_is_left_out_and_named():
    env = _AxonEnv()
    run, _ = _currents(["soma", "ais9"])
    uv, sections = rsf._project(run, env, env.io)
    assert sections == {"contributing": ["soma"], "unplaced": ["ais9"]}
    np.testing.assert_allclose(
        uv, rsf._project(_currents(["soma"])[0], env, env.io)[0], rtol=1e-4, atol=1e-6
    )


def test_a_session_says_which_sections_are_in_its_traces(tmp_path):
    env = _AxonEnv()
    run, _ = _currents(["soma", "dend"])
    session = str(tmp_path / "session")
    store(run, session, env)
    extras = load(session).recording.ext("livn", "simulation.json")
    assert extras["trace_sections"] == ["soma", "dend"]
    assert "unplaced_sections" not in extras
    assert "trace_sections" not in load(session).simulation


# -- the lazy recording ------------------------------------------------------------------


def test_load_keeps_the_traces_on_disk(tmp_path):
    import pickle

    import zarr

    session = str(tmp_path / "session")
    store(_run(), session, _Env(), voltage=True)
    recording = load(session).recording

    # a zarr array, not a numpy copy of the whole trace
    assert isinstance(recording._traces, zarr.Array)
    assert recording.num_samples(0) == N_FRAMES
    assert recording.segments == 1 and recording.fs == pytest.approx(10000.0)

    window = recording.read(0, 10, 7)
    assert window.dtype == np.float32 and window.shape == (7, 3)
    np.testing.assert_array_equal(
        window, load(session).run.potential.values[:, 10:17].T
    )
    only = recording.read(0, 10, 7, channels=[2, 0])
    np.testing.assert_array_equal(only, window[:, [2, 0]])
    with pytest.raises(IndexError):
        recording.read(0, N_FRAMES - 3, 7)
    with pytest.raises(KeyError):
        recording.read(0, 0, 3, channels=[9])

    # a recording pickles by path and reopens
    again = pickle.loads(pickle.dumps(recording))
    np.testing.assert_array_equal(again.read(0, 10, 7), window)


def test_a_window_of_the_recording_is_a_run(tmp_path):
    session = str(tmp_path / "session")
    store(_run(), session, _Env(), voltage=True, currents=True)
    recording = load(session).recording
    whole = recording.to_run()
    part = recording.to_run(0, 10, 20)

    assert part.t0 == pytest.approx(10 * DT) and part.duration == pytest.approx(20 * DT)
    np.testing.assert_array_equal(
        part.potential.values, whole.potential.values[:, 10:30]
    )
    np.testing.assert_array_equal(part.voltage, whole.voltage[:, 10:30])
    # spikes in the window, relative to its start (the frames 3 and 49 fall outside)
    assert part.spike_ids.size == 0

    spikes = recording.spikes(0)
    np.testing.assert_array_equal(spikes.ids, [1, 0, 2, 1])
    np.testing.assert_allclose(spikes.times, [0.0, 0.3, 0.3, 4.9])
    np.testing.assert_array_equal(
        recording.ground_truth("currents").values, whole.current
    )
    with pytest.raises(Missing):
        recording.ground_truth("templates")


def test_a_session_without_spikes_says_so(tmp_path):
    session = str(tmp_path / "session")
    store(_run().drop_spikes(), session, _Env())
    with pytest.raises(Missing, match="spikes"):
        load(session).recording.spikes(0)
    assert load(session).run.spikes is None


def test_inputs_are_stored_on_the_frame_grid(tmp_path):
    inputs = np.arange(15, dtype=np.float32).reshape(5, 3)  # 1 ms rows, 3 channels
    session = str(tmp_path / "session")
    store(_run(), session, _Env(), inputs=inputs, inputs_dt=1.0)
    recording = load(session).recording

    # 0.1 ms frames hold each 1 ms row for ten frames; the run is 5 ms long
    got = recording.inputs(0, 0, N_FRAMES)
    assert got.dtype == np.float32 and got.shape == (N_FRAMES, 3)
    np.testing.assert_array_equal(got, np.repeat(inputs, 10, axis=0))
    np.testing.assert_array_equal(recording.inputs(0, 12, 3), got[12:15])
    assert recording.ext("livn", "simulation.json")["inputs"]["source_dt"] == 1.0


def test_a_policy_is_recorded_and_rendered(tmp_path):
    from livn.policy import BiphasicPulsePolicy

    policy = BiphasicPulsePolicy(
        n_channels=3,
        channels=[1],
        dt=0.1,
        amplitude_uA=2.0,
        phase_duration=0.2,
        interphase_gap=0.1,
        pulse_times=[0.0],
    )
    session = str(tmp_path / "session")
    store(_run(), session, _Env(), stimulus=policy)
    recording = load(session).recording

    simulation = load(session).simulation
    assert simulation["stimulus"]["cls"].endswith("BiphasicPulsePolicy")
    delivered = recording.inputs(0, 0, N_FRAMES)
    np.testing.assert_allclose(delivered[:, 1][:6], [-2, -2, 0, 2, 2, 0])
    assert not delivered[:, [0, 2]].any() and not delivered[6:].any()


def test_a_spec_only_stimulus_keeps_working_and_has_no_inputs(tmp_path):
    session = str(tmp_path / "session")
    store(_run(), session, _Env(), stimulus={"cls": "x.Y", "kwargs": {"a": 1}})
    recording = load(session).recording
    assert load(session).simulation["stimulus"] == {"cls": "x.Y", "kwargs": {"a": 1}}
    with pytest.raises(Missing, match="no input series"):
        recording.inputs(0, 0, 5)


def test_inputs_must_fit_the_stimulating_channels(tmp_path):
    with pytest.raises(RSFError, match="has 2 channels"):
        store(
            _run(), str(tmp_path / "a"), _Env(), inputs=np.zeros((5, 2)), inputs_dt=1.0
        )
    with pytest.raises(RSFError, match="inputs_dt"):
        store(_run(), str(tmp_path / "b"), _Env(), inputs=np.zeros((5, 3)))
    assert not os.path.exists(tmp_path / "a") and not os.path.exists(tmp_path / "b")


@pytest.fixture
def physical(tmp_path, physical_session):
    """``physical(name="physical", **kwargs)``: a hand-written session under ``tmp_path``, opened."""

    def make(name="physical", **kwargs):
        return rsf.RsfRecording(physical_session(str(tmp_path / name), **kwargs))

    return make


def test_a_livn_session_states_its_clock_losses_and_clips(tmp_path):
    session = str(tmp_path / "session")
    store(_run(t0=20.0), session, _Env())
    traces = os.path.join(session, "traces.zarr")
    recording = load(session).recording

    # a simulation's clock: one tick per frame, from the frame count at frame 0
    assert recording.first_sample_tick == 200 and recording.tick_rate == 10_000.0
    np.testing.assert_array_equal(recording.tick_at([0, 7]), [200, 207])
    np.testing.assert_array_equal(recording.frame_at([200, 207]), [0, 7])
    # an empty gaps table: nothing was lost; livn never clips
    np.testing.assert_array_equal(recording.gaps, np.zeros((0, 2)))
    np.testing.assert_array_equal(recording.clipped_samples, [0, 0, 0])
    np.testing.assert_array_equal(recording.valid_spans(0), [[0, N_FRAMES]])
    assert len(recording.stim_markers()) == 0

    # the operator's fields are the format's; the medium is livn's own, under rsf/ext/livn
    observe = _doc(os.path.join(traces, "rsf", "observe.json"))
    assert observe["kind"] == "rsf.point_source/1" and "livn" not in observe
    assert {"electrode_coordinates_um", "output_radius_um", "min_distance_um"} <= set(
        observe
    )
    assert recording.ext("livn", "observe.json")["medium"]["cls"] == "livn.io.Medium"
    assert "coordinates_um" in load(session).simulation["sources"]


def test_a_label_is_the_first_name_version(tmp_path):
    session = str(tmp_path / "session")
    store(_run(), session, _Env(), label="baseline")
    folder = os.path.join(session, "traces.zarr", "rsf", "notes", "label")
    (key,) = os.listdir(folder)
    version = _doc(os.path.join(folder, key))
    assert version["item"] == "session" and version["text"] == "baseline"
    assert key.startswith(rsf._instant(version["at"]).strftime("%Y%m%dT%H%M%S.%fZ"))
    assert load(session).recording.label == "baseline"
    # no label given: the system's name is the attribute, and no version is written
    store(_run(), str(tmp_path / "other"), _Env())
    assert not os.path.exists(
        os.path.join(tmp_path, "other", "traces.zarr", "rsf", "notes")
    )
    assert rsf.RsfRecording(str(tmp_path / "other")).label == "toy"


def test_int32_traces_hold_what_int16_cannot(tmp_path):
    run = _run()
    env = _Env()
    uv = np.asarray(rsf._project(run, env, env.io)[0])
    gain = float(np.abs(uv).max()) / 100_000.0  # a step int16 cannot reach the peak at
    with pytest.raises(RSFError, match="dtype='int32'"):
        store(run, str(tmp_path / "a"), env, gain_uV=gain)
    session = str(tmp_path / "b")
    store(run, session, env, gain_uV=gain, dtype="int32")
    recording = load(session).recording
    assert str(recording._traces.dtype) == "int32"
    np.testing.assert_allclose(recording.read(0, 0, N_FRAMES).T, uv, atol=gain)


def test_rows_read_through_their_own_gains_at_either_width(physical):
    raw = np.asarray([[100_000, -3], [-70_000, 4]], dtype=np.int32)
    recording = physical(traces=raw, gains=[0.0596, 0.5], dtype="int32", clipped=[0, 2])
    np.testing.assert_allclose(
        recording.read(0, 0, 2), raw * np.asarray([0.0596, 0.5]), rtol=1e-6
    )
    np.testing.assert_array_equal(recording.clipped_samples, [0, 2])
    np.testing.assert_array_equal(recording.row_property("band"), ["wide", "wide"])
    assert recording.row_property("impedance") is None
    # a store that does not count its clips says nothing about them
    assert physical("uncounted").clipped_samples is None


def test_ticks_and_frames_on_a_device_clock_of_another_rate(physical):
    # 25 kHz ticks at 10 kHz frames: a frame spans 2.5 ticks
    t0 = 912_384_050
    recording = physical(tick_rate=25_000.0, anchors=[(0, t0)])
    np.testing.assert_allclose(recording.tick_at([0, 1, 10]), [t0, t0 + 2.5, t0 + 25])
    np.testing.assert_allclose(recording.frame_at([t0, t0 + 5, t0 + 6]), [0, 2, 2.4])
    assert np.isnan(recording.frame_at([t0 - 1])[0])  # before the first anchor
    assert np.isnan(recording.frame_at([t0 + 2_500])[0])  # past the last frame
    assert recording.to_run(0, 0, 1).t0 == pytest.approx(
        t0 / 25.0
    )  # ms on the device clock
    np.testing.assert_allclose(recording.elapsed([0, 10]), [0.0, 0.001])


def test_ticks_beyond_float_precision_keep_their_differences(physical):
    t0 = 2**62 + 1  # past 2^53, where a float64 tick loses digits
    recording = physical(anchors=[(0, t0), (600, t0 + 700)])
    marker = np.asarray([t0 + 750], dtype=np.uint64)
    np.testing.assert_array_equal(recording.frame_at(marker), [650])
    np.testing.assert_allclose(recording.elapsed([650]), [0.075])


def test_a_gap_cuts_the_frames_and_moves_time(physical):
    # the clock states the jump: frames 0..599, then 100 frames lost, then 600..999
    t0 = 1_000
    recording = physical(anchors=[(0, t0), (600, t0 + 700)], gaps=[(600, 100)])
    np.testing.assert_array_equal(recording.valid_spans(0), [[0, 600], [600, 1000]])
    np.testing.assert_allclose(
        recording.elapsed([599, 600, 601]), [0.0599, 0.07, 0.0701]
    )
    np.testing.assert_allclose(recording.tick_at([599, 600]), [t0 + 599, t0 + 700])
    # a tick inside the gap is on no stored frame
    np.testing.assert_allclose(
        recording.frame_at([t0 + 599, t0 + 650, t0 + 700]), [599, np.nan, 600]
    )
    np.testing.assert_allclose(
        recording.frame_of_elapsed([0.0599, 0.065, 0.07]), [599, np.nan, 600]
    )

    # without a clock the gap is the only statement that time jumped
    unclocked = physical("unclocked", anchors=None, gaps=[(600, 100)])
    assert unclocked.first_sample_tick is None and np.isnan(unclocked.tick_at([0])[0])
    np.testing.assert_array_equal(unclocked.valid_spans(0), [[0, 600], [600, 1000]])
    np.testing.assert_allclose(unclocked.elapsed([599, 600]), [0.0599, 0.07])
    np.testing.assert_allclose(
        unclocked.frame_of_elapsed([0.0599, 0.065, 0.07]), [599, np.nan, 600]
    )


def test_dropped_spans_leave_the_other_rows_whole(physical):
    recording = physical(dropped=[(100, 50, "a-001"), (900, 0, "a-000")])
    np.testing.assert_array_equal(recording.valid_spans(0), [[0, 100], [150, 1000]])
    np.testing.assert_array_equal(
        recording.valid_spans(0, channels=["a-000"]), [[0, 1000]]
    )
    np.testing.assert_array_equal(
        recording.valid_spans(0, channels=["a-001"]), [[0, 100], [150, 1000]]
    )


def test_stim_markers_are_placed_on_frames(physical):
    t0 = 5_000
    recording = physical(
        tick_rate=20_000.0,
        anchors=[(0, t0)],
        markers=[(t0 + 40, "hs0-070"), (t0 + 3, "din-0"), (t0 - 2, "hs0-070")],
    )
    markers = recording.stim_markers()
    np.testing.assert_array_equal(markers.tick, [t0 - 2, t0 + 3, t0 + 40])
    np.testing.assert_allclose(markers.frame, [np.nan, 1.5, 20.0])
    np.testing.assert_array_equal(markers.channel, ["hs0-070", "din-0", "hs0-070"])


def test_the_current_name_description_and_notes(physical, rsf_note):
    item, other = "5b1f0c3e9a7d4e2b8c64d0a1f3e79b52", "0123456789abcdef0123456789abcdef"
    uid = "{:032x}".format
    notes = {
        "label": [
            # the later instant wins though it sorts earlier as text
            (
                f"20261004T100541.207311Z-{uid(1)}.json",
                rsf_note("session", "2026-10-04T10:05:41.207311Z", "Slice 3"),
            ),
            (
                f"20261004T103000.000000Z-{uid(2)}.json",
                rsf_note(
                    "session", "2026-10-04T12:30:00.000000+02:00", "Slice 3, baseline"
                ),
            ),
            # one instant, and the greater key wins the tie
            (
                f"20261004T103000.000000Z-{uid(3)}.json",
                rsf_note("session", "2026-10-04T10:30:00Z", "Slice 3, wash"),
            ),
        ],
        "description": [
            (
                f"20261004T100600.500000Z-{uid(4)}.json",
                rsf_note(
                    "session", "2026-10-04T10:06:00.5Z", "Cortical culture, DIV 21"
                ),
            ),
            ("not-a-key.json", rsf_note("session", "2026-10-04T11:00:00Z", "skipped")),
        ],
        "note": [
            (
                f"20261004T091250.100000Z-{uid(5)}.json",
                rsf_note(item, "2026-10-04T09:12:50.100000Z", "drug", frame=120),
            ),
            (
                f"20261004T091251.000000Z-{uid(6)}.json",
                rsf_note(item, "2026-10-04T09:12:51Z", "drug added", frame=120),
            ),
            (
                f"20261004T091300.000000Z-{uid(7)}.json",
                rsf_note(other, "2026-10-04T09:13:00Z", "washed out", frame=300),
            ),
            (
                f"20261004T100000.000000Z-{uid(8)}.json",
                rsf_note(other, "2026-10-04T10:00:00Z", "", removed=True, frame=300),
            ),
            # an item that does not suit its kind is skipped
            (
                f"20261004T100100.000000Z-{uid(9)}.json",
                rsf_note("session", "2026-10-04T10:01:00Z", "x"),
            ),
        ],
    }
    recording = physical(notes=notes, label="started as")
    assert recording.label == "Slice 3, wash"
    assert recording.metadata["label"] == "Slice 3, wash"
    assert recording.description == "Cortical culture, DIV 21"
    assert [(n["item"], n["text"], n["frame"]) for n in recording.notes] == [
        (item, "drug added", 120)
    ]
    # no versions: the name is the attribute, the description empty
    plain = physical("plain", label="started as")
    assert (plain.label, plain.description, plain.notes) == ("started as", "", [])
    # a removed name falls back to the attribute
    removed = {
        "label": [
            (
                f"20261004T100541.207311Z-{uid(1)}.json",
                rsf_note("session", "2026-10-04T10:05:41.207311Z", "", removed=True),
            )
        ]
    }
    assert physical("removed", notes=removed, label="first").label == "first"


def test_what_livn_cannot_read_is_refused(tmp_path, physical, physical_session):
    with pytest.raises(RSFError, match="not sorted"):
        physical("unsorted", gaps=[(600, 10), (300, 5)])
    path = physical_session(str(tmp_path / "format2"))
    node = os.path.join(path, "traces.zarr", "rsf", "zarr.json")
    meta = _doc(node)
    meta["attributes"]["format"] = 2
    with open(node, "w") as f:
        json.dump(meta, f)
    with pytest.raises(RSFError, match="format 2"):
        rsf.RsfRecording(path)
    with pytest.raises(RSFError, match="no tick anchor"):
        physical("markers", anchors=None, markers=[(5, "x")]).stim_markers()


def test_stimulation_records_are_read_as_written_and_not_interpreted(physical):
    """RSF defines a record's ``started_at`` only; the rest is the producer's, so the records
    come back raw and are not rendered as inputs."""
    written = [
        {"started_at": "2026-10-04T09:12:34Z", "program": {"segments": [[1, 5]]}},
        {"started_at": "2026-10-04T09:13:00Z"},
    ]
    recording = physical("records", stimulation=written)
    assert recording.stimulation_records() == [
        {**record, "index": i} for i, record in enumerate(written)
    ]
    with pytest.raises(
        Missing, match="neutral representation of the delivered stimulus"
    ):
        recording.inputs(0, 0, 10)
    assert physical("none").stimulation_records() == []


def test_the_rsf_conformance_fixtures_read(tmp_path):
    """Every valid session of RSF's conformance kit opens and reads (``python
    rsf_fixtures.py <dir>`` beside the spec; ``LIVN_RSF_FIXTURES=<dir>``)."""
    root = os.environ.get("LIVN_RSF_FIXTURES")
    if not root or not os.path.isfile(os.path.join(root, "cases.json")):
        pytest.skip("LIVN_RSF_FIXTURES names no generated RSF fixture directory")
    cases = _doc(os.path.join(root, "cases.json"))
    for name in cases["valid"]:
        if name == "in_progress":  # livn reads complete sessions only
            with pytest.raises(RSFError, match="not complete"):
                rsf.RsfRecording(os.path.join(root, name))
            continue
        recording = rsf.RsfRecording(os.path.join(root, name))
        n = recording.num_samples()
        assert recording.read(0, 0, min(n, 4)).shape[0] == min(n, 4), name
        assert recording.valid_spans(0)[-1, 1] == n, name
        recording.stim_markers()
    assert rsf.RsfRecording(os.path.join(root, "notes")).label == "Slice 3, baseline"
    for name in ("gaps_unsorted", "stim_marker_without_clock", "group_recording_id"):
        with pytest.raises(RSFError):
            recording = rsf.RsfRecording(os.path.join(root, "broken", name))
            recording.stim_markers()
