import json
import os
import re

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


def test_the_author_is_written_only_when_given(tmp_path):
    named, unnamed = str(tmp_path / "named"), str(tmp_path / "unnamed")
    store(_run(), named, _Env(), author="Ada Lovelace")
    store(_run(), unnamed, _Env())

    assert _attrs(os.path.join(named, "traces.zarr", "rsf"))["author"] == "Ada Lovelace"
    assert load(named).recording.author == "Ada Lovelace"
    # no name from the OS user: absent means unknown
    assert "author" not in _attrs(os.path.join(unnamed, "traces.zarr", "rsf"))
    assert load(unnamed).recording.author is None


@pytest.mark.parametrize("author", ["", " \t ", "ada@example.org"])
def test_an_author_that_is_no_name_is_refused(tmp_path, author):
    with pytest.raises(RSFError, match="author"):
        store(_run(), str(tmp_path / "session"), _Env(), author=author)
    assert not os.path.exists(tmp_path / "session")


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


def test_a_sorting_beside_the_session_reads_like_its_ground_truth(tmp_path):
    import shutil

    session = str(tmp_path / "session")
    store(_run(), session, _Env())
    os.makedirs(os.path.join(session, "sortings"))
    shutil.copytree(
        os.path.join(session, "ground_truth", "spikes.zarr"),
        os.path.join(session, "sortings", "sorter-1.zarr"),
    )
    recording = load(session).recording

    assert recording.sortings() == ["sortings/sorter-1.zarr"]
    truth = recording.spikes()
    for name in ("sortings/sorter-1.zarr", "sorter-1"):
        sorted_ = recording.spikes(sorting=name)
        np.testing.assert_array_equal(sorted_.ids, truth.ids)
        np.testing.assert_array_equal(sorted_.times, truth.times)
    with pytest.raises(Missing, match="none of the sortings"):
        recording.spikes(sorting="sorter-2")


def test_inputs_are_stored_on_the_frame_grid(tmp_path):
    inputs = np.arange(15, dtype=np.float32).reshape(5, 3)  # 1 ms rows, 3 channels, uA
    session = str(tmp_path / "session")
    store(_run(), session, _Env(), inputs=inputs, inputs_dt=1.0)
    recording = load(session).recording

    # 0.1 ms frames hold each 1 ms row for ten frames; the run is 5 ms long
    got = recording.inputs(0, 0, N_FRAMES, physical=True)
    assert got.dtype == np.float32 and got.shape == (N_FRAMES, 3)
    np.testing.assert_allclose(got, np.repeat(inputs, 10, axis=0), rtol=1e-6)
    np.testing.assert_array_equal(recording.inputs(0, 12, 3, physical=True), got[12:15])
    (record,) = recording.stimulation_records()
    assert record["livn"]["source_dt"] == 1.0 and record["livn"]["units"] == "uA"
    assert record["livn"]["channels"] == [0, 1, 2]


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
    assert simulation["stimulus"]["record"] == 0
    # livn writes the cathodic phase first and negative; RSF states it positive, a
    # fraction of the MEA's 100 uA range, and physical=True turns it back
    np.testing.assert_allclose(
        recording.inputs(0, 0, 6)[:, 1], [0.02, 0.02, 0, -0.02, -0.02, 0], rtol=1e-6
    )
    delivered = recording.inputs(0, 0, N_FRAMES, physical=True)
    np.testing.assert_allclose(delivered[:, 1][:6], [-2, -2, 0, 2, 2, 0], rtol=1e-6)
    assert not delivered[:, [0, 2]].any() and not delivered[6:].any()
    assert rsf._livn_scale("electrode", 100.0) == -100.0
    assert rsf._livn_scale("light", 2.0) == 2.0

    rows = recording.stim_events()
    np.testing.assert_array_equal(rows.frame, [0, 2, 3, 5])
    np.testing.assert_array_equal(rows.event, [0, 1, 2, 3])
    assert set(rows.channel) == {"electrode/1"} and set(rows.timing) == {"simulated"}
    (record,) = recording.stimulation_records()
    assert record["kind"] == "electrode" and record["timing"] == "simulated"
    assert [c["id"] for c in record["channels"]] == [
        "electrode/0",
        "electrode/1",
        "electrode/2",
    ]
    assert record["channels"][2]["shape"] == {
        "kind": "contact",
        "contact_id": 2,
        "x": 200.0,
        "y": 0.0,
    }
    assert recording.stimulus_full_scales()["electrode/1"]["value"] == 100.0
    assert record["livn"]["stimulus"]["cls"].endswith("BiphasicPulsePolicy")


def test_a_spec_only_stimulus_keeps_working_and_has_no_inputs(tmp_path):
    session = str(tmp_path / "session")
    store(_run(), session, _Env(), stimulus={"cls": "x.Y", "kwargs": {"a": 1}})
    recording = load(session).recording
    assert load(session).simulation["stimulus"] == {"cls": "x.Y", "kwargs": {"a": 1}}
    with pytest.raises(Missing, match="no delivered stimulus"):
        recording.inputs(0, 0, 5)
    assert recording.stimulation_records() == [] and len(recording.stim_events()) == 0
    assert load(session).conditions().inputs is None


def test_inputs_must_fit_the_stimulating_channels(tmp_path):
    with pytest.raises(RSFError, match="has 2 channels"):
        store(
            _run(), str(tmp_path / "a"), _Env(), inputs=np.zeros((5, 2)), inputs_dt=1.0
        )
    with pytest.raises(RSFError, match="inputs_dt"):
        store(_run(), str(tmp_path / "b"), _Env(), inputs=np.zeros((5, 3)))
    # a current beyond the stimulator's range is not a fraction of it
    with pytest.raises(RSFError, match="max_current_uA"):
        store(
            _run(),
            str(tmp_path / "c"),
            _Env(),
            inputs=np.full((5, 3), 150.0),
            inputs_dt=1.0,
        )
    for name in "abc":
        assert not os.path.exists(tmp_path / name)

    wide = MEA(np.asarray(_Env().io.electrode_coordinates), max_current_uA=500.0)
    session = str(tmp_path / "wide")
    store(_run(), session, _Env(io=wide), inputs=np.full((5, 3), 150.0), inputs_dt=1.0)
    recording = load(session).recording
    np.testing.assert_allclose(recording.inputs(0, 0, 1), [[-0.3, -0.3, -0.3]])
    assert recording.stimulus_full_scales()["electrode/0"]["value"] == 500.0
    assert load(session).io.max_current_uA == 500.0


def _light_io(*wavelengths, irradiance=2.0):
    from livn.io import PatternedLight

    lights = [
        PatternedLight(
            [[3, 10.0 * k, 0.0, 30.0], [7, 50.0, 20.0, 40.0]],
            irradiance=irradiance * (k + 1),
            wavelength=w,
            labels=["near", "far"],
        )
        for k, w in enumerate(wavelengths)
    ]
    return ComposedIO(lights[0] if len(lights) == 1 else lights, _Env().io)


def test_a_light_session_round_trips(tmp_path):
    io = _light_io(470.0)
    inputs = np.zeros((N_FRAMES, 2), dtype=np.float32)
    inputs[5:20, 0] = 1.0
    inputs[10:N_FRAMES, 1] = 0.25  # still on at the end
    session = str(tmp_path / "lit")
    store(_run(), session, _Env(io=io), inputs=inputs, inputs_dt=DT)
    recording = load(session).recording

    np.testing.assert_array_equal(recording.inputs(0, 0, N_FRAMES), inputs)
    np.testing.assert_allclose(
        recording.inputs(0, 0, N_FRAMES, physical=True), inputs * 2.0
    )
    channels = recording.stimulus_channels()
    assert [c["id"] for c in channels] == ["light/3", "light/7"]
    assert channels[1]["shape"] == {
        "kind": "disc",
        "x": 50.0,
        "y": 20.0,
        "radius": 40.0,
    }
    assert channels[0]["label"] == "near" and channels[0]["wavelength_nm"] == 470.0
    assert channels[0]["full_scale"] == {"units": "mW/mm2", "value": 2.0}
    assert channels[0]["kind"] == "light" and channels[0]["index"] == 0

    # the program is closed at the store's length, the point after the last frame
    rows = recording.stim_events()
    closing = rows.frame == N_FRAMES
    assert set(rows.channel[closing]) == {"light/3", "light/7"}
    assert not rows.value[closing].any()
    assert rows.event[closing].tolist() == [rows.event.max()] * 2

    # the light reads back as the device that delivered it
    light = recording.stimulus_io()
    np.testing.assert_array_equal(
        np.asarray(light.spots)[:, 1:], np.asarray(io.inputs.spots)[:, 1:]
    )
    assert light.irradiance == 2.0 and light.wavelength == 470.0
    assert light.labels == ["near", "far"]


def test_several_devices_are_several_programs(tmp_path):
    io = _light_io(470.0, 590.0)
    inputs = np.zeros((N_FRAMES, 4), dtype=np.float32)
    inputs[5:20, 0] = 1.0
    inputs[8:12, 3] = 0.5  # the two programs overlap in time on their own channels
    session = str(tmp_path / "two")
    store(_run(), session, _Env(io=io), inputs=inputs, inputs_dt=DT)
    recording = load(session).recording

    records = recording.stimulation_records()
    assert [r["kind"] for r in records] == ["light", "light"]
    assert [c["id"] for c in recording.stimulus_channels()] == [
        "light/0/3",
        "light/0/7",
        "light/1/3",
        "light/1/7",
    ]
    assert [c["wavelength_nm"] for c in records[1]["channels"]] == [590.0, 590.0]
    assert load(session).simulation["stimulus"] == {"records": [0, 1]}
    rows = recording.stim_events()
    assert set(
        rows.program[np.char.startswith(rows.channel.astype(str), "light/1")]
    ) == {1}
    np.testing.assert_array_equal(recording.inputs(0, 0, N_FRAMES), inputs)

    # two wavelengths are two PatternedLights, in the order of the inputs' columns
    from livn.io import InputDevices

    devices = recording.stimulus_io()
    assert isinstance(devices, InputDevices) and devices.sizes == (2, 2)
    assert [d.wavelength for d in devices.devices] == [470.0, 590.0]
    assert [d.irradiance for d in devices.devices] == [2.0, 4.0]
    np.testing.assert_array_equal(
        devices.reach(np.asarray([[0, 0.0, 0.0, 0.0]])),
        io.inputs.reach(np.asarray([[0, 0.0, 0.0, 0.0]])),
    )

    # an electrode and a light: one record each
    from livn.io import PatternedLight

    mixed = ComposedIO(
        ["outputs", PatternedLight([[0, 0.0, 0.0, 30.0]], irradiance=1.5)], _Env().io
    )
    both = np.zeros((N_FRAMES, 4), dtype=np.float32)
    both[0:2, 1] = -4.0  # cathodic, 4 uA
    both[3:9, 3] = 1.0
    session = str(tmp_path / "mixed")
    store(_run(), session, _Env(io=mixed), inputs=both, inputs_dt=DT)
    recording = load(session).recording
    assert [r["kind"] for r in recording.stimulation_records()] == [
        "electrode",
        "light",
    ]
    assert recording.inputs(0, 0, 1)[0, 1] == pytest.approx(0.04)
    np.testing.assert_allclose(
        recording.inputs(0, 0, N_FRAMES, physical=True), both * [1, 1, 1, 1.5]
    )
    np.testing.assert_allclose(load(session).conditions().inputs, both, rtol=1e-6)
    devices = recording.stimulus_io()
    assert isinstance(devices.devices[0], MEA) and devices.sizes == (3, 1)
    np.testing.assert_array_equal(
        devices.devices[0].electrode_coordinates[:, :3],
        np.asarray(_Env().io.electrode_coordinates)[:, :3],
    )


def test_a_device_rsf_cannot_place_gets_a_record_without_channels(tmp_path):
    from livn.io import LightArray

    io = ComposedIO(LightArray([[0, 0.0, 0.0, 0.0]]), _Env().io)
    session = str(tmp_path / "fibre")
    store(_run(), session, _Env(io=io), inputs=np.ones((5, 1)), inputs_dt=1.0)
    recording = load(session).recording
    (record,) = recording.stimulation_records()
    assert "kind" not in record and record["livn"]["units"] == "mW"
    assert len(recording.stim_events()) == 0
    with pytest.raises(Missing, match="no delivered stimulus"):
        recording.inputs(0, 0, 5)


def test_a_masked_session_leaves_its_channels_unplaced(tmp_path):
    session = str(tmp_path / "masked")
    store(
        _run(),
        session,
        _Env(io=_light_io(470.0)),
        inputs=np.ones((5, 2), dtype=np.float32),
        inputs_dt=1.0,
        withhold=[{"mask": {"fields": ["electrode_positions"]}}],
    )
    recording = load(session).recording
    assert {c["shape"]["kind"] for c in recording.stimulus_channels()} == {"unplaced"}
    np.testing.assert_array_equal(recording.inputs(0, 0, 3), np.ones((3, 2)))
    with pytest.raises(ValueError, match="unplaced"):
        recording.stimulus_io()


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


def test_a_simulation_states_the_temperature_it_ran_at(tmp_path):
    plain = str(tmp_path / "plain")
    store(_run(), plain, _Env())
    assert not os.path.exists(os.path.join(plain, "traces.zarr", "rsf", "conditions"))
    recording = rsf.RsfRecording(plain)
    assert recording.environment() == {} and recording.prep is None

    env = _Env()
    env._celsius = 23.0  # what the neuron and native backends hold h.celsius at
    session = str(tmp_path / "warm")
    store(_run(), session, env)
    table = os.path.join(session, "traces.zarr", "rsf", "conditions")
    assert _attrs(table) == {"items": {"network": {"source": "livn_native"}}}
    recording = rsf.RsfRecording(session)
    held = recording.environment()[("temperature", "network")]
    assert (held.value, held.units, held.basis, held.origin, held.source) == (
        23.0,
        "C",
        "setpoint",
        "table",
        "livn_native",
    )
    np.testing.assert_array_equal(held.frames, [0])
    assert recording.prep is None
    assert not os.path.exists(
        os.path.join(session, "traces.zarr", "rsf", "notes", "condition")
    )


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


def _light_record(scale=None, lag=250, **extra):
    def spot(n, x):
        return {
            "id": f"light/s{n}",
            "shape": {"kind": "disc", "x": x, "y": 0.0, "radius": 25.0},
            "full_scale": {"units": "mW/mm2", "value": scale},
            "bipolar": False,
            "wavelength_nm": None,
            "label": f"spot {n}",
        }

    return {
        "started_at": "2026-10-04T09:12:34Z",
        "kind": "light",
        "channels": [spot(1, 0.0), spot(2, 100.0)],
        "marker_channel": "di-0",
        "marker_edge": "onset",
        "timing": "marker",
        "placement_lag_ms": lag,
        "acme": {"program": [1, 2]},
        **extra,
    }


def test_records_without_the_neutral_members_have_no_stimulus(physical):
    written = [
        {"started_at": "2026-10-04T09:12:34Z", "program": {"segments": [[1, 5]]}},
        {"started_at": "2026-10-04T09:13:00Z"},
    ]
    recording = physical("records", stimulation=written)
    assert recording.stimulation_records() == [
        {**record, "index": i} for i, record in enumerate(written)
    ]
    assert recording.stimulus_channels() == []
    with pytest.raises(Missing, match="no delivered stimulus"):
        recording.inputs(0, 0, 10)
    assert physical("none").stimulation_records() == []
    # a store from before stim_events: no table, no stimulus
    old = physical("old", stimulation=written, stim_events=None)
    assert len(old.stim_events()) == 0
    with pytest.raises(Missing):
        old.inputs(0, 0, 10)


def test_the_best_placed_row_wins_its_key(physical):
    rows = [
        (100, "light/s1", 1.0, 0, 0, "marker"),
        (105, "light/s1", 1.0, 0, 0, "schedule"),  # later in the table, placed worse
        (210, "light/s1", 0.0, 0, 1, "sidereal"),  # a source the spec does not define
        (205, "light/s1", 0.0, 0, 1, "schedule"),
        (300, "light/s2", 0.6, 0, 2, "schedule"),
        (302, "light/s2", 0.6, 0, 2, "schedule"),  # one source twice: the later row
        (400, "light/s2", 0.0, 0, 3, "from_marker"),
    ]
    recording = physical("supersede", stimulation=[_light_record()], stim_events=rows)
    raw = recording.stim_events(resolve=False)
    assert len(raw) == len(rows) and raw.timing[2] == "sidereal"
    resolved = recording.stim_events()
    np.testing.assert_array_equal(resolved.frame, [100, 205, 302, 400])
    np.testing.assert_array_equal(
        resolved.timing, ["marker", "schedule", "schedule", "from_marker"]
    )
    # the same rows in another order resolve alike
    shuffled = physical(
        "shuffled", stimulation=[_light_record()], stim_events=rows[::-1]
    ).stim_events()
    np.testing.assert_array_equal(shuffled.frame, [100, 205, 300, 400])

    got = recording.inputs(0, 0, 500)
    assert got.shape == (500, 2)
    assert not got[:100].any() and (got[100:205, 0] == 1).all()
    assert not got[205:, 0].any()
    np.testing.assert_allclose(got[302:400, 1], 0.6)
    assert not got[400:, 1].any() and not got[:302, 1].any()


def test_a_value_holds_across_a_window_without_events(physical):
    rows = [
        (10, "light/s1", 0.5, 0, 0, "marker"),
        (900, "light/s1", 0.0, 0, 1, "from_marker"),
    ]
    recording = physical("held", stimulation=[_light_record()], stim_events=rows)
    np.testing.assert_allclose(recording.inputs(0, 400, 20)[:, 0], 0.5)
    assert not recording.inputs(0, 400, 20)[:, 1].any()
    np.testing.assert_allclose(recording.inputs(0, 895, 10)[:, 0], [0.5] * 5 + [0] * 5)


def test_a_calibration_gives_a_light_its_full_scale(physical, rsf_note):
    rows = [
        (10, "light/s1", 0.5, 0, 0, "marker"),
        (10, "light/s2", 1.0, 0, 0, "marker"),
    ]

    def calibration(item, at, value, *, removed=False):
        version = rsf_note(item, at, "" if removed else "power meter", removed=removed)
        return {
            **version,
            "full_scale": {"units": "mW/mm2", "value": value},
            "measured_at": "2026-10-06T14:05:00Z",
            "method": None if removed else "power meter at the sample plane",
        }

    uid = "0123456789abcdef0123456789abcdef"
    notes = {
        "stimulus_calibration": [
            (
                f"20261006T142208.311904Z-{uid}.json",
                calibration("light/s1", "2026-10-06T14:22:08.311904Z", 12.4),
            ),
            (
                f"20261006T150000.000000Z-{uid}.json",
                calibration("light/s1", "2026-10-06T15:00:00.000000Z", 12.9),
            ),
            # a unit that is not the channel's is no calibration of it
            (
                f"20261006T160000.000000Z-{uid}.json",
                {
                    **calibration("light/s2", "2026-10-06T16:00:00.000000Z", 3.0),
                    "full_scale": {"units": "uA", "value": 3.0},
                },
            ),
        ]
    }
    recording = physical(
        "calibrated", stimulation=[_light_record()], stim_events=rows, notes=notes
    )
    scales = recording.stimulus_full_scales()
    assert scales["light/s1"]["value"] == 12.9
    assert scales["light/s1"]["source"] == "calibration"
    assert scales["light/s2"] == {
        "units": "mW/mm2",
        "value": None,
        "source": "record",
        "measured_at": None,
        "method": None,
        "text": None,
    }
    with pytest.raises(Missing, match="light/s2"):
        recording.inputs(0, 10, 1, physical=True)
    assert recording.stimulus_io().devices[0].irradiance == 12.9

    measured = physical(
        "measured",
        stimulation=[_light_record(scale=2.0)],
        stim_events=rows,
        notes=notes,
    )
    np.testing.assert_allclose(
        measured.inputs(0, 10, 1, physical=True), [[0.5 * 12.9, 2.0]]
    )
    # a removed calibration falls back to the record
    notes["stimulus_calibration"].append(
        (
            f"20261007T000000.000000Z-{uid}.json",
            calibration("light/s1", "2026-10-07T00:00:00.000000Z", None, removed=True),
        )
    )
    removed = physical(
        "removed", stimulation=[_light_record(scale=2.0)], stim_events=rows, notes=notes
    )
    assert removed.stimulus_full_scales()["light/s1"]["value"] == 2.0


def test_a_conditions_table_gives_each_item_its_value_and_basis(physical):
    rows = [
        (0, "temperature", "mea_base", 37.0, "C", "setpoint"),
        (100, "temperature", "mea_base", 36.8, "C", "measured"),
        (400, "temperature", "mea_base", 37.1, "C", "measured"),
        (400, "temperature", "mea_base", 37.2, "C", "measured"),  # one frame: the later
        (0, "temperature", "chamber", 30.0, "C", "measured"),
    ]
    items = {"mea_base": {"source": "MCS TC01 2434"}}
    recording = physical("conditions", conditions=rows, conditions_items=items)

    # before the first measurement only the setpoint holds
    early = recording.environment(50)[("temperature", "mea_base")]
    assert (early.value, early.basis, early.origin) == (37.0, "setpoint", "table")
    assert early.units == "C" and early.source == "MCS TC01 2434"
    # a measurement is taken before the setpoint, and holds until the next row
    later = recording.environment(399)[("temperature", "mea_base")]
    assert (later.value, later.basis) == (36.8, "measured")
    assert recording.environment(400)[("temperature", "mea_base")].value == 37.2
    assert (
        recording.environment(399, basis="setpoint")[("temperature", "mea_base")].value
        == 37.0
    )
    # over a window, the steps in effect and their frame-weighted mean
    window = recording.environment(start=300, length=200)
    held = window[("temperature", "mea_base")]
    np.testing.assert_array_equal(held.frames, [100, 400])
    np.testing.assert_allclose(held.values, [36.8, 37.2])
    assert held.value == pytest.approx((36.8 * 100 + 37.2 * 100) / 200)
    chamber = window[("temperature", "chamber")]
    assert (chamber.value, chamber.basis, chamber.source) == (30.0, "measured", None)
    assert list(recording.environment()) == [
        ("temperature", "chamber"),
        ("temperature", "mea_base"),
    ]
    # no table and no notes: unknown, not a value; an empty table states nothing either
    assert physical("plain").environment() == {}
    assert physical("empty", conditions=[]).environment(10) == {}


def _condition_note(rsf_note, item, at, value, basis, *, removed=False):
    return {
        **rsf_note(item, at, "" if removed else "held its setpoint", removed=removed),
        "value": None if removed else value,
        "units": "C",
        "basis": basis,
        "method": None if removed else "MCS TC01 2434 held its setpoint; not read",
    }


def test_a_condition_note_states_what_the_table_does_not(physical, rsf_note):
    uid = "{:032x}".format
    notes = {
        "condition": [
            (
                f"20261020T100000.000000Z-{uid(1)}.json",
                _condition_note(
                    rsf_note,
                    "temperature/mea_base",
                    "2026-10-20T10:00:00Z",
                    36.0,
                    "stated",
                ),
            ),
            (
                f"20261021T090000.000000Z-{uid(2)}.json",
                _condition_note(
                    rsf_note,
                    "temperature/mea_base",
                    "2026-10-21T09:00:00Z",
                    37.0,
                    "setpoint",
                ),
            ),
            # a removed item is not stated
            (
                f"20261021T090000.000000Z-{uid(3)}.json",
                _condition_note(
                    rsf_note, "co2/chamber", "2026-10-21T09:00:00Z", 5.0, "stated"
                ),
            ),
            (
                f"20261021T100000.000000Z-{uid(4)}.json",
                _condition_note(
                    rsf_note,
                    "co2/chamber",
                    "2026-10-21T10:00:00Z",
                    None,
                    "stated",
                    removed=True,
                ),
            ),
            # an item that is not <quantity>/<item> is skipped
            (
                f"20261021T110000.000000Z-{uid(5)}.json",
                _condition_note(
                    rsf_note, "temperature", "2026-10-21T11:00:00Z", 20.0, "stated"
                ),
            ),
        ]
    }
    recording = physical("noted", notes=notes)
    environment = recording.environment(10)
    assert list(environment) == [("temperature", "mea_base")]
    noted = environment[("temperature", "mea_base")]
    assert (noted.value, noted.units, noted.basis, noted.origin) == (
        37.0,
        "C",
        "setpoint",
        "note",
    )
    assert noted.method.startswith("MCS TC01 2434") and noted.author == "Mara Okafor"

    # the table's row wins where one is in effect, the note elsewhere
    both = physical(
        "both",
        notes=notes,
        conditions=[(500, "temperature", "mea_base", 36.9, "C", "measured")],
    )
    assert both.environment(499)[("temperature", "mea_base")].origin == "note"
    at = both.environment(500)[("temperature", "mea_base")]
    assert (at.value, at.basis, at.origin) == (36.9, "measured", "table")


def test_a_calibration_version_gives_a_channel_its_wavelength(physical, rsf_note):
    rows = [(10, "light/s1", 1.0, 0, 0, "marker")]
    uid = "{:032x}".format

    def calibration(item, at, value, **wavelength):
        return {
            **rsf_note(item, at, "Polygon1000, operator"),
            "full_scale": {"units": "mW/mm2", "value": value},
            "measured_at": None if value is None else "2026-10-06T14:05:00Z",
            "method": None if value is None else "power meter at the sample plane",
            **wavelength,
        }

    notes = {
        "stimulus_calibration": [
            (
                f"20261006T142208.311904Z-{uid(1)}.json",
                calibration("light/s1", "2026-10-06T14:22:08.311904Z", 12.4),
            ),
            # a later version restates the power and adds the wavelength
            (
                f"20261021T090000.000000Z-{uid(2)}.json",
                calibration(
                    "light/s1",
                    "2026-10-21T09:00:00Z",
                    12.4,
                    wavelength_nm=470.0,
                    wavelength_basis="stated",
                ),
            ),
            # the first version of a channel can be the wavelength alone
            (
                f"20261021T090000.000000Z-{uid(3)}.json",
                calibration(
                    "light/s2",
                    "2026-10-21T09:00:00Z",
                    None,
                    wavelength_nm=473.0,
                    wavelength_basis="measured",
                ),
            ),
        ]
    }
    record = _light_record()
    record["channels"][0]["wavelength_nm"] = 465.0
    record["channels"][0]["wavelength_basis"] = "device_default"
    recording = physical(
        "wavelength", stimulation=[record], stim_events=rows, notes=notes
    )
    assert recording.stimulus_wavelengths() == {
        "light/s1": {"value": 470.0, "basis": "stated", "source": "calibration"},
        "light/s2": {"value": 473.0, "basis": "measured", "source": "calibration"},
    }
    scales = recording.stimulus_full_scales()
    assert scales["light/s1"]["value"] == 12.4
    # a version with no measured power states the power unknown
    assert scales["light/s2"]["value"] is None
    assert scales["light/s2"]["source"] == "calibration"
    devices = recording.stimulus_io().devices
    assert [d.wavelength for d in devices] == [470.0, 473.0]

    # no versions: the record's wavelength and basis, else unknown
    plain = physical("plain-light", stimulation=[record], stim_events=rows)
    assert plain.stimulus_wavelengths() == {
        "light/s1": {"value": 465.0, "basis": "device_default", "source": "record"},
        "light/s2": {"value": None, "basis": None, "source": None},
    }


def test_the_preparation_says_what_it_is(physical):
    prep = {
        "id": "culture-8",
        "label": "sample 8",
        "kind": "culture",
        "species": "mouse",
        "age": {"value": 11, "units": "DIV"},
        "expression": [
            {"construct": "ChR2-YFP", "method": "line", "since": None},
            {"construct": "jGCaMP8", "method": "transfection", "since": "2026-10-13"},
        ],
    }
    recording = physical("prep", system={"prep": prep})
    assert recording.prep == {**prep, "age": {"value": 11.0, "units": "DIV"}}
    assert recording.metadata["prep"] == recording.prep
    # the members RSF 1 had alone: the rest is None
    old = physical("old-prep", system={"prep": {"id": "c1", "label": None}})
    assert old.prep == {
        "id": "c1",
        "label": None,
        "kind": None,
        "species": None,
        "age": None,
        "expression": None,
    }
    assert physical("no-system").prep is None


def test_a_derived_session_shows_when_its_source_was_recorded(physical, rsf_note):
    derived_from = {
        "recording_id": "1" * 32,
        "session": "../calibration2-sample8-DIV11",
        "started_at": "2026-10-20T14:02:11.000000Z",
        "created_at": "2026-10-20T14:02:10.000000Z",
        "changes": [
            {
                "what": "stim_events",
                "how": "retimed_from_digital_in",
                "params": {"marker_channel": "hs0-068", "tolerance_ms": 2.5},
            }
        ],
        "author": "Ada Lovelace",
        "acme": "kept, unknown",
    }
    carried = {
        **rsf_note(
            "5b1f0c3e9a7d4e2b8c64d0a1f3e79b52", "2026-10-20T14:10:00Z", "run", frame=5
        ),
        "carried_from": "1" * 32,
    }
    recording = physical(
        "derived",
        started_at="2026-10-21T08:00:00.000000Z",
        extra_attributes={
            "derived_from": derived_from,
            "zone": "Europe/Berlin",
            "author": "Charles Babbage",
        },
        notes={
            "note": [(f"20261020T141000.000000Z-{'2' * 32}.json", carried)],
        },
    )
    assert recording.derived_from == {
        k: derived_from[k]
        for k in (
            "recording_id",
            "session",
            "started_at",
            "created_at",
            "author",
            "changes",
        )
    }
    assert recording.started_at == "2026-10-20T14:02:11.000000Z"
    assert recording.metadata["started_at"] == recording.started_at
    assert recording.zone == "Europe/Berlin"
    assert recording.author == "Charles Babbage"
    assert recording.metadata["author"] == recording.author
    assert [(n["text"], n["carried_from"]) for n in recording.notes] == [
        ("run", "1" * 32)
    ]
    # a derived session from before these members: its own time, no changes
    older = physical(
        "derived-old",
        started_at="2026-10-21T08:00:00.000000Z",
        extra_attributes={"derived_from": {"recording_id": "1" * 32, "session": None}},
    )
    assert older.derived_from["changes"] is None
    assert older.derived_from["author"] is None
    assert older.started_at == "2026-10-21T08:00:00.000000Z"
    plain = physical("not-derived")
    assert plain.derived_from is None and plain.zone is None
    assert plain.author is None
    assert plain.started_at == plain.record["started_at"]


def _rig_channels(marks="light"):
    def bit(weight, **more):
        return {"weight": weight, "label": None, "source": None, "marks": None, **more}

    channels = {
        "e-ref": {"kind": "reference", "contact_id": "R", "label": "reference"},
        "aux-0": {"kind": "analog_in", "label": "aux 0", "units": "V", "scale": 1e-6},
        "aux-1": {"kind": "analog_in", "label": "aux 1", "units": "uV"},
        "din": {
            "kind": "digital_in",
            "label": "digital in",
            "bits": {
                "0": bit(4, source="light trigger out", marks=marks),
                "1": bit(8),
            },
        },
        "dout": {"kind": "digital_out", "label": "digital out"},
        "side": {"kind": "stim_sideband", "bits": {"0": bit(2, marks="electrode")}},
        "chk": {"kind": "checksum"},
        "clk-lo": {"kind": "clock", "part": "low"},
        "clk-hi": {"kind": "clock", "part": "high"},
    }
    return {"kind": "channels", "backend": "sim-rig", "channels": channels}


def test_the_channels_document_describes_what_is_not_an_electrode(physical):
    recording = physical(
        "rig",
        channels=("e-00", "e-01", "e-02", "din"),
        documents={"channels.json": _rig_channels()},
    )
    channels = recording.channels()
    assert not {"e-00", "e-01", "e-02"} & set(channels)  # electrodes are the wiring's
    assert len(channels) == 9
    kinds = [c["kind"] for c in channels.values()]
    assert kinds.count("analog_in") == 2 and kinds.count("clock") == 2
    assert channels["e-ref"]["contact_id"] == "R"
    assert (channels["aux-0"]["units"], channels["aux-0"]["scale"]) == ("V", 1e-6)
    assert channels["aux-1"]["scale"] is None
    assert [channels[f"clk-{p}"]["part"] for p in ("lo", "hi")] == ["low", "high"]
    trigger = channels["din"]["bits"][0]
    assert trigger == {
        "weight": 4,
        "label": None,
        "source": "light trigger out",
        "marks": "light",
        "carried_from": None,
    }
    assert channels["dout"]["bits"] == {}
    assert physical("plain").channels() == {}


def test_a_channel_version_restates_what_was_learnt_later(physical, rsf_note):
    uid = "{:032x}".format

    def version(item, at, **members):
        return {**rsf_note(item, at, ""), **members}

    notes = {
        "channel": [
            # the cable, declared the day after
            (
                f"20261021T090000.000000Z-{uid(1)}.json",
                version(
                    "din/0",
                    "2026-10-21T09:00:00Z",
                    source="light trigger out",
                    label="light trigger",
                    marks="light",
                    weight=64,  # a version never changes a weight
                ),
            ),
            # carried from the source session, a channel's units and scale
            (
                f"20261020T120000.000000Z-{uid(2)}.json",
                {
                    **version(
                        "aux-1",
                        "2026-10-20T12:00:00Z",
                        units="mW/mm2",
                        scale=0.002,
                        label="photodiode",
                        kind="digital_in",  # nor a kind
                    ),
                    "carried_from": "1" * 32,
                },
            ),
            # a version on a channel or a bit the document does not describe
            (
                f"20261021T090000.000000Z-{uid(3)}.json",
                version("nowhere/0", "2026-10-21T09:00:00Z", source="x", marks="light"),
            ),
            (
                f"20261021T090000.000000Z-{uid(4)}.json",
                version("din/5", "2026-10-21T09:00:00Z", source="x", marks="light"),
            ),
        ]
    }
    recording = physical(
        "learnt",
        documents={"channels.json": _rig_channels(marks=None)},
        notes=notes,
    )
    channels = recording.channels()
    assert channels["din"]["bits"][0] == {
        "weight": 4,
        "label": "light trigger",
        "source": "light trigger out",
        "marks": "light",
        "carried_from": None,
    }
    assert sorted(channels["din"]["bits"]) == [0, 1]
    photodiode = channels["aux-1"]
    assert (photodiode["kind"], photodiode["units"], photodiode["scale"]) == (
        "analog_in",
        "mW/mm2",
        0.002,
    )
    assert photodiode["label"] == "photodiode"
    assert photodiode["carried_from"] == "1" * 32
    assert "nowhere" not in channels
    # a version alone, without the document, describes nothing
    assert physical("versions-alone", notes=notes).channels() == {}


def test_the_marking_bit_of_a_program_is_found_by_marks(physical):
    rows = [
        (100, "light/s1", 1.0, 0, 0, "marker"),
        (200, "light/s1", 0.0, 0, 1, "marker"),
    ]
    traces = np.zeros((1000, 2), dtype=np.int16)
    traces[100:200, 1] = 4  # bit 0 alone, the light's
    electrode = {
        "started_at": "2026-10-04T09:13:00Z",
        "kind": "electrode",
        "channels": [
            {
                "id": "electrode/35",
                "shape": {"kind": "contact", "contact_id": "35", "x": 0.0, "y": 0.0},
                "full_scale": {"units": "uA", "value": 100.0},
                "bipolar": True,
                "wavelength_nm": None,
            }
        ],
        "marker_channel": None,
        "marker_edge": None,
        "timing": "schedule",
        "placement_lag_ms": 0,
    }
    recording = physical(
        "marked",
        channels=("e-00", "din"),
        traces=traces,
        stimulation=[_light_record(), electrode],
        stim_events=rows,
        documents={"channels.json": _rig_channels()},
    )
    light = recording.marking_bit(0)
    assert light == {
        "channel": "din",
        "bit": 0,
        "weight": 4,
        "label": None,
        "source": "light trigger out",
        "marks": "light",
        "gain_to_uV": 0.195,
    }
    # half the weight on the stored row, or on a scaled read once the gain is undone
    scaled = recording.read(0, 0, 1000, channels=[light["channel"]])[:, 0]
    raw = np.round(scaled / light["gain_to_uV"]).astype(np.int64)
    high = raw >= light["weight"] / 2
    np.testing.assert_array_equal(
        np.flatnonzero(np.diff(high.astype(int)) == 1) + 1, [100]
    )
    # the electrode program is marked by the sideband word, not a row of this store
    sideband = recording.marking_bit(recording.stimulation_records()[1])
    assert (sideband["channel"], sideband["bit"], sideband["gain_to_uV"]) == (
        "side",
        0,
        None,
    )

    # a bit naming one of the record's channels wins over one naming its kind
    document = _rig_channels()
    document["channels"]["side"]["bits"]["3"] = {
        "weight": 16,
        "label": None,
        "source": None,
        "marks": "light/s2",
    }
    named = physical(
        "named",
        stimulation=[_light_record()],
        stim_events=rows,
        documents={"channels.json": document},
    )
    assert (named.marking_bit(0)["channel"], named.marking_bit(0)["bit"]) == (
        "side",
        3,
    )
    # nothing marks it: None
    unmarked = physical(
        "unmarked",
        stimulation=[_light_record()],
        stim_events=rows,
        documents={"channels.json": _rig_channels(marks=None)},
    )
    assert unmarked.marking_bit(0) is None
    assert physical("bare", stimulation=[_light_record()]).marking_bit(0) is None


def test_a_simulation_describes_no_other_channels(tmp_path):
    session = str(tmp_path / "simulated")
    store(_run(), session, _Env())
    assert not os.path.exists(
        os.path.join(session, "traces.zarr", "rsf", "channels.json")
    )
    assert rsf.RsfRecording(session).channels() == {}


def test_a_growing_store_says_how_far_its_stimulus_is_final(
    physical, physical_session, tmp_path
):
    rows = [(100, "light/s1", 1.0, 0, 0, "schedule")]
    path = physical_session(
        str(tmp_path / "live"),
        stimulation=[_light_record(lag=25)],
        stim_events=rows,
        finished=False,
    )
    with pytest.raises(RSFError, match="growing=True"):
        rsf.RsfRecording(path)
    recording = rsf.RsfRecording(path, growing=True)
    # 1000 frames at 10 kHz, schedule rows up to 25 ms late
    assert recording.stimulus_final_to == 1000 - 250
    np.testing.assert_allclose(recording.inputs(0, 99, 2)[:, 0], [0.0, 1.0])
    assert recording.__getstate__()["growing"] is True
    done = physical("done", stimulation=[_light_record(lag=25)], stim_events=rows)
    assert done.stimulus_final_to == 1000


def test_the_reference_stimulus_session_reads(tmp_path):
    root = os.environ.get("LIVN_RSF_FIXTURES")
    if not root or not os.path.isdir(os.path.join(root or "", "stimulus")):
        pytest.skip("LIVN_RSF_FIXTURES names no generated stimulus session")
    recording = rsf.RsfRecording(os.path.join(root, "stimulus"))
    ids = [c["id"] for c in recording.stimulus_channels()]
    assert ids == ["light/s1-7f3a", "light/s2-7f3a", "electrode/35"]
    got = recording.inputs(0, 0, recording.num_samples())
    want = np.zeros_like(got)
    want[300:310, 0] = want[900:910, 0] = 1.0
    want[1500:1800, :2] = 0.6
    want[2400:2404, 2], want[2404:2408, 2] = 0.2, -0.2
    np.testing.assert_allclose(got, want)
    scales = recording.stimulus_full_scales()
    assert scales["light/s1-7f3a"]["value"] == 12.9
    assert scales["light/s2-7f3a"]["value"] is None
    assert scales["electrode/35"]["value"] == 100.0
    devices = recording.stimulus_io()
    assert devices.sizes == (1, 1, 1)  # spot 1 measured, spot 2 relative, the contact


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


def _two_light_programs(stop=1.6):
    first = np.zeros((20, 2), dtype=np.float32)
    first[2:7, 0] = 1.0
    second = np.zeros((30, 2), dtype=np.float32)
    second[0:5, 1] = 0.5
    second[10:15, 1] = 0.25
    second[18:25, 1] = 0.75
    origin = {
        "generator": "example.SpotProtocol",
        "software": {"name": "example", "version": "1.0"},
        "params": {"block": 0, "hash": "ab12"},
        "seed": 4,
    }
    return [
        {"inputs": first, "inputs_dt": DT, "origin": origin},
        {
            "inputs": second,
            "inputs_dt": DT,
            "start": 1.0,
            "stop": stop,
            "members": {"example": {"round": 2}},
        },
    ]


def test_each_program_is_a_record_with_its_plan(tmp_path):
    session = str(tmp_path / "programs")
    store(_run(), session, _Env(io=_light_io(470.0)), programs=_two_light_programs())
    recording = load(session).recording

    first, second = recording.stimulation_records()
    # each record states the channels its program drives, by site name and instrument
    assert [c["id"] for c in first["channels"]] == ["light/near"]
    assert [c["id"] for c in second["channels"]] == ["light/far"]
    assert first["channels"][0]["name"] == "near"
    assert first["channels"][0]["source"] == "livn PatternedLight"
    assert first["channels"][0]["shape"] == {
        "kind": "disc",
        "x": 0.0,
        "y": 0.0,
        "radius": 30.0,
    }
    assert first["origin"]["generator"] == "example.SpotProtocol"
    assert first["origin"]["params"] == {"block": 0, "hash": "ab12"}
    assert "origin" not in second and second["example"] == {"round": 2}
    assert second["livn"]["device"] == 0 and second["livn"]["channels"] == [7]
    assert load(session).simulation["stimulus"] == {"records": [0, 1]}

    # the plan, from each program's start, closing with an event that zeroes the channel
    plan = recording.planned(1)
    np.testing.assert_allclose(plan.time, [0.0, 0.5, 1.0, 1.5, 1.8, 2.5])
    np.testing.assert_allclose(plan.value, [0.5, 0, 0.25, 0, 0.75, 0])
    np.testing.assert_array_equal(plan.event, np.arange(6))
    np.testing.assert_array_equal(plan.load, np.zeros(6))
    assert set(plan.channel) == {"light/far"}
    assert recording.planned(0).event.tolist() == [0, 1]

    # program 0 delivered its plan; program 1 was stopped at its frame 16, frame 26
    assert recording.check_delivered(0)["ok"]
    report = recording.check_delivered(1)
    assert report["ok"] and not report["mistimed"] and not report["missing"]
    assert report["stop"] == {
        "frame": 26,
        "event": 5,
        "undelivered": [(4, "light/far")],
    }
    assert report["aligned"] == [
        {"load": 0, "event": 0, "frame": 10, "timing": "simulated"}
    ]
    rows = recording.stim_events()
    mine = rows.program == 1
    np.testing.assert_array_equal(rows.frame[mine], [10, 15, 20, 25, 26])
    np.testing.assert_array_equal(rows.event[mine], [0, 1, 2, 3, 5])  # no new ordinal

    # the delivered stimulus holds each program's values; the two overlap on two spots
    got = recording.inputs(0, 0, N_FRAMES)
    want = np.zeros((N_FRAMES, 2), dtype=np.float32)
    want[2:7, 0] = 1.0
    want[10:15, 1], want[20:25, 1], want[25:26, 1] = 0.5, 0.25, 0.0
    np.testing.assert_allclose(got, want)


def test_a_check_reports_what_does_not_match_the_plan(tmp_path):
    session = str(tmp_path / "programs")
    store(_run(), session, _Env(io=_light_io(470.0)), programs=_two_light_programs())
    # the same session with one delivered row moved and one value changed
    events = os.path.join(session, "traces.zarr", "rsf", "stim_events")
    import zarr

    table = zarr.open_group(events, mode="r+")
    frames, values = table["frame"][...], table["value"][...]
    frames[1] += 3  # program 0's offset
    values[2] = 0.4  # program 1's first value
    table["frame"][...] = frames
    table["value"][...] = values
    recording = rsf.RsfRecording(session)
    report = recording.check_delivered(0)
    assert report["mistimed"] == [(1, "light/near", 10, 7)] and not report["ok"]
    assert recording.check_delivered(1)["values"] == [
        (0, "light/far", 0.5, pytest.approx(0.4))
    ]
    # a record without a plan is not compared
    plain = str(tmp_path / "plain")
    store(_run(), plain, _Env(), inputs=np.ones((5, 3)), inputs_dt=1.0)
    assert rsf.RsfRecording(plain).planned(0) is None
    assert rsf.RsfRecording(plain).check_delivered(0)["planned"] is False


def test_a_policy_program_states_its_origin(tmp_path):
    from livn.policy import BiphasicPulsePolicy

    policy = BiphasicPulsePolicy(
        n_channels=3,
        channels=[1],
        dt=0.1,
        amplitude_uA=2.0,
        phase_duration=0.2,
        interphase_gap=0.1,
        pulse_times=[0.5],
    )
    session = str(tmp_path / "policy")
    store(_run(), session, _Env(), programs=[{"policy": policy}])
    recording = load(session).recording
    (record,) = recording.stimulation_records()
    assert record["origin"] == {
        "generator": "livn.policy.BiphasicPulsePolicy",
        "software": {"name": "livn", "version": rsf._version()},
        "params": json.loads(json.dumps(policy.serialize())),
        "seed": None,
    }
    assert [(c["id"], c["name"], c["source"]) for c in record["channels"]] == [
        ("electrode/1", "1", "livn MEA")
    ]
    assert recording.check_delivered(0)["ok"]
    delivered = recording.inputs(0, 0, N_FRAMES, physical=True)
    np.testing.assert_allclose(delivered[5:11, 0], [-2, -2, 0, 2, 2, 0], rtol=1e-6)


def test_programs_that_cannot_be_one_session_are_refused(tmp_path):
    io = _light_io(470.0)
    on = np.ones((10, 2), dtype=np.float32) * [1.0, 0.0]
    with pytest.raises(RSFError, match="one program at a time"):
        store(
            _run(),
            str(tmp_path / "a"),
            _Env(io=io),
            programs=[
                {"inputs": on, "inputs_dt": DT},
                {"inputs": on, "inputs_dt": DT, "start": 0.5},
            ],
        )
    with pytest.raises(RSFError, match="livn's own"):
        store(
            _run(),
            str(tmp_path / "b"),
            _Env(io=io),
            programs=[{"inputs": on, "inputs_dt": DT, "members": {"livn": {}}}],
        )
    with pytest.raises(RSFError, match="the format's"):
        store(
            _run(),
            str(tmp_path / "c"),
            _Env(io=io),
            programs=[{"inputs": on, "inputs_dt": DT, "members": {"origin": {}}}],
        )
    with pytest.raises(RSFError, match="instead of stimulus"):
        store(
            _run(),
            str(tmp_path / "d"),
            _Env(io=io),
            inputs=on,
            inputs_dt=DT,
            programs=[{"inputs": on, "inputs_dt": DT}],
        )
    with pytest.raises(RSFError, match="sets channels that are not"):
        store(
            _run(),
            str(tmp_path / "e"),
            _Env(io=io),
            programs=[{"inputs": on, "inputs_dt": DT, "channels": [7]}],
        )
    for name in "abcde":
        assert not os.path.exists(tmp_path / name)


def _prep(**extra):
    return {
        "id": "sim-culture-1",
        "label": "simulated culture 1",
        "kind": "culture",
        "species": "rat",
        "expression": [
            {
                "construct": "ChR2H134R",
                "method": "transduction",
                "since": None,
                "age": {"value": 7, "units": "DIV"},
            }
        ],
        "models": "c-2026-0412",
        **extra,
    }


def test_sessions_of_one_simulated_preparation_read_as_one(tmp_path):
    day15 = store(
        _run(),
        str(tmp_path / "day15"),
        _Env(),
        prep=_prep(age={"value": 15, "units": "DIV"}),
    )
    day14 = store(
        _run(),
        str(tmp_path / "day14"),
        _Env(),
        prep=_prep(age={"value": 14, "units": "DIV"}),
    )
    other = store(_run(), str(tmp_path / "other"), _Env(), prep={"label": "another"})
    a, b, c = (rsf.RsfRecording(p) for p in (day15, day14, other))

    # the run record repeats the record's prep, so it stands alone
    system = a.record["env"]["system"]["simulation"]
    assert system["prep"] == a.simulation["prep"]
    assert a.substrate == "simulated" and a.prep["models"] == "c-2026-0412"
    assert a.prep["expression"][0]["age"] == {"value": 7.0, "units": "DIV"}
    assert a.prep["age"] == {"value": 15.0, "units": "DIV"}

    assert rsf.same_preparation(a, b) and not rsf.same_preparation(a, c)
    # a simulated id is minted when none is given, and never the one it models
    assert re.fullmatch(r"[0-9a-f]{32}", c.prep["id"]) and c.prep["models"] is None
    groups = rsf.preparations([a, c, b])
    assert [r.path for r in groups[("simulated", "sim-culture-1")]] == [day14, day15]
    assert len(groups) == 2

    with pytest.raises(RSFError, match="never its own id"):
        store(_run(), str(tmp_path / "self"), _Env(), prep={"id": "x", "models": "x"})
    with pytest.raises(RSFError, match="prep has"):
        store(_run(), str(tmp_path / "bad"), _Env(), prep={"day": 3})


def test_a_run_epoch_over_two_blocks_joins_across_sessions(tmp_path):
    run_item = "c0ffee" + "0" * 26
    epochs = [
        {
            "item": run_item,
            "frame": 0,
            "end_frame": N_FRAMES,
            "kind": "protocol",
            "text": "spot mapping run",
            "programs": [0, 1],
            "origin": {
                "generator": "example.Protocol",
                "software": {"name": "example", "version": None},
                "params": {"hash": "ab12"},
                "seed": None,
            },
        },
        {
            "key": "block 0",
            "frame": 0,
            "end": 1.0,
            "kind": "block",
            "parent": run_item,
            "programs": [0],
        },
        {
            "key": "block 1",
            "start": 1.0,
            "end_frame": N_FRAMES,
            "kind": "block",
            "parent": run_item,
            "programs": [1],
        },
    ]
    first = store(
        _run(),
        str(tmp_path / "day1"),
        _Env(io=_light_io(470.0)),
        programs=_two_light_programs(),
        epochs=epochs,
    )
    second = store(
        _run(),
        str(tmp_path / "day2"),
        _Env(io=_light_io(470.0)),
        programs=_two_light_programs()[:1],
        epochs=[
            {**epochs[0], "programs": [0], "end_frame": 30},
            {
                "key": "the block",
                "frame": 0,
                "end_frame": 30,
                "kind": "block",
                "parent": run_item,
            },
        ],
    )
    one, two = rsf.RsfRecording(first), rsf.RsfRecording(second)

    spans = one.epochs()
    assert [(e["kind"], e["frame"], e["end_frame"]) for e in spans] == [
        ("block", 0, 10),
        ("protocol", 0, N_FRAMES),
        ("block", 10, N_FRAMES),
    ]
    run = spans[1]
    assert run["item"] == run_item and run["programs"] == [0, 1]
    assert run["origin"]["params"] == {"hash": "ab12"} and not run["open"]
    assert {e["parent"] for e in (spans[0], spans[2])} == {run_item}
    assert [e["kind"] for e in one.epoch_of(1)] == ["protocol", "block"]
    # without programs, a record belongs to the epochs holding its first frame
    assert sorted(e["kind"] for e in two.epoch_of(0)) == ["block", "protocol"]

    joined = rsf.join_epochs([one, two])
    (entry,) = [e for e in joined if e["item"] == run_item]
    assert entry["text"] == "spot mapping run"
    assert [(s["frame"], s["end_frame"], s["programs"]) for s in entry["sessions"]] == [
        (0, N_FRAMES, [0, 1]),
        (0, 30, [0]),
    ]
    assert [s["recording_id"] for s in entry["sessions"]] == [
        one.record["recording_id"],
        two.record["recording_id"],
    ]
    assert len(joined) == 4  # the run once, and the three blocks


def test_epochs_that_are_not_spans_of_the_session_are_refused(tmp_path):
    def attempt(name, epochs, **kwargs):
        store(_run(), str(tmp_path / name), _Env(), epochs=epochs, **kwargs)

    with pytest.raises(RSFError, match="outside its parent"):
        attempt(
            "nest",
            [
                {"key": "run", "frame": 10, "end_frame": 20, "kind": "protocol"},
                {
                    "key": "b",
                    "frame": 5,
                    "end_frame": 15,
                    "kind": "block",
                    "parent": "run",
                },
            ],
        )
    with pytest.raises(RSFError, match="names no epoch"):
        attempt("parent", [{"frame": 0, "kind": "block", "parent": "nobody"}])
    with pytest.raises(RSFError, match="not an index into programs"):
        attempt("programs", [{"frame": 0, "kind": "block", "programs": [0]}])
    with pytest.raises(RSFError, match=r"frame of 0\.\.50"):
        attempt("bounds", [{"frame": 0, "end_frame": N_FRAMES + 1, "kind": "block"}])
    with pytest.warns(UserWarning, match="not one of"):
        attempt("kind", [{"frame": 0, "kind": "wash_in"}])


def test_an_epoch_whose_parent_is_missing_reads_at_the_top(tmp_path):
    session = store(
        _run(),
        str(tmp_path / "orphan"),
        _Env(),
        epochs=[{"frame": 3, "kind": "block", "parent": "a" * 32}],
    )
    (epoch,) = rsf.RsfRecording(session).epochs()
    assert epoch["parent"] is None and epoch["parent_missing"]
    assert (
        epoch["open"] and epoch["end_frame"] == N_FRAMES and epoch["programs"] is None
    )


def test_the_description_notes_zone_and_other_documents_are_written(tmp_path):
    session = store(
        _run(),
        str(tmp_path / "noted"),
        _Env(),
        description="a simulated day",
        notes=[{"text": "bath change", "frame": 12, "author": "Ada"}],
        zone="Europe/Berlin",
        ext={"example": {"day.json": {"drift": [0.1, 0.2]}}},
    )
    recording = rsf.RsfRecording(session)
    assert (
        recording.description == "a simulated day" and recording.zone == "Europe/Berlin"
    )
    (note,) = recording.notes
    assert (note["text"], note["frame"], note["author"]) == ("bath change", 12, "Ada")
    assert note["zone"] == "Europe/Berlin"
    assert recording.ext("example", "day.json") == {"drift": [0.1, 0.2]}
    with pytest.raises(RSFError, match="livn's own"):
        store(_run(), str(tmp_path / "x"), _Env(), ext={"livn": {"a.json": {}}})
    with pytest.raises(RSFError, match="plainly"):
        store(_run(), str(tmp_path / "y"), _Env(), ext={"example": {"../a.json": {}}})
    with pytest.raises(RSFError, match="IANA"):
        store(_run(), str(tmp_path / "z"), _Env(), zone="Mars/Olympus")


class _CultureSystem(_System):
    def __init__(self):
        self.population_ranges = {"EXC": (0, 2), "INH": (2, 1)}
        self.connections_config = {
            "synapses": {
                "EXC": {
                    "EXC": {"type": "excitatory"},
                    "INH": {"type": "inhibitory"},
                    "STIM": {"type": "excitatory"},
                },
                "INH": {"EXC": {"type": "excitatory"}},
            }
        }


class _OpsinModel:
    def opsin_config(self):
        return {
            "mechanism": "ChR2H134R",
            "populations": ["EXC"],
            "params": {"gmax": 5e-4},
        }


class _CultureEnv(_Env):
    def __init__(self, io=None):
        super().__init__(io)
        self.system = _CultureSystem()
        self.model = _OpsinModel()

    def connections(self):
        names = np.asarray
        return {
            "pre_gid": np.array([0, 1, 2, 7]),
            "post_gid": np.array([1, 2, 0, 0]),
            "syn_id": np.arange(4),
            "pre_population": names(["EXC", "EXC", "INH", "STIM"], dtype=object),
            "post_population": names(["EXC", "INH", "EXC", "EXC"], dtype=object),
            "receptor": names(["AMPA", "AMPA", "GABA_A", "AMPA"], dtype=object),
            "strength": np.ones(4),
            "weight": np.array([0.5, 0.4, 1.2, 0.3]),
            "delay": np.array([1.0, 1.5, 0.8, 0.0]),
        }


_CultureEnv.__module__ = "livn.backend.native.env"


def test_the_ground_truth_tables_round_trip(tmp_path):
    io = _light_io(470.0)
    env = _CultureEnv(io=io)
    session = store(
        _run(),
        str(tmp_path / "truth"),
        env,
        programs=_two_light_programs(),
        populations=True,
        expression=True,
        connections=True,
        stimulus_coupling=True,
    )
    recording = rsf.RsfRecording(session)

    sources = recording.sources()
    np.testing.assert_array_equal(sources["ids"], [0, 1, 2])
    assert sources["populations"] == ["EXC", "EXC", "INH"]
    construct = sources["expression"]["ChR2H134R"]
    assert construct["units"] == "S/cm2"
    np.testing.assert_allclose(construct["values"], [5e-4, 5e-4, 0.0])

    table = recording.connections()
    np.testing.assert_array_equal(table["pre"], ["0", "1", "2", "STIM/7"])
    np.testing.assert_array_equal(table["post"], ["1", "2", "0", "0"])
    np.testing.assert_array_equal(table["sign"], [1, 1, -1, 1])
    np.testing.assert_allclose(table["weight"], [0.5, 0.4, 1.2, 0.3])
    np.testing.assert_allclose(table["delay"], [1.0, 1.5, 0.8, 0.0])
    assert table["units"] == "1" and table["inputs"] == ["STIM/7"]
    assert table["reference"] == rsf.CONNECTION_WEIGHT_REFERENCE
    np.testing.assert_array_equal(table["receptor"], ["AMPA", "AMPA", "GABA_A", "AMPA"])

    coupling = recording.stimulus_coupling()
    assert coupling["channel_ids"] == ["light/near", "light/far"]
    assert coupling["source_ids"] == ["0", "1", "2"] and coupling["units"] == "mW/mm2"
    soma = np.asarray(env.active_neuron_coordinates())
    np.testing.assert_allclose(
        coupling["gain"], np.asarray(io.inputs.reach(soma)), rtol=1e-6
    )

    # table stores carry no record, and the index lists them without a space
    for name in ("connections", "stimulus_coupling"):
        attributes = _attrs(os.path.join(session, "ground_truth", f"{name}.zarr"))
        assert "annotations" not in attributes
    index = _doc(os.path.join(session, "traces.zarr", "rsf", "stores.json"))
    assert {
        "path": "ground_truth/connections.zarr",
        "space": None,
        "written_by": "livn",
    } in index
    assert {
        "path": "ground_truth/stimulus_coupling.zarr",
        "space": None,
        "written_by": "livn",
    } in index


def test_the_coupling_is_absent_under_a_source_position_mask(tmp_path):
    with pytest.warns(UserWarning, match="source positions"):
        session = store(
            _run(),
            str(tmp_path / "masked"),
            _CultureEnv(io=_light_io(470.0)),
            programs=_two_light_programs(),
            connections=True,
            stimulus_coupling=True,
            withhold=[{"mask": {"fields": ["source_positions"]}}],
        )
    recording = rsf.RsfRecording(session)
    assert recording.stimulus_coupling() is None
    assert recording.connections() is not None  # it reveals no position
    assert not os.path.exists(
        os.path.join(session, "ground_truth", "stimulus_coupling.zarr")
    )
    assert recording.sources()["coordinates_um"] is None


def test_ground_truth_tables_given_by_the_caller_are_checked(tmp_path):
    def attempt(name, **kwargs):
        store(_run(), str(tmp_path / name), _Env(io=_light_io(470.0)), **kwargs)

    table = {
        "pre": ["0", "poisson"],
        "post": ["1", "2"],
        "weight": [0.1, 0.2],
        "units": "nS",
        "sign": [1, 1],
        "inputs": ["poisson"],
    }
    attempt("given", connections=table)
    assert rsf.RsfRecording(str(tmp_path / "given")).connections()["inputs"] == [
        "poisson"
    ]
    with pytest.raises(RSFError, match="neither sources nor named in inputs"):
        attempt("unnamed", connections={**table, "inputs": []})
    with pytest.raises(RSFError, match="not sources"):
        attempt("post", connections={**table, "post": ["1", "9"]})
    with pytest.raises(RSFError, match="says of what"):
        attempt("fraction", expression={"ChR2": {"units": "1", "values": [1, 1, 0]}})
    attempt(
        "expressed",
        expression={
            "ChR2": {"units": "1", "values": {0: 1.0}, "reference": "the maximum"}
        },
    )
    expressed = rsf.RsfRecording(str(tmp_path / "expressed")).sources()["expression"]
    np.testing.assert_allclose(expressed["ChR2"]["values"], [1.0, 0.0, 0.0])
    assert expressed["ChR2"]["reference"] == "the maximum"


def test_without_the_new_keywords_a_session_is_written_as_before(tmp_path):
    inputs = np.zeros((N_FRAMES, 2), dtype=np.float32)
    inputs[5:20, 0] = 1.0
    session = store(
        _run(),
        str(tmp_path / "plain"),
        _Env(io=_light_io(470.0)),
        inputs=inputs,
        inputs_dt=DT,
    )
    rsf_dir = os.path.join(session, "traces.zarr", "rsf")
    (record,) = rsf.RsfRecording(session).stimulation_records()
    assert [c["id"] for c in record["channels"]] == ["light/3", "light/7"]
    assert not {"name", "source"} & set(record["channels"][0])
    assert "origin" not in record
    assert not os.path.exists(os.path.join(rsf_dir, "stimulation", "0000", "planned"))
    assert not os.path.exists(os.path.join(rsf_dir, "notes"))
    assert os.listdir(os.path.join(rsf_dir, "ext")) == ["livn"]
    group = _attrs(rsf_dir)
    assert "zone" not in group
    simulation = _doc(os.path.join(rsf_dir, "simulation.json"))
    assert "prep" not in simulation
    assert set(simulation["sources"]) == {"ids", "coordinates_um"}
    assert sorted(os.listdir(os.path.join(session, "ground_truth"))) == ["spikes.zarr"]
    assert rsf.RsfRecording(session).epochs() == []


def test_a_calibration_states_the_latency_of_a_channel(physical, rsf_note):
    rows = [
        (100, "light/s1", 1.0, 0, 0, "marker"),
        (300, "light/s1", 0.0, 0, 1, "marker"),
    ]
    uid = "0123456789abcdef0123456789abcdef"
    version = {
        **rsf_note("light/s1", "2026-10-06T14:22:08.311904Z", "photodiode"),
        "full_scale": {"units": "mW/mm2", "value": 12.4},
        "measured_at": "2026-10-06T14:05:00Z",
        "method": "photodiode at the sample plane",
        "latency_ms": 2.0,
    }
    notes = {"stimulus_calibration": [(f"20261006T142208.311904Z-{uid}.json", version)]}
    recording = physical(
        "latency", stimulation=[_light_record()], stim_events=rows, notes=notes
    )
    calibrations = recording.stimulus_calibrations()
    assert calibrations["light/s1"]["latency_ms"] == 2.0
    assert calibrations["light/s1"]["full_scale"] == {"units": "mW/mm2", "value": 12.4}
    assert "light/s2" not in calibrations
    # 2 ms at 10 kHz: the light reaches the sample 20 frames after the delivered edge
    delivered = recording.inputs(0, 0, 400)[:, 0]
    at_sample = recording.inputs(0, 0, 400, latency=True)[:, 0]
    np.testing.assert_array_equal(np.flatnonzero(delivered), np.arange(100, 300))
    np.testing.assert_array_equal(np.flatnonzero(at_sample), np.arange(120, 320))
