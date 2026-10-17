"""livn.data: the Recording protocol, dispatch, the Arrow backend."""

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pytest

from livn.data import Missing, Recording
from livn.data import open as open_recording
from livn.io import MEA
from livn.run import Run, Series

if TYPE_CHECKING:
    pass

pytest.importorskip("datasets")
pytest.importorskip("pyarrow")

from livn.data import arrow

FS = 1000.0
C, T = 4, 100
POSITIONS = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0]])


def _signal(row):
    rng = np.random.default_rng(row)
    return rng.normal(size=(C, T)).astype(np.float32)


@pytest.fixture
def release(tmp_path):
    """A dataset laid out like the HF release, written the way `datasets` writes one."""
    from datasets import Dataset, Features, List, Value

    root = tmp_path / "release" / "c1"
    features = Features(
        {
            "sample_id": Value("string"),
            "cp": List(List(Value("float32"))),
            "cit": List(List(Value("int32"))),
            "ctt": List(List(Value("float64"))),
        }
    )
    empty = [[] for _ in range(C)]
    rows = {"sample_id": ["0", "1"], "cp": [], "cit": [], "ctt": []}
    for row in range(2):
        rows["cp"].append(_signal(row).tolist())
        cit, ctt = [list(e) for e in empty], [list(e) for e in empty]
        if row == 0:
            # unit 7 is seen on channel 1 three times and on channel 2 for two of them
            cit[1], ctt[1] = [7, 7, 7], [10.0, 40.5, 90.0]
            cit[2], ctt[2] = [7, 7], [10.0, 90.0]
        rows["cit"].append(cit)
        rows["ctt"].append(ctt)
    dataset = Dataset.from_dict(rows, features=features)
    dataset.save_to_disk(str(root / "train"))
    dataset.select([0]).save_to_disk(str(root / "test"))
    (root / "dataset_dict.json").write_text(json.dumps({"splits": ["train", "test"]}))
    (root / "conversion_metadata.json").write_text(
        json.dumps(
            {
                "sampling_frequency": FS,
                "num_channels": C,
                "bad_channel_ids": [3],
                "unit_peak_channel": {"7": 1},
            }
        )
    )
    np.savez(
        root / "templates.npz",
        templates=np.zeros((1, 9, C), np.float32),
        sampling_frequency=np.array(FS),
        probe_positions=POSITIONS,
    )
    (root / "dataset_stats.json").write_text(
        json.dumps({"global": {"scale": 2.0}, "preprocess_fingerprint": "ds=3;fs=1000"})
    )
    return root


def test_a_recording_satisfies_the_protocol(release, run_recording):
    recording = arrow.ArrowRecording(release)
    assert isinstance(recording, Recording)
    assert isinstance(run_recording(Run(duration=1.0)), Recording)


def test_rows_are_segments_read_by_window(release):
    recording = arrow.ArrowRecording(release)
    assert recording.segments == 2 and recording.fs == FS
    assert recording.num_samples(1) == T
    assert recording.sample_id(1) == "1"

    window = recording.read(1, 20, 30)
    assert window.dtype == np.float32 and window.shape == (30, C)
    np.testing.assert_array_equal(window, _signal(1)[:, 20:50].T)
    np.testing.assert_array_equal(
        recording.read(0, 0, T, channels=[2, 0]), _signal(0)[[2, 0]].T
    )
    assert recording.read(0, T, 0).shape == (0, C)
    with pytest.raises(IndexError):
        recording.read(0, 90, 20)
    with pytest.raises(IndexError):
        recording.read(5, 0, 1)
    with pytest.raises(KeyError):
        recording.read(0, 0, 1, channels=[17])


def test_splits_are_selected(release):
    assert arrow.ArrowRecording(release, split="test").segments == 1
    with pytest.raises(arrow.ArrowError, match="no split 'val'"):
        arrow.ArrowRecording(release, split="val")
    # a single saved split opens directly, finding the side files above it
    single = arrow.ArrowRecording(release / "test", split=None)
    assert single.segments == 1 and single.fs == FS


def test_spikes_merge_the_channels_and_keep_the_peak(release):
    recording = arrow.ArrowRecording(release)
    events = recording.spikes(0)
    np.testing.assert_array_equal(events.ids, [7, 7, 7])
    np.testing.assert_allclose(events.times, [10.0, 40.5, 90.0])
    assert events.t0 == 0.0 and events.duration == pytest.approx(T / FS * 1000.0)
    assert recording.metadata["spike_peak_channel"][0] == {7: 1}
    assert recording.peak_channels() == {7: 1}  # the shipped map wins

    assert recording.spikes(1).ids.size == 0
    assert recording.ground_truth("spikes", 0).ids.size == 3


def test_what_the_release_ships_is_in_the_metadata(release):
    recording = arrow.ArrowRecording(release)
    np.testing.assert_array_equal(recording.bad_channels, [3])
    np.testing.assert_array_equal(recording.channel_ids, np.arange(C))
    assert recording.metadata["preprocess_fingerprint"] == "ds=3;fs=1000"
    assert recording.metadata["dataset_stats"]["global"]["scale"] == 2.0
    assert recording.metadata["templates"]["templates"].shape == (1, 9, C)
    assert recording.metadata["conversion"]["bad_channel_ids"] == [3]
    assert recording.ground_truth("templates")["probe_positions"].shape == (C, 2)

    assert isinstance(recording.io, MEA)
    np.testing.assert_allclose(recording.io.electrode_coordinates[:, 1:3], POSITIONS)
    assert recording.metadata["columns"] == ["sample_id", "cp", "cit", "ctt"]


def test_inputs_are_never_zeros(release):
    recording = arrow.ArrowRecording(release)
    with pytest.raises(Missing, match="no `in` column"):
        recording.inputs(0, 0, 10)
    with pytest.raises(Missing):
        recording.ground_truth("voltage")
    with pytest.raises(Missing):
        recording.ground_truth("patch")


def test_a_row_is_a_run(release):
    run = arrow.ArrowRecording(release).to_run(0, 20, 60)
    assert run.t0 == pytest.approx(20.0) and run.duration == pytest.approx(60.0)
    np.testing.assert_array_equal(run.potential.values, _signal(0)[:, 20:80])
    # spikes inside [20, 80) ms, relative to the window
    np.testing.assert_allclose(run.spike_times, [20.5])
    np.testing.assert_array_equal(run.potential.ids, np.arange(C))


def test_reads_are_thread_safe_and_survive_pickling(release):
    recording = arrow.ArrowRecording(release)
    again = pickle.loads(pickle.dumps(recording))
    starts = list(range(0, 80, 5)) * 4

    def read(start):
        return start, again.read(start % 2, start, 16)

    with ThreadPoolExecutor(8) as pool:
        for start, got in pool.map(read, starts):
            np.testing.assert_array_equal(
                got, _signal(start % 2)[:, start : start + 16].T
            )


# -- dispatch -------------------------------------------------------------------------


def test_open_reads_the_layout_off_a_local_path(release, tmp_path):
    recording = open_recording(release)
    assert isinstance(recording, arrow.ArrowRecording)
    assert open_recording(release, split="test").segments == 1
    assert open_recording(release, backend="arrow").segments == 2

    with pytest.raises(ValueError, match="neither an rsf session"):
        open_recording(tmp_path)
    with pytest.raises(ValueError, match="unknown backend"):
        open_recording(release, backend="hdf5")


def test_open_sends_a_session_to_rsf(tmp_path):
    pytest.importorskip("zarr", minversion="3")
    import test_data_rsf

    from livn.data import rsf

    session = str(tmp_path / "session")
    rsf.store(test_data_rsf._run(), session, test_data_rsf._Env())
    recording = open_recording(session)
    assert isinstance(recording, rsf.RsfRecording)
    assert isinstance(recording, Recording)


def test_hub_uris_download_only_their_config(release, monkeypatch):
    import huggingface_hub

    calls = []

    def fake(repo_id, **kw):
        calls.append((repo_id, kw))
        return str(release.parent)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    recording = open_recording("hf://frthjf/spe1-zhao2026-benchmark/c1")
    assert isinstance(recording, arrow.ArrowRecording)
    assert recording.segments == 2
    ((repo, kw),) = calls
    assert repo == "frthjf/spe1-zhao2026-benchmark"
    assert kw["repo_type"] == "dataset" and kw["allow_patterns"] == ["c1/**"]

    with pytest.raises(arrow.ArrowError, match="hf://<owner>/<repo>"):
        arrow.snapshot("hf://onlyowner")
    with pytest.raises(ValueError, match="hub dataset"):
        open_recording("hf://a/b/c", backend="rsf")


def test_a_run_in_memory_is_a_recording(run_recording):
    values = np.arange(C * 50, dtype=np.float32).reshape(C, 50)
    run = Run(
        duration=50.0,
        potential=Series(ids=np.arange(C), values=values, dt=1.0),
    ).add_spikes(np.array([3, 3]), np.array([5.0, 30.0]))
    recording = run_recording(run, io=None)

    assert recording.fs == 1000.0 and recording.segments == 1
    assert recording.num_samples(0) == 50
    np.testing.assert_array_equal(recording.read(0, 10, 5), values[:, 10:15].T)
    np.testing.assert_array_equal(
        recording.read(0, 0, 3, channels=[3, 1]), values[[3, 1], :3].T
    )
    np.testing.assert_array_equal(recording.spikes(0).ids, [3, 3])
    assert recording.to_run() is run
    part = recording.to_run(0, 15, 20)
    assert part.duration == 20.0 and part.potential.values.shape == (C, 20)
    np.testing.assert_allclose(part.spike_times, [15.0])
    with pytest.raises(Missing, match="no input series"):
        recording.inputs(0, 0, 5)
    with pytest.raises(Missing):
        recording.ground_truth("voltage")
    with pytest.raises(IndexError):
        recording.read(1, 0, 1)
    with pytest.raises(Missing):
        run_recording(Run(duration=1.0)).read(0, 0, 1)


# -- writing --------------------------------------------------------------------------


class _Env:
    comm = None

    def __init__(self):
        self.io = MEA(np.column_stack([np.arange(C), POSITIONS, np.zeros(C)]))

    def channel_recording(self, ids, times):
        ids, times = np.asarray(ids), np.asarray(times)
        near = {0: [0, 1], 1: [2, 3]}  # unit -> channels that see it
        cit = {
            c: ids[[u in near and c in near[u] for u in ids.tolist()]] for c in range(C)
        }
        ctt = {
            c: times[[u in near and c in near[u] for u in ids.tolist()]]
            for c in range(C)
        }
        return cit, ctt


def _run(n=50, with_cells=False):
    run = Run(
        duration=n * 1.0,
        potential=Series(ids=np.arange(C), values=_signal(0)[:, :n], dt=1.0),
    ).add_spikes(np.array([0, 1, 0]), np.array([3.0, 12.0, 40.0]))
    if with_cells:
        run = run.add_voltage(
            np.array([0, 1]),
            np.ones((2, n), np.float32),
            dt=1.0,
            sections=np.array(["soma", "soma"]),
        ).add_current(
            np.array([0, 0, 1]),
            np.ones((3, n), np.float32),
            dt=0.5,
            sections=np.array(["soma", "dend", "soma"]),
        )
    return run


def test_store_then_open_round_trips(tmp_path):
    root = str(tmp_path / "ds")
    env = _Env()
    inputs = np.arange(2 * C, dtype=np.float32).reshape(2, C)
    assert arrow.store(_run(), root, env=env, inputs=inputs, inputs_dt=25.0) == root
    arrow.store(
        _run(30),
        root,
        env=env,
        sample_id="late",
        inputs=inputs,
        inputs_dt=25.0,
        split="test",
    )

    recording = open_recording(root)
    assert recording.segments == 1 and recording.sample_id(0) == "0000000000"
    np.testing.assert_array_equal(recording.read(0, 0, 50), _signal(0).T[:50])
    np.testing.assert_array_equal(recording.channel_ids, np.arange(C))
    assert recording.fs == 1000.0

    # the spike is on every channel that sees its unit, and is one spike back
    events = recording.spikes(0)
    np.testing.assert_array_equal(events.ids, [0, 1, 0])
    np.testing.assert_allclose(events.times, [3.0, 12.0, 40.0])
    assert recording.metadata["spike_peak_channel"][0] == {0: 0, 1: 2}

    delivered = recording.inputs(0, 0, 50)
    np.testing.assert_array_equal(delivered[:25], np.tile(inputs[0], (25, 1)))
    np.testing.assert_array_equal(delivered[25:], np.tile(inputs[1], (25, 1)))

    assert isinstance(recording.io, MEA)
    np.testing.assert_allclose(
        recording.io.electrode_coordinates, env.io.electrode_coordinates
    )
    test = open_recording(root, split="test")
    assert test.sample_id(0) == "late" and test.num_samples(0) == 30
    with open(os.path.join(root, "dataset_dict.json")) as f:
        splits = json.load(f)["splits"]
    assert splits == [
        "train",
        "test",
    ]


def test_store_appends_shards_and_keeps_the_schema(tmp_path):
    root = str(tmp_path / "ds")
    env = _Env()
    for _ in range(3):
        arrow.store(_run(), root, env=env)
    recording = open_recording(root)
    assert recording.segments == 3
    assert [recording.sample_id(i) for i in range(3)] == [f"{i:010d}" for i in range(3)]
    shards = sorted(
        f for f in os.listdir(os.path.join(root, "train")) if f.endswith(".arrow")
    )
    assert shards == ["data-00000.arrow", "data-00001.arrow", "data-00002.arrow"]

    # a new store object picks up where the last left off
    arrow.store(_run(), root, env=env)
    assert open_recording(root).segments == 4

    with pytest.raises(arrow.ArrowError, match="same ones"):
        arrow.store(_run(), root, env=env, inputs=np.zeros((1, C)), inputs_dt=50.0)
    with pytest.raises(arrow.ArrowError, match="Hz"):
        slow = Run(
            duration=50.0,
            potential=Series(ids=np.arange(C), values=_signal(0)[:, :50], dt=2.0),
        ).add_spikes(np.array([0]), np.array([1.0]))
        arrow.store(slow, root, env=env)
    assert open_recording(root).segments == 4


def test_cell_space_ground_truth_shares_the_schema(tmp_path):
    root = str(tmp_path / "ds")
    run = _run(with_cells=True)
    arrow.store(run, root, env=_Env(), ground_truth=("spikes", "voltage", "current"))
    recording = open_recording(root)

    voltage = recording.ground_truth("voltage")
    np.testing.assert_array_equal(voltage.ids, [0, 1])
    assert voltage.dt == 1.0 and voltage.values.shape == (2, 50)
    current = recording.ground_truth("currents")
    assert current.dt == 0.5
    np.testing.assert_array_equal(current.ids, [0, 0, 1])
    np.testing.assert_array_equal(current.sections, ["soma", "dend", "soma"])

    # the cell-space spikes are the ground truth and are used as they are
    np.testing.assert_array_equal(recording.spikes(0).ids, [0, 1, 0])
    back = recording.to_run(0, 0, 50)
    assert back.voltage_ids.tolist() == [0, 1] and back.current_dt == 0.5


def test_cell_space_only_needs_no_channels(tmp_path):
    root = str(tmp_path / "ds")
    run = Run(duration=100.0).add_spikes(np.array([2, 5]), np.array([10.0, 20.0]))
    arrow.store(run, root, ground_truth=("spikes",))
    recording = open_recording(root)
    np.testing.assert_array_equal(recording.spikes(0).ids, [2, 5])
    assert np.isnan(recording.fs)
    with pytest.raises(Missing, match="no `cp`"):
        recording.read(0, 0, 1)
    with pytest.raises(arrow.ArrowError, match="nothing to write"):
        arrow.store(run, str(tmp_path / "empty"))


def test_store_names_what_it_cannot_assign(tmp_path):
    with pytest.raises(arrow.ArrowError, match="channel_spikes"):
        arrow.store(_run(), str(tmp_path / "a"))
    with pytest.raises(arrow.ArrowError, match="ground_truth names"):
        arrow.store(_run(), str(tmp_path / "b"), env=_Env(), ground_truth=("lfp",))
    assert arrow.store(None, str(tmp_path / "c")) is None
    assert not os.path.exists(tmp_path / "c")

    # by-hand per-channel spikes stand in for the env
    cit = {1: np.array([4]), 3: np.array([4])}
    ctt = {1: np.array([8.0]), 3: np.array([8.0])}
    arrow.store(_run(), str(tmp_path / "d"), channel_spikes=(cit, ctt))
    events = open_recording(str(tmp_path / "d")).spikes(0)
    np.testing.assert_array_equal(events.ids, [4])
    np.testing.assert_allclose(events.times, [8.0])


def test_side_files_are_written_once(tmp_path):
    root = str(tmp_path / "ds")
    arrow.store(
        _run(),
        root,
        env=_Env(),
        bad_channels=[2],
        metadata={"system": "toy"},
        templates={"templates": np.zeros((1, 3, C)), "probe_positions": POSITIONS},
        dataset_stats={"preprocess_fingerprint": "abc"},
    )
    arrow.store(_run(), root, env=_Env(), bad_channels=[0])
    recording = open_recording(root)
    np.testing.assert_array_equal(recording.bad_channels, [2])
    assert recording.metadata["conversion"]["system"] == "toy"
    assert recording.metadata["preprocess_fingerprint"] == "abc"
    assert recording.metadata["templates"]["templates"].shape == (1, 3, C)
