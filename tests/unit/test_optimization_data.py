import json

import numpy as np
import pytest

pytest.importorskip("grain")
pytest.importorskip("scipy")

from livn.data import Missing
from livn.optimization.data import CMR, Bandpass, Standardize, Window, windows
from livn.run import Events, Run, Series

FS = 10_000.0  # Hz; dt = 0.1 ms
C, T = 3, 20_000  # two seconds


@pytest.fixture
def make_recording(run_recording):
    def make(seed=0, spikes=True, signal=None):
        rng = np.random.default_rng(seed)
        values = (
            rng.normal(size=(C, T)).astype(np.float32) if signal is None else signal
        )
        ev = None
        if spikes:
            times = np.sort(rng.uniform(0, T / FS * 1e3, 300))
            ev = Events(ids=rng.integers(5, 9, 300), times=times, duration=T / FS * 1e3)
        run = Run(
            channels={"spikes": ev} if ev is not None else {},
            potential=Series(ids=np.arange(10, 10 + C), values=values, dt=1e3 / FS),
            duration=T / FS * 1e3,
        )
        return run_recording(run)

    return make


def _same(a, b):
    a, b = a.as_dict(), b.as_dict()
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.fixture
def make_pipe(make_recording):
    def make(rec=None, **kw):
        kw = {"context": 5.0, "batch_size": 12, "per_segment": 6, "threads": 2} | kw
        return windows(rec if rec is not None else make_recording(), **kw)

    return make


def _take(it, n):
    return [next(it) for _ in range(n)]


def test_the_same_seed_gives_the_same_windows_and_another_seed_does_not(make_pipe):
    a, b, c = make_pipe(seed=1), make_pipe(seed=1), make_pipe(seed=2)
    _same(a.batch(3), b.batch(3))
    assert not np.array_equal(a.batch(3).start, c.batch(3).start)


def test_every_epoch_redraws_the_starts(make_pipe):
    pipe = make_pipe(per_segment=12)
    assert pipe.epoch_steps == 1
    first, second = pipe.batch(0), pipe.batch(1)
    assert not np.array_equal(first.start, second.start)
    assert not np.array_equal(np.sort(first.start), np.sort(second.start))


def test_batches_are_fixed_shape_float32(make_pipe):
    pipe = make_pipe(context=5.0, gap=1.0, prediction=2.0, fs=5000.0, spikes=True)
    batch = pipe.batch(0)
    assert batch.context.shape == (12, 25, C) and batch.context.dtype == np.float32
    assert batch.prediction.shape == (12, 10, C)
    assert batch.spikes.shape == (12, 4, 64)
    assert batch.channel_mask.shape == (12, C)
    assert batch.start.shape == (12,) and batch.start.dtype == np.int64
    assert (batch.start % 2 == 0).all(), "starts sit on the decimation grid"


@pytest.mark.parametrize("processes", [1, 2, 3, 4])
def test_process_slices_concatenate_to_the_global_batch(processes, make_pipe):
    pipe = make_pipe(batch_size=12, spikes=True)
    its = [pipe.iterate(0, p, processes) for p in range(processes)]
    for step in range(4):
        parts = [next(it) for it in its]
        assert parts[0].context.shape[0] == 12 // processes
        whole = Window(
            **{
                k: np.concatenate([getattr(p, k) for p in parts])
                for k in parts[0].as_dict()
            }
        )
        _same(whole, pipe.batch(step))


@pytest.mark.parametrize("processes", [1, 2, 3])
def test_resuming_at_a_step_replays_the_same_batches(processes, make_pipe):
    pipe = make_pipe(batch_size=12)
    full = _take(pipe.iterate(0, 0, processes), 5)
    resumed = _take(pipe.iterate(3, 0, processes), 2)
    _same(resumed[0], full[3])
    _same(resumed[1], full[4])


def test_a_batch_that_does_not_divide_is_refused(make_pipe):
    with pytest.raises(ValueError, match="not divisible"):
        make_pipe(batch_size=10).iterate(0, 0, 4)


def test_the_process_defaults_come_from_jax(make_pipe):
    pipe = make_pipe()
    _same(next(pipe.iterate()), pipe.batch(0))


def test_worker_processes_give_the_same_batches(make_pipe):
    pipe, workers = make_pipe(), make_pipe(processes=2)
    _same(next(workers.iterate(1, 0, 1)), pipe.batch(1))


def test_windows_are_the_recording_samples_at_their_start(make_pipe, make_recording):
    rec = make_recording()
    batch = make_pipe(rec, mask_bad_channels=False).batch(0)
    values = np.asarray(rec.run.potential.values)
    for i in range(12):
        s = int(batch.start[i])
        np.testing.assert_array_equal(batch.context[i], values[:, s : s + 50].T)


def test_grid_head_and_tail_starts_are_deterministic(make_pipe, make_recording):
    rec = make_recording()
    grid = make_pipe(rec, starts="grid", per_segment=5, batch_size=5).batch(0).start
    assert grid[0] == 0 and grid[-1] == T - 50 and (np.diff(grid) > 0).all()
    head = make_pipe(rec, starts="head", batch_size=2).batch(0).start
    tail = make_pipe(rec, starts="tail", batch_size=2).batch(0).start
    assert (head == 0).all() and (tail == T - 50).all()


def test_spikes_are_window_relative_ms_and_inf_padded_per_unit(
    make_pipe, make_recording
):
    rec = make_recording()
    pipe = make_pipe(rec, spikes=True, max_spikes=8)
    np.testing.assert_array_equal(pipe.unit_ids, [5, 6, 7, 8])
    batch = pipe.batch(0)
    events = rec.run.spikes
    for i in range(12):
        t0 = batch.start[i] / FS * 1e3
        row = batch.spikes[i]
        assert row.shape == (4, 8)
        for u, unit in enumerate(pipe.unit_ids):
            mine = events.times[events.ids == unit]
            mine = mine[(mine >= t0) & (mine < t0 + 5.0)] - t0
            got = row[u][np.isfinite(row[u])]
            np.testing.assert_allclose(got, mine[:8], atol=1e-4)
            assert np.isinf(row[u][len(got) :]).all()
            assert (got >= 0).all() and (got < 5.0).all()


def test_spikes_beyond_capacity_keep_the_earliest(make_pipe):
    pipe = make_pipe(context=100.0, spikes=True, max_spikes=2, batch_size=2)
    row = pipe.batch(0).spikes[0]
    assert (np.diff(row[np.isfinite(row).all(axis=1)], axis=1) >= 0).all()


def test_inputs_raise_when_the_recording_has_none(make_pipe):
    with pytest.raises(Missing):
        make_pipe(inputs=True)


def test_inputs_are_strided_to_the_target_rate(make_pipe, make_recording):
    rec = make_recording()
    frames = np.arange(T, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32)
    rec.inputs = lambda segment, start, length: frames[start : start + length]
    batch = make_pipe(rec, inputs=True, fs=5000.0).batch(0)
    assert batch.inputs.shape == (12, 25, 2)
    np.testing.assert_array_equal(
        batch.inputs[0, :, 0], batch.start[0] + 2 * np.arange(25)
    )


def test_bad_channels_are_zeroed_and_masked(make_pipe, make_recording):
    rec = make_recording()
    rec.bad_channels = np.array([11])
    batch = make_pipe(rec).batch(0)
    assert not batch.channel_mask[0, 1] and batch.channel_mask[0, [0, 2]].all()
    assert (batch.context[..., 1] == 0).all()
    assert (make_pipe(rec, mask_bad_channels=False).batch(0).context[..., 1] != 0).any()


def test_a_window_policy_can_move_the_start(make_pipe):
    seen = []

    def policy(item, rng, recording):
        seen.append(item["length"])
        return {**item, "start": 100}

    batch = make_pipe(window_policy=policy).batch(0)
    assert (batch.start == 100).all() and set(seen) == {50}

    with pytest.raises(ValueError, match="window_policy returned start"):
        make_pipe(window_policy=lambda item, rng, rec: {**item, "start": T}).batch(0)


def test_splits_select_recordings_and_segments(make_pipe, make_recording):
    train, test = make_recording(0), make_recording(1)
    train.metadata["split"], test.metadata["split"] = "train", "test"
    pipe = make_pipe([train, test], split="test", per_segment=2, batch_size=2)
    assert (pipe.batch(0).recording == 1).all()
    assert (
        len(
            make_pipe([train, test], split=None, per_segment=2, batch_size=2)
            .batch(0)
            .recording
        )
        == 2
    )
    with pytest.raises(ValueError, match="no segments"):
        make_pipe([train], split="val")


def test_windows_stay_inside_the_valid_spans(make_pipe, make_recording):
    rec = make_recording()
    spans = np.asarray([[0, 5000], [5000, 12000]])
    rec.valid_spans = lambda segment: spans
    length = 50
    for starts in ("random", "grid"):
        batch = make_pipe(rec, starts=starts, per_segment=24, batch_size=24).batch(0)
        inside = (batch.start[:, None] >= spans[:, :1].T) & (
            batch.start[:, None] + length <= spans[:, 1:].T
        )
        assert inside.any(axis=1).all(), starts
    assert (
        make_pipe(rec, starts="tail", per_segment=1, batch_size=1).batch(0).start[0]
        == 12000 - length
    )
    with pytest.raises(ValueError, match="valid spans"):
        make_pipe(
            rec, window_policy=lambda item, rng, r: {**item, "start": 4980}
        ).batch(0)
    rec.valid_spans = lambda segment: np.asarray([[0, 30], [30, 60]])
    with pytest.raises(ValueError, match="holds one window"):
        make_pipe(rec)


def test_a_window_longer_than_the_segment_is_refused(make_pipe):
    with pytest.raises(ValueError, match="holds one window"):
        make_pipe(context=5000.0)


def test_decimation_keeps_a_sine_and_is_zero_phase_to_the_edges(
    make_pipe, make_recording
):
    f = 200.0
    t = np.arange(T) / FS
    wave = np.sin(2 * np.pi * f * t).astype(np.float32)
    rec = make_recording(signal=np.stack([wave] * C))
    pipe = make_pipe(
        rec, context=20.0, fs=2500.0, starts="grid", per_segment=4, batch_size=4
    )
    batch = pipe.batch(0)
    for i in range(4):  # includes the first and last window, which touch the edges
        s = int(batch.start[i])
        expected = np.sin(2 * np.pi * f * (s + 4 * np.arange(50)) / FS)
        np.testing.assert_allclose(
            batch.context[i, :, 0], expected, atol=6e-2
        )  # the edge windows see a flat extension


def test_a_fractional_decimation_is_refused(make_pipe):
    with pytest.raises(ValueError, match="not a divisor"):
        make_pipe(fs=3000.0)


def test_bandpass_removes_the_offset_and_cmr_the_common_signal(
    make_pipe, make_recording
):
    rng = np.random.default_rng(3)
    common = rng.normal(size=T).astype(np.float32) * 5
    values = rng.normal(size=(C, T)).astype(np.float32) * 0.1 + common + 40.0
    rec = make_recording(signal=values)
    raw = make_pipe(rec).batch(0).context
    out = make_pipe(rec, preprocess=(Bandpass(300.0, 3000.0), CMR())).batch(0).context
    assert abs(raw.mean()) > 30 and abs(out.mean()) < 1
    assert out.std() < 0.5 * raw.std()


def test_preprocessing_at_the_target_rate_filters_the_decimated_window(
    make_pipe, make_recording
):
    from scipy import signal as sp

    rng = np.random.default_rng(5)
    values = rng.normal(size=(C, T)).astype(np.float32)
    rec = make_recording(signal=values)
    pre = (Bandpass(300.0, 3000.0), CMR())
    kw = {
        "fs": 5000.0,
        "context": 10.0,
        "starts": "grid",
        "per_segment": 3,
        "batch_size": 3,
    }
    batch = make_pipe(rec, preprocess=pre, preprocess_at="target", **kw).batch(0)

    for i in range(3):
        s = int(batch.start[i])
        # decimate the padded read, crop, then filter that window alone: its edges stay in it
        lo, hi = s - 22, s + 100 + 22
        raw = values[:, max(lo, 0) : min(hi, T)].T.astype(np.float64)
        raw = np.pad(raw, ((max(-lo, 0), max(hi - T, 0)), (0, 0)), mode="edge")
        x = sp.resample_poly(raw, 1, 2, axis=0)[11 : 11 + 50]
        x = Bandpass(300.0, 3000.0)(x, 5000.0, np.ones(C, bool))
        x = CMR()(x, 5000.0, np.ones(C, bool))
        np.testing.assert_allclose(batch.context[i], x, atol=1e-3)
    assert (
        make_pipe(rec, preprocess=pre, preprocess_at="target", **kw).stats.fingerprint
        == "decim=2;fs=5000;pre=bp(300,3000,3)+cmr;at=target"
    )
    with pytest.raises(ValueError, match="preprocess_at"):
        make_pipe(preprocess_at="later")


def _stats_pipe(make_pipe, rec, **kw):
    return make_pipe(rec, normalize=Standardize("fit", n_windows=8, **kw), seed=4)


def test_fitted_stats_are_robust_deterministic_and_applied(make_pipe, make_recording):
    values = np.random.default_rng(0).normal(3.0, 2.0, size=(C, T)).astype(np.float32)
    rec = make_recording(signal=values)
    a, b = (
        make_pipe(rec, normalize=Standardize("fit", n_windows=60), seed=4) for _ in "ab"
    )
    assert a.stats.source == "fit"
    np.testing.assert_array_equal(a.stats.median, b.stats.median)
    np.testing.assert_allclose(a.stats.median, 3.0, atol=0.3)
    np.testing.assert_allclose(a.stats.scale, 2.326 * 2, rtol=0.1)  # (p99-p1)/2
    assert abs(a.batch(0).context.mean()) < 0.2
    assert a.stats.fingerprint == "decim=1;fs=10000"


def test_the_fingerprint_names_decimation_and_preprocessing(make_pipe):
    pipe = make_pipe(fs=2500.0, preprocess=(Bandpass(300, 3000, 2), CMR()))
    assert pipe.stats.fingerprint == "decim=4;fs=2500;pre=bp(300,3000,2)+cmr"
    assert pipe.stats.median is None


def test_stats_are_cached_under_the_given_directory_and_reused(
    tmp_path, make_pipe, make_recording
):
    rec = make_recording()
    a = _stats_pipe(make_pipe, rec, cache=tmp_path / "run")
    files = list((tmp_path / "run").glob("stats-*.json"))
    assert len(files) == 1 and a.stats.source == "fit"
    document = json.loads(files[0].read_text())
    assert document["preprocess_fingerprint"] == a.stats.fingerprint
    b = _stats_pipe(make_pipe, rec, cache=tmp_path / "run")
    assert b.stats.source == "cache"
    np.testing.assert_array_equal(a.stats.scale, b.stats.scale)
    # other preprocessing is another cache entry
    make_pipe(
        rec, normalize=Standardize("fit", tmp_path / "run", 8), preprocess=(CMR(),)
    )
    assert len(list((tmp_path / "run").glob("stats-*.json"))) == 2


def test_the_cache_is_never_inside_the_dataset(tmp_path, make_pipe, make_recording):
    rec = make_recording()
    rec.path = tmp_path / "dataset"
    with pytest.raises(ValueError, match="inside the dataset"):
        _stats_pipe(make_pipe, rec, cache=tmp_path / "dataset" / "stats")


def test_dataset_stats_metadata_is_not_used_implicitly(make_pipe, make_recording):
    rec = make_recording()
    rec.metadata["dataset_stats"] = {
        "preprocess_fingerprint": "decim=1;fs=10000",
        "median": 1.0,
        "scale": 4.0,
    }
    pipe = make_pipe(rec, normalize=Standardize(n_windows=4))
    assert pipe.stats.source == "fit"
    assert pipe.stats.fingerprint == "decim=1;fs=10000"
    other = make_pipe(rec, normalize=Standardize(n_windows=4), preprocess=(CMR(),))
    assert other.stats.fingerprint == "decim=1;fs=10000;pre=cmr"
    # reusing a dataset's stats is the caller's call
    given = make_pipe(rec, normalize=Standardize(rec.metadata["dataset_stats"]))
    assert given.stats.source == "given"
    np.testing.assert_array_equal(given.stats.scale, [4.0] * C)


def test_given_stats_come_from_a_dict_or_a_file(tmp_path, make_pipe, make_recording):
    rec = make_recording()
    given = make_pipe(rec, normalize=Standardize({"median": [0, 1, 2], "scale": 2.0}))
    assert given.stats.source == "given"
    np.testing.assert_array_equal(given.stats.median, [0, 1, 2])
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"median": 0.5, "scale": 3.0}))
    np.testing.assert_array_equal(
        make_pipe(rec, normalize=Standardize(path)).stats.scale, [3.0] * C
    )


def test_inputs_of_rsf_sessions_come_from_their_stimulus_events(
    tmp_path, physical_session
):
    """A session's delivered stimulus reaches the windows through the RSF reader, the rule
    simulated and recorded sessions share: each channel's fraction of its full scale, held
    from event to event."""
    pytest.importorskip("zarr", minversion="3")
    from livn.data.rsf import RsfRecording

    record = {
        "started_at": "2026-10-04T09:12:34Z",
        "kind": "electrode",
        "channels": [
            {
                "id": "electrode/4",
                "shape": {"kind": "contact", "contact_id": 4, "x": 0.0, "y": 0.0},
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
    rows = [
        (0, "electrode/4", 0.2, 0, 0, "schedule"),
        (8_000, "electrode/4", -0.2, 0, 1, "schedule"),
        (12_000, "electrode/4", 0.0, 0, 2, "schedule"),
    ]
    path = physical_session(
        str(tmp_path / "recorded"),
        n_frames=T,
        stimulation=[record],
        stim_events=rows,
    )
    pipe = windows(
        RsfRecording(path),
        context=5.0,
        batch_size=12,
        per_segment=6,
        threads=2,
        inputs=True,
        fs=5000.0,
        mask_bad_channels=False,
    )
    batch = pipe.batch(0)
    assert batch.inputs.shape == (12, 25, 1)
    for start, got in zip(batch.start, batch.inputs[:, :, 0], strict=True):
        frames = start + 2 * np.arange(25)
        want = np.where(frames < 8_000, 0.2, np.where(frames < 12_000, -0.2, 0.0))
        np.testing.assert_allclose(got, want)
