import numpy as np
import pytest

from livn.backend import backend
from livn.decoding import Slice
from livn.run import Events, Run, Series

try:
    import mpi4py  # noqa: F401

    _has_mpi4py = True
except ImportError:
    _has_mpi4py = False


def _run(t0=0.0, duration=10.0, dt=0.5, n=3, seed=0):
    rng = np.random.default_rng(seed)
    t = int(duration / dt)
    spike_times = np.sort(rng.uniform(0, duration, size=7))
    spike_ids = rng.integers(0, n, size=7)
    ids = np.arange(n)
    return (
        Run(t0=t0, duration=duration)
        .add_spikes(spike_ids, spike_times)
        .add_voltage(ids, rng.normal(size=(n, t)), dt=dt)
        .add_current(ids, rng.normal(size=(n, t)), dt=dt)
    )


def test_named_access_matches_positions():
    run = _run()

    it, tt, iv, v, im, mp = run

    assert it is run.spike_ids
    assert tt is run.spike_times
    assert iv is run.voltage_ids
    assert v is run.voltage
    assert im is run.current_ids
    assert mp is run.current

    assert len(run) == 6
    assert run[0] is run.spike_ids
    assert run[5] is run.current
    assert list(run[:2]) == [run.spike_ids, run.spike_times]


def test_tuple_unpacking_forms():
    run = _run()

    it, t, iv, v, *rest = run
    assert len(rest) == 2

    it, tt, iv, vv, im, mp = run
    assert (it, tt, iv, vv, im, mp) == tuple(run)


def test_missing_channels_are_none():
    run = (
        Run(duration=10.0)
        .add_spikes(np.array([1]), np.array([0.5]))
        .add_voltage(None, None)
        .add_current(None, None)
    )

    assert run.voltage_ids is None
    assert run.voltage is None
    assert run.current is None
    assert "voltage" not in run
    assert tuple(run)[2:] == (None, None, None, None)


def test_the_standard_channels_are_named_both_ways():
    run = _run(duration=10.0, dt=0.5)

    assert set(run.channels) == {"spikes", "voltage", "current"}

    assert set(run.drop_spikes().channels) == {"voltage", "current"}
    assert set(run.drop_voltage().channels) == {"spikes", "current"}
    assert set(run.drop_current().channels) == {"spikes", "voltage"}

    stripped = run.drop_voltage().drop_current().drop_current()
    assert set(stripped.channels) == {"spikes"}
    assert set(run.channels) == {"spikes", "voltage", "current"}
    assert stripped.voltage is None


def test_add_is_immutable_and_infers_kind():
    run = Run(duration=10.0)
    with_spikes = run.add("spikes", np.array([1]), np.array([0.5]))
    with_voltage = with_spikes.add("voltage", np.array([1]), np.zeros((1, 20)), dt=0.5)

    assert "spikes" not in run
    assert "voltage" not in with_spikes

    assert isinstance(with_voltage["spikes"], Events)
    assert isinstance(with_voltage["voltage"], Series)
    assert with_voltage.voltage_dt == 0.5

    assert isinstance(run.add("x", np.array([1]), np.zeros((1, 4)))["x"], Series)


def test_concat_reapplies_the_time_offset():
    a = _run(t0=0.0, duration=10.0, seed=1)
    b = _run(t0=10.0, duration=6.0, seed=2)

    joined = a.concat(b)

    assert joined.t0 == 0.0
    assert joined.duration == 16.0

    np.testing.assert_allclose(
        joined.spike_times,
        np.concatenate([a.spike_times, b.spike_times + 10.0]),
    )
    np.testing.assert_array_equal(
        joined.spike_ids, np.concatenate([a.spike_ids, b.spike_ids])
    )
    assert joined.spike_times.max() < joined.duration

    np.testing.assert_allclose(
        joined.voltage, np.concatenate([a.voltage, b.voltage], axis=1)
    )
    assert joined.voltage.shape[1] == int(16.0 / 0.5)


def test_concat_without_tracked_t0_assumes_chunks_follow():
    a = _run(t0=0.0, duration=10.0, seed=1)
    b = _run(t0=0.0, duration=6.0, seed=2)

    joined = a.concat(b)

    assert joined.duration == 16.0
    np.testing.assert_allclose(
        joined.spike_times,
        np.concatenate([a.spike_times, b.spike_times + 10.0]),
    )


def test_concat_of_chunks_equals_a_single_run():
    dt = 0.5
    duration = 30.0
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(0, duration, size=40))
    ids = rng.integers(0, 4, size=40)
    voltage = rng.normal(size=(4, int(duration / dt)))

    single = (
        Run(t0=0.0, duration=duration)
        .add_spikes(ids, times)
        .add_voltage(np.arange(4), voltage, dt=dt)
    )

    chunked = None
    for start in (0.0, 10.0, 20.0):
        mask = (times >= start) & (times < start + 10.0)
        lo, hi = int(start / dt), int((start + 10.0) / dt)
        chunk = (
            Run(t0=start, duration=10.0)
            .add_spikes(ids[mask], times[mask] - start)
            .add_voltage(np.arange(4), voltage[:, lo:hi], dt=dt)
        )
        chunked = chunk if chunked is None else chunked.concat(chunk)

    np.testing.assert_allclose(chunked.spike_times, single.spike_times)
    np.testing.assert_array_equal(chunked.spike_ids, single.spike_ids)
    np.testing.assert_allclose(chunked.voltage, single.voltage)
    assert chunked.duration == single.duration


def test_merge_over_disjoint_gids():
    dt = 0.5
    a = (
        Run(duration=10.0)
        .add_spikes(np.array([0, 1]), np.array([1.0, 2.0]))
        .add_voltage(np.array([0, 1]), np.ones((2, 20)), dt=dt)
    )
    b = (
        Run(duration=10.0)
        .add_spikes(np.array([2]), np.array([3.0]))
        .add_voltage(np.array([2]), np.full((1, 20), 2.0), dt=dt)
    )

    merged = a.merge(b)

    np.testing.assert_array_equal(merged.spike_ids, [0, 1, 2])
    np.testing.assert_allclose(merged.spike_times, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(merged.voltage_ids, [0, 1, 2])
    assert merged.voltage.shape == (3, 20)
    np.testing.assert_allclose(merged.voltage[2], 2.0)
    assert merged.duration == 10.0


def test_merge_equals_the_single_process_result():
    rng = np.random.default_rng(11)
    ids = np.arange(6)
    times = np.sort(rng.uniform(0, 10.0, size=12))
    spike_ids = rng.integers(0, 6, size=12)
    voltage = rng.normal(size=(6, 20))

    single = (
        Run(duration=10.0)
        .add_spikes(spike_ids, times)
        .add_voltage(ids, voltage, dt=0.5)
    )

    merged = None
    for worker in (ids[:3], ids[3:]):
        mask = np.isin(spike_ids, worker)
        part = (
            Run(duration=10.0)
            .add_spikes(spike_ids[mask], times[mask])
            .add_voltage(worker, voltage[worker], dt=0.5)
        )
        merged = part if merged is None else merged.merge(part)

    order = np.argsort(merged.spike_times, kind="stable")
    np.testing.assert_allclose(merged.spike_times[order], single.spike_times)
    np.testing.assert_array_equal(merged.spike_ids[order], single.spike_ids)
    np.testing.assert_array_equal(merged.voltage_ids, single.voltage_ids)
    np.testing.assert_allclose(merged.voltage, single.voltage)


def _windowed_by_hand(run, start, stop, dt):
    it, tt, iv, vv, im, mp = run
    mask = (tt >= start) & (tt < stop)
    lo, hi = int(start / dt), int(stop / dt)
    return it[mask], tt[mask] - start, iv, vv[:, lo:hi], im, mp[:, lo:hi]


def test_slice_matches_decoding_slice_without_an_env():
    dt = 0.5
    run = _run(duration=20.0, dt=dt, seed=3)

    expected = _windowed_by_hand(run, 5.0, 15.0, dt)

    for actual in (
        tuple(run.slice(5.0, 15.0)),
        tuple(Slice(start=5.0, stop=15.0)(run)),
    ):
        for a, b in zip(actual, expected):
            if a is None or b is None:
                assert a is b
            else:
                np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_slice_metadata_and_defaults():
    run = _run(t0=100.0, duration=20.0, dt=0.5)

    windowed = run.slice(5.0, 15.0)
    assert windowed.t0 == 105.0
    assert windowed.duration == 10.0
    assert windowed["voltage"].t0 == 105.0
    assert windowed.voltage.shape[1] == 20

    assert run.slice(5.0).duration == 15.0

    with pytest.raises(ValueError, match="does not align"):
        run.slice(0.1, 10.0)


def test_select_gids():
    run = _run(duration=10.0, n=4, seed=5)

    selected = run.select(gids=[1, 2])

    assert set(np.unique(selected.spike_ids).tolist()) <= {1, 2}
    np.testing.assert_array_equal(selected.voltage_ids, [1, 2])
    np.testing.assert_allclose(selected.voltage, run.voltage[[1, 2]])
    np.testing.assert_allclose(selected.current, run.current[[1, 2]])

    assert run.select() is run


def test_select_population():
    ranges = {"EXC": (0, 2), "INH": (2, 2)}
    run = _run(duration=10.0, n=4, seed=5)

    selected = run.select(population="INH", population_ranges=ranges)
    np.testing.assert_array_equal(selected.voltage_ids, [2, 3])

    selected = run.select(population=["EXC", "INH"], population_ranges=ranges)
    np.testing.assert_array_equal(selected.voltage_ids, [0, 1, 2, 3])

    with pytest.raises(ValueError, match="requires population_ranges"):
        run.select(population="INH")

    with pytest.raises(ValueError, match="Unknown population"):
        run.select(population="MISSING", population_ranges=ranges)


def test_operations_reject_mismatched_metadata():
    a = _run(duration=10.0, dt=0.5)
    b = _run(duration=10.0, dt=0.25)

    with pytest.raises(ValueError, match="dt"):
        a.concat(b)
    with pytest.raises(ValueError, match="dt"):
        a.merge(b)
    with pytest.raises(TypeError):
        a.concat(tuple(a))


def test_pickles_roundtrip():
    import pickle

    run = _run()
    restored = pickle.loads(pickle.dumps(run))

    np.testing.assert_allclose(restored.spike_times, run.spike_times)
    np.testing.assert_allclose(restored.voltage, run.voltage)
    assert restored.duration == run.duration


@pytest.mark.skipif("ax" not in backend(), reason="requires a jax backend")
@pytest.mark.traces
def test_survives_jit_and_vmap_as_a_return_value():
    import jax
    import jax.numpy as jnp

    from livn.run import IS_PYTREE

    assert IS_PYTREE

    def simulate(v):
        return Run(duration=10.0).add(
            "voltage", jnp.arange(v.shape[0]), v * 2.0, dt=0.5
        )

    v = jnp.ones((3, 20))

    jitted = jax.jit(simulate)(v)
    assert isinstance(jitted, Run)
    np.testing.assert_allclose(np.asarray(jitted.voltage), 2.0)
    assert jitted.duration == 10.0
    assert jitted.voltage_dt == 0.5

    batched = jax.vmap(simulate)(jnp.ones((4, 3, 20)))
    assert isinstance(batched, Run)
    assert batched.voltage.shape == (4, 3, 20)

    leaves = jax.tree_util.tree_leaves(simulate(v))
    assert len(leaves) == 2


@pytest.mark.skipif("ax" in backend(), reason="jax-free install only")
def test_container_module_does_not_import_jax():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, livn.run; assert 'jax' not in sys.modules",
        ],
        check=True,
    )


def _padded_run(duration=10.0):
    times = np.array([[1.0, 4.0, np.inf], [2.0, np.inf, np.inf], [0.5, 3.0, 8.0]])
    return Run(t0=0.0, duration=duration).add_spikes(
        np.arange(3), times, padded=True
    ), times


def test_a_padded_channel_reports_its_rectangle():
    run, times = _padded_run()

    padded = run.spikes.padded
    assert padded is not None
    np.testing.assert_array_equal(np.asarray(padded.times), times)
    np.testing.assert_array_equal(np.asarray(padded.ids), [0, 1, 2])
    np.testing.assert_array_equal(np.asarray(padded.fired), np.isfinite(times))


def test_a_padded_channel_compacts_to_the_ragged_form():
    run, _ = _padded_run()

    np.testing.assert_allclose(
        np.asarray(run.spike_times), [0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
    )
    np.testing.assert_array_equal(np.asarray(run.spike_ids), [2, 0, 1, 2, 0, 2])


def test_a_compact_channel_has_no_rectangle():
    assert _run().spikes.padded is None


def test_compacting_keeps_the_rectangle_alongside():
    run, times = _padded_run()
    compacted = run.spikes.compact()

    assert compacted.padded is not None
    np.testing.assert_array_equal(np.asarray(compacted.padded.times), times)


def test_a_padded_channel_composes_like_any_other():
    run, _ = _padded_run()

    windowed = run.slice(2.0, 5.0)
    np.testing.assert_allclose(np.asarray(windowed.spike_times), [0.0, 1.0, 2.0])

    selected = run.select([2])
    np.testing.assert_allclose(np.asarray(selected.spike_times), [0.5, 3.0, 8.0])


def test_a_raster_bins_the_events_onto_a_grid():
    run, _ = _padded_run()

    raster = np.asarray(run.spikes.raster(1.0))

    assert raster.shape == (3, 11)
    assert raster.sum() == 6
    assert raster[0, 1] and raster[0, 4]
    assert raster[2, 0] and raster[2, 3] and raster[2, 8]
    assert not raster[1, 0] and raster[1, 2]


def test_a_raster_of_a_compact_channel_agrees_with_the_padded_one():
    padded, _ = _padded_run()
    compact = Run(t0=0.0, duration=10.0).add_spikes(
        np.asarray(padded.spike_ids), np.asarray(padded.spike_times)
    )

    np.testing.assert_array_equal(
        np.asarray(padded.spikes.raster(1.0)),
        np.asarray(compact.spikes.raster(1.0, n_cells=3)),
    )


def test_a_batched_rectangle_has_no_ragged_form():
    times = np.full((2, 3, 4), np.inf)
    times[..., 0] = 1.0
    run = Run(t0=0.0, duration=10.0).add_spikes(np.arange(3), times, padded=True)

    assert run.spikes.raster(1.0).shape == (2, 3, 11)
    with pytest.raises(ValueError, match="no ragged form"):
        run.spike_times


def test_events_are_immutable():
    run, _ = _padded_run()
    with pytest.raises(AttributeError, match="immutable"):
        run.spikes.t0 = 5.0


@pytest.mark.skipif("ax" not in backend(), reason="needs jax")
@pytest.mark.traces
def test_compacting_under_a_trace_points_at_the_rectangle():
    import jax
    import jax.numpy as jnp

    def traced(times):
        run = Run(t0=0.0, duration=10.0).add_spikes(jnp.arange(3), times, padded=True)
        return run.spike_times

    times = jnp.asarray([[1.0, jnp.inf], [2.0, 3.0], [4.0, jnp.inf]])
    with pytest.raises(RuntimeError, match=r"\.padded"):
        jax.jit(traced)(times)

    assert (
        jax.jit(
            lambda t: (
                Run(t0=0.0, duration=10.0)
                .add_spikes(jnp.arange(3), t, padded=True)
                .spikes.padded.times.sum()
            )
        )(jnp.zeros((3, 2)))
        == 0.0
    )


def test_declaring_padded_storage_for_a_ragged_list_is_rejected():
    run = Run(t0=0.0, duration=10.0)

    with pytest.raises(ValueError, match=r"padded event times must be"):
        run.add_spikes(np.arange(3), np.array([1.0, 2.0]), padded=True)

    with pytest.raises(ValueError, match="one row id per row"):
        run.add_spikes(np.arange(2), np.zeros((3, 4)), padded=True)
