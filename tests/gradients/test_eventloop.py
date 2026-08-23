import math

import pytest

pytest.importorskip("diffrax")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from livn.models.eventloop import (  # noqa: E402
    SolverConfig,
    event_solve,
    resample,
)


def _ramp(slope=1.0, threshold=1.0, hold=0.0):

    def drift(t, y, args):
        return jnp.array([slope])

    def cond_fn(t, y, args, **kwargs):
        return y[0] - threshold

    def transition(t_event, y, args, mask, key):
        return t_event + hold, jnp.array([0.0])

    def hold_fn(t_event, y, t_resume, ts, args, mask):
        return jnp.full((ts.shape[0], 1), threshold)

    return dict(
        drift=drift,
        cond_fn=cond_fn,
        transition=transition,
        hold_fn=hold_fn if hold else None,
    )


def _solve(t1=5.0, dt=0.1, config=None, **kwargs):
    config = config or SolverConfig(
        dt_solver=0.005, points_per_segment=16, max_rate=2.0
    )
    return event_solve(
        y0=jnp.array([0.0]), t0=0.0, t1=t1, dt=dt, config=config, **_ramp(**kwargs)
    )


def _events(solution):
    times = np.asarray(solution.event_times)
    return times[np.isfinite(times)]


def test_event_times_are_exact():
    solution = _solve()

    np.testing.assert_allclose(_events(solution), [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-4)
    assert not bool(solution.saturated)
    assert bool(solution.solver_ok)


def test_the_trace_resamples_onto_the_requested_grid():
    solution = _solve(t1=3.0)
    ts = jnp.arange(0.0, 3.0001, 0.1)

    ys = solution.sample(ts)[:, 0]

    np.testing.assert_allclose(ys[5], 0.5, atol=1e-3)
    np.testing.assert_allclose(ys[9], 0.9, atol=1e-3)
    np.testing.assert_allclose(ys[10], 0.0, atol=1e-3)
    np.testing.assert_allclose(ys[15], 0.5, atol=1e-3)


def test_a_refractory_hold_delays_the_next_event():
    solution = _solve(t1=6.0, hold=1.0)

    np.testing.assert_allclose(_events(solution), [1.0, 3.0, 5.0], atol=1e-4)

    ys = solution.sample(jnp.array([1.5, 2.5, 3.5]))[:, 0]
    np.testing.assert_allclose(ys, [1.0, 0.5, 1.0], atol=1e-3)


def test_exceeding_the_event_budget_fails_loudly():
    config = SolverConfig(dt_solver=0.005, points_per_segment=16, max_rate=0.4)

    with pytest.raises(Exception, match="event budget exhausted"):
        jax.block_until_ready(_solve(config=config).event_times)


def test_saturation_is_reported_when_it_is_not_raised():
    config = SolverConfig(
        dt_solver=0.005, points_per_segment=16, max_rate=0.4, throw=False
    )

    solution = _solve(config=config)

    assert bool(solution.saturated)
    assert len(_events(solution)) < 5


def test_a_failed_inner_solve_is_not_returned_silently():
    config = SolverConfig(
        dt_solver=0.005, points_per_segment=16, max_rate=2.0, max_steps=3
    )

    with pytest.raises(Exception, match="inner solve failed"):
        jax.block_until_ready(_solve(config=config).event_times)


def test_buffers_are_linear_in_duration():
    sizes = []
    for duration in (10.0, 20.0, 40.0):
        solution = _solve(t1=duration)
        sizes.append(solution.ts.shape[0])

    growth = [b / a for a, b in zip(sizes, sizes[1:])]
    assert all(1.5 < g < 2.5 for g in growth), sizes


def test_event_times_are_differentiable():
    def loss(slope):
        solution = _solve(t1=5.0, slope=slope)
        times = solution.event_times
        return jnp.sum(jnp.where(jnp.isfinite(times), times, 0.0))

    slope = 1.3
    gradient = float(jax.grad(loss)(slope))
    finite_difference = float((loss(slope + 1e-4) - loss(slope - 1e-4)) / 2e-4)

    assert gradient != 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-2)


def test_jit_and_vmap():
    def spikes(slope):
        times = _solve(t1=5.0, slope=slope).event_times
        return jnp.sum(jnp.isfinite(times))

    assert int(eqx.filter_jit(spikes)(1.0)) == 5
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(spikes)(jnp.array([1.0, 2.0]))), [5, 10]
    )


def test_resample_handles_repeated_sample_times():
    ts = jnp.array([0.0, 1.0, 1.0, 2.0])
    ys = jnp.array([[0.0], [1.0], [0.0], [1.0]])

    out = resample(ts, ys, jnp.array([0.5, 1.0, 1.5, 5.0]))[:, 0]

    np.testing.assert_allclose(np.asarray(out), [0.5, 0.0, 0.5, 1.0])


def test_an_unbounded_segment_runs_to_the_next_event():
    config = SolverConfig(
        dt_solver=0.005, points_per_segment=16, max_rate=2.0, max_segment_span=math.inf
    )

    solution = _solve(t1=5.0, config=config)

    np.testing.assert_allclose(_events(solution), [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-4)
    assert int(solution.blocks) == 5


def test_the_brownian_path_survives_reverse_mode():
    import diffrax

    def gradient_through(path_cls):
        def loss(slope):
            path = path_cls(-1.0, 6.0, tol=1e-3, shape=(1,), key=jax.random.PRNGKey(0))
            solution = event_solve(
                y0=jnp.array([0.0]),
                t0=0.0,
                t1=5.0,
                dt=0.1,
                config=SolverConfig(
                    dt_solver=0.01, points_per_segment=16, max_rate=2.0
                ),
                extra_terms=[
                    diffrax.ControlTerm(lambda t, y, args: jnp.array([[0.2]]), path)
                ],
                **_ramp(slope=slope),
            )
            return jnp.sum(jnp.where(jnp.isfinite(solution.event_times), 1.0, 0.0))

        return jax.grad(loss)(1.0)

    from livn.models.eventloop import BrownianPath

    assert np.isfinite(float(gradient_through(BrownianPath)))

    with pytest.raises(RuntimeError, match="cannot be autodifferentiated"):
        gradient_through(diffrax.VirtualBrownianTree)
