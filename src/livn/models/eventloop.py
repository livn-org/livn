"""Event-driven solver harness.

## License information

Parts of this module (the segment loop and BrownianPath) are adapted
from the snnax project (https://github.com/cholberg/snnax), Copyright 2024
Christian Holberg, and were received under the MIT license.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from diffrax import BrownianIncrement, SpaceTimeLevyArea, VirtualBrownianTree
from diffrax._brownian.tree import _levy_diff, _make_levy_val
from diffrax._custom_types import RealScalarLike, levy_tree_transpose
from diffrax._misc import linear_rescale
from jaxtyping import Array, PyTree

__all__ = [
    "BrownianPath",
    "EventSolution",
    "LoopBudget",
    "SolverConfig",
    "event_solve",
    "loop_budget",
    "resample",
]


class BrownianPath(VirtualBrownianTree):
    @eqxi.doc_remove_args("_spline")
    def __init__(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: tuple[int, ...] | PyTree[jax.ShapeDtypeStruct],
        key,
        levy_area: type[BrownianIncrement | SpaceTimeLevyArea] = BrownianIncrement,
        _spline: str = "sqrt",
    ):
        super().__init__(t0, t1, tol, shape, key, levy_area, _spline)

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike | None = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> PyTree[Array] | BrownianIncrement | SpaceTimeLevyArea:
        t0 = jax.lax.stop_gradient(t0)
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            levy_out = levy_0
            levy_out = jtu.tree_map(_make_levy_val, self.shape, levy_out)
        else:
            t1 = jax.lax.stop_gradient(t1)
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)
            levy_out = jtu.tree_map(_levy_diff, self.shape, levy_0, levy_1)

        levy_out = levy_tree_transpose(self.shape, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, (BrownianIncrement, SpaceTimeLevyArea))
        return levy_out if use_levy else levy_out.W


@dataclasses.dataclass(frozen=True)
class SolverConfig:
    dt_solver: float = 0.01
    points_per_segment: int = 64
    max_segment_span: float | None = None
    max_rate: float = 0.2
    max_steps: int | None = None
    solver: Any = dataclasses.field(default_factory=diffrax.Euler)
    stepsize_controller: Any = dataclasses.field(
        default_factory=diffrax.ConstantStepSize
    )
    root_finder: Any = dataclasses.field(
        default_factory=lambda: optx.Newton(1e-2, 1e-2, optx.rms_norm)
    )
    throw: bool = True


# SolverConfig is frozen, so one shared instance is safe as a default argument
# (and avoids rebuilding the solver/root-finder on every call).
DEFAULT_SOLVER_CONFIG = SolverConfig()


class EventSolution(eqx.Module):
    ts: Any
    ys: Any
    event_times: Any
    event_masks: Any
    t1: Any
    y1: Any
    saturated: Any
    solver_ok: Any
    blocks: Any

    def sample(self, ts_out):
        return resample(self.ts, self.ys, ts_out)


def resample(ts, ys, ts_out):
    n = ts.shape[0]
    i1 = jnp.clip(jnp.searchsorted(ts, ts_out, side="right"), 1, n - 1)
    i0 = i1 - 1
    t_lo, t_hi = ts[i0], ts[i1]
    y_lo, y_hi = ys[i0], ys[i1]

    span = t_hi - t_lo
    safe = span > 0
    weight = jnp.clip(
        jnp.where(safe, (ts_out - t_lo) / jnp.where(safe, span, 1.0), 0.0), 0.0, 1.0
    )
    weight = weight.reshape(weight.shape + (1,) * (ys.ndim - 1))

    return y_lo + weight * (y_hi - y_lo)


@dataclasses.dataclass(frozen=True)
class LoopBudget:
    blocks: int
    events: int
    segments: int
    points_per_block: int
    points_per_hold: int
    span: float
    max_steps: int

    def samples(self) -> int:
        return self.blocks * (self.points_per_block + self.points_per_hold)

    def nbytes(self, state_size: int, itemsize: int = 4) -> int:
        return self.samples() * (state_size + 1) * itemsize


def loop_budget(
    t0: float,
    t1: float,
    dt: float,
    config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    holds=False,
) -> LoopBudget:
    """Size a solve without running it."""
    t0, t1, dt = float(t0), float(t1), float(dt)
    duration = t1 - t0
    if duration <= 0:
        raise ValueError(f"t1 must be after t0, got t0={t0}, t1={t1}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    points = int(config.points_per_segment)
    if points < 2:
        raise ValueError(f"points_per_segment must be >= 2, got {points}")

    span = config.max_segment_span
    span = (points - 1) * dt if span is None else float(span)
    if span <= 0:
        raise ValueError(f"max_segment_span must be positive, got {span}")
    unbounded = math.isinf(span)

    events = math.ceil(duration * float(config.max_rate)) + 1
    segments = 1 if unbounded else math.ceil(duration / span) + 1

    max_steps = config.max_steps
    if max_steps is None:
        reach = duration if unbounded else span
        max_steps = math.ceil(reach / config.dt_solver) + 16

    return LoopBudget(
        blocks=events + segments,
        events=events,
        segments=segments,
        points_per_block=points,
        points_per_hold=points if holds else 1,
        span=span,
        max_steps=int(max_steps),
    )


def _grid(t_a, t_b, n: int, endpoint: bool = True):
    weights = jnp.linspace(0.0, 1.0, n, endpoint=endpoint)
    grid = t_a + (t_b - t_a) * weights
    # ...and clipping, also monotone, keeps the ends from rounding outside
    return jnp.clip(grid, jnp.minimum(t_a, t_b), jnp.maximum(t_a, t_b))


class _LoopState(eqx.Module):
    seg_ts: Any
    seg_ys: Any
    hold_ts: Any
    hold_ys: Any
    event_times: Any
    event_masks: Any
    solver_ok: Any
    block: Any
    events: Any
    t: Any
    y: Any
    key: Any


def _buffers(state: _LoopState):
    return (
        state.seg_ts,
        state.seg_ys,
        state.hold_ts,
        state.hold_ys,
        state.event_times,
        state.event_masks,
        state.solver_ok,
    )


def event_solve(
    *,
    drift: Callable,
    cond_fn: Any,
    transition: Callable,
    y0,
    t0: float,
    t1: float,
    dt: float,
    args: Any = None,
    hold_fn: Callable | None = None,
    extra_terms: Sequence[Any] | None = None,
    key: Any = None,
    config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> EventSolution:
    """Integrate ``drift`` from ``t0`` to ``t1``, restarting at every event.

    Args:
        drift: ``(t, y, args) -> dy``.
        cond_fn: Event condition ``(t, y, args, **kwargs) -> scalar``, or a list
            of them (one per event source).  An event fires on a sign change.
        transition: ``(t_event, y_event, args, event_mask, key) -> (t_resume, y_resume)``.
            ``t_resume`` may be later than ``t_event`` (a refractory window);
            the state in between is described by ``hold_fn``.  ``key`` is a fresh
            split per event, or ``None`` when the solve was given no key.
        y0: Initial state, an array of any shape.
        t0: Start time in ms.  Must be a Python float.
        t1: End time in ms.  Must be a Python float.
        dt: Sample spacing of the recorded trace in ms.  Must be a Python float.
        args: Passed through to ``drift``/``cond_fn``/``transition``.
        hold_fn: ``(t_event, y_event, t_resume, ts, args, event_mask) -> ys``
            evaluating the state analytically across ``[t_event, t_resume)``.
            Omit when the transition is instantaneous.
        extra_terms: Additional diffrax terms (diffusion, input spike trains)
            combined with the drift into a ``MultiTerm``.
        key: PRNG key for a stochastic ``transition``.  It is split once per
            iteration, so the draw a transition sees depends on how many
            segments preceded it, not on wall-clock time.
        config: See :class:`SolverConfig`.

    Returns:
        An :class:`EventSolution`.
    """
    t0, t1, dt = float(t0), float(t1), float(dt)
    duration = t1 - t0

    budget = loop_budget(t0, t1, dt, config, holds=hold_fn is not None)
    n_points = budget.points_per_block
    n_hold = budget.points_per_hold
    n_events = budget.events
    n_blocks = budget.blocks
    span = budget.span
    unbounded = math.isinf(span)
    max_steps = budget.max_steps

    y0 = jnp.asarray(y0)
    y_shape = y0.shape
    dtype = jnp.result_type(y0.dtype, jnp.float32)

    term = diffrax.ODETerm(drift)
    if extra_terms:
        term = diffrax.MultiTerm(term, *extra_terms)
    event = diffrax.Event(cond_fn, config.root_finder)
    n_cond = len(jax.tree_util.tree_leaves(cond_fn))

    def body(state: _LoopState) -> _LoopState:
        block = state.block
        t_start = jnp.minimum(state.t, t1)
        t_stop = (
            jnp.asarray(t1, dtype) if unbounded else jnp.minimum(t_start + span, t1)
        )
        # Under vmap the loop runs until every batch element's cond is false, so a cell
        # that already reached t1 keeps executing this body with its state update masked
        # out. Left alone that is a zero-length solve, and while its value is discarded,
        # its adjoint is not and its degenerate interval can poisons the gradient of the
        # whole batch. So we give those spent iterations a well-conditioned interval
        # which can only fire when t_stop <= t_start i.e. the cell is done.
        t_stop = jnp.where(t_stop > t_start, t_stop, t_start + config.dt_solver)
        ts_seg = _grid(t_start, t_stop, n_points)

        saveat = diffrax.SaveAt(
            subs=(diffrax.SubSaveAt(ts=ts_seg), diffrax.SubSaveAt(t1=True))
        )
        sol = diffrax.diffeqsolve(
            term,
            config.solver,
            t_start,
            t_stop,
            config.dt_solver,
            state.y,
            args,
            throw=False,
            stepsize_controller=config.stepsize_controller,
            saveat=saveat,
            event=event,
            max_steps=max_steps,
        )

        t_event = sol.ts[1][0]
        y_event = sol.ys[1][0]
        mask = jnp.reshape(
            jnp.asarray(jax.tree_util.tree_leaves(sol.event_mask)), (n_cond,)
        )
        slack = config.dt_solver  # a root landing just off the end is rounding
        found = (
            jnp.isfinite(t_event)
            & (t_event >= t_start - slack)
            & (t_event <= t_stop + slack)
        )
        t_event = jnp.clip(jnp.where(found, t_event, t_stop), t_start, t_stop)
        fired = jnp.any(mask) & found

        keep = ts_seg <= t_event
        seg_ts = jnp.where(keep, ts_seg, t_event)
        seg_ys = jnp.where(
            keep.reshape((n_points,) + (1,) * len(y_shape)), sol.ys[0], y_event
        )

        if state.key is None:
            next_key, event_key = None, None
        else:
            next_key, event_key = jr.split(state.key, 2)

        t_resume, y_resume = transition(t_event, y_event, args, mask, event_key)
        t_next = jnp.where(fired, jnp.minimum(t_resume, t1), t_event)
        y_next = jnp.where(fired, y_resume, y_event)

        # The hold covers [t_event, t_next) exclusively; the post-transition
        # state at t_next is carried by the next segment's first sample, so a
        # jump that lands at the end of the window is not smeared by the
        # resampling.
        hold_ts = _grid(t_event, t_next, n_hold, endpoint=False)
        if hold_fn is None:
            hold_ys = jnp.broadcast_to(y_event, (n_hold, *y_shape))
        else:
            hold_ys = jnp.where(
                fired,
                hold_fn(t_event, y_event, t_next, hold_ts, args, mask),
                jnp.broadcast_to(y_event, (n_hold, *y_shape)),
            )

        return _LoopState(
            seg_ts=state.seg_ts.at[block].set(seg_ts),
            seg_ys=state.seg_ys.at[block].set(seg_ys),
            hold_ts=state.hold_ts.at[block].set(hold_ts),
            hold_ys=state.hold_ys.at[block].set(hold_ys),
            event_times=state.event_times.at[block].set(
                jnp.where(fired, t_event, jnp.inf)
            ),
            event_masks=state.event_masks.at[block].set(mask),
            solver_ok=state.solver_ok.at[block].set(
                (
                    (sol.result == diffrax.RESULTS.successful)
                    # terminating on the event is the normal outcome here, not a failure
                    | (sol.result == diffrax.RESULTS.event_occurred)
                )
                & found
            ),
            block=block + 1,
            events=state.events + fired.astype(state.events.dtype),
            t=t_next,
            y=y_next,
            key=next_key,
        )

    def cond(state: _LoopState):
        return (state.t < t1) & (state.events < n_events) & (state.block < n_blocks)

    init = _LoopState(
        seg_ts=jnp.full((n_blocks, n_points), jnp.inf, dtype),
        seg_ys=jnp.zeros((n_blocks, n_points, *y_shape), dtype),
        hold_ts=jnp.full((n_blocks, n_hold), jnp.inf, dtype),
        hold_ys=jnp.zeros((n_blocks, n_hold, *y_shape), dtype),
        event_times=jnp.full((n_blocks,), jnp.inf, dtype),
        event_masks=jnp.zeros((n_blocks, n_cond), bool),
        solver_ok=jnp.ones((n_blocks,), bool),
        block=jnp.zeros((), jnp.int32),
        events=jnp.zeros((), jnp.int32),
        t=jnp.asarray(t0, dtype),
        y=y0.astype(dtype),
        key=key,
    )

    final = eqxi.while_loop(
        cond,
        body,
        init,
        buffers=_buffers,
        max_steps=n_blocks,
        kind="checkpointed",
    )

    # Unwritten blocks repeat the final state so that resampling past the end of
    # the run extrapolates as a constant instead of reading uninitialised memory.
    written = jnp.arange(n_blocks) < final.block
    ts = jnp.concatenate(
        [
            jnp.where(written[:, None], final.seg_ts, final.t),
            jnp.where(written[:, None], final.hold_ts, final.t),
        ],
        axis=1,
    ).reshape(-1)
    keep_y = written.reshape((n_blocks, 1) + (1,) * len(y_shape))
    ys = jnp.concatenate(
        [
            jnp.where(keep_y, final.seg_ys, final.y),
            jnp.where(keep_y, final.hold_ys, final.y),
        ],
        axis=1,
    ).reshape((n_blocks * (n_points + n_hold), *y_shape))

    event_times = jnp.where(written, final.event_times, jnp.inf)
    saturated = final.t < t1
    solver_ok = jnp.all(jnp.where(written, final.solver_ok, True))

    if config.throw:
        checked = eqx.error_if(
            event_times,
            ~solver_ok,
            "the inner solve failed (step budget exhausted or the event "
            "root-find did not converge); its output is not a solution.",
        )
        event_times = eqx.error_if(
            checked,
            saturated,
            f"event budget exhausted before t1: the run needs more than "
            f"{n_events} events (max_rate={config.max_rate}/ms over "
            f"{duration} ms). Raise SolverConfig.max_rate, or set throw=False "
            f"to receive the saturation flag instead of an error.",
        )

    return EventSolution(
        ts=ts,
        ys=ys,
        event_times=event_times,
        event_masks=jnp.where(written[:, None], final.event_masks, False),
        t1=final.t,
        y1=final.y,
        saturated=saturated,
        solver_ok=solver_ok,
        blocks=final.block,
    )
