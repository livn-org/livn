import dataclasses
import json
import os
import re
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Best",
    "Callback",
    "Checkpointer",
    "EarlyStopping",
    "EqxCheckpointer",
    "Every",
    "FitState",
    "History",
    "JsonlLogger",
    "NaNGuard",
    "data_mesh",
    "group_chain",
    "initialize",
]


def group_chain(
    lr,
    warmup: int = 0,
    decay: int | None = None,
    clip_norm: float | None = None,
    trust_region: float | None = None,
    weight_decay: float = 0.0,
):
    """The optimizer of one parameter group.

    ``zero_nans`` -> ``clip_by_global_norm`` -> ``adam`` (``adamw`` when ``weight_decay > 0``)
    with a learning-rate schedule -> an optional per-element ``clip`` of the update.

    Args:
        lr: peak learning rate, or an optax schedule (then ``warmup`` and ``decay`` are ignored)
        warmup: steps of linear warmup from 0 to ``lr``
        decay: steps of cosine decay to 0 after the warmup; ``None`` holds ``lr``
        clip_norm: clip the gradient's global norm to this
        trust_region: clip every element of the update to ``[-trust_region, trust_region]``,
            a bound on how far one step can move a parameter whatever its gradient
        weight_decay: decoupled weight decay (adamw)
    """
    import optax

    if callable(lr):
        schedule = lr
    elif warmup and decay:
        schedule = optax.warmup_cosine_decay_schedule(
            0.0, lr, warmup_steps=warmup, decay_steps=warmup + decay
        )
    elif warmup:
        schedule = optax.join_schedules(
            [optax.linear_schedule(0.0, lr, warmup), optax.constant_schedule(lr)],
            [warmup],
        )
    elif decay:
        schedule = optax.cosine_decay_schedule(lr, decay)
    else:
        schedule = lr

    chain = [optax.zero_nans()]
    if clip_norm is not None:
        chain.append(optax.clip_by_global_norm(clip_norm))
    if weight_decay:
        chain.append(optax.adamw(schedule, weight_decay=weight_decay))
    else:
        chain.append(optax.adam(schedule))
    if trust_region is not None:
        chain.append(optax.clip(trust_region))
    return optax.chain(*chain)


@dataclasses.dataclass
class FitState:
    raw: Any
    opt_state: Any
    step: int
    key: Any = None
    constrain: Callable = lambda raw: raw
    checkpointer: Any = None

    @property
    def theta(self):
        return self.constrain(self.raw)

    def payload(self):
        """for the checkpoint ``(raw, opt_state, step, key)``."""
        return (self.raw, self.opt_state, np.asarray(self.step), self.key)

    def replace(self, **changes) -> "FitState":
        return dataclasses.replace(self, **changes)


class Callback:
    """Plain host object called around the loop. Override what you need.

    ``metrics`` is a dict with

    - ``step``
    - ``loss``: a float, or an array with one entry per problem when ``batched``
    - ``raw``: the point ``loss`` was evaluated at. An optimistix minimiser evaluates one point per
      step, a line-search trial the solver may then reject, so this is not always the point the
      step ends on
    - ``aux``: what the loss returned beside its value (for optimistix, at the last accepted point)
    - ``grad_norm``: when the optimizer exposes the gradient. The norm over the gradient's entries
      that are not NaN; an infinite entry still makes it infinite
    - ``nan_grads``: with ``grad_norm``, how many gradient entries were NaN
    - ``stepped``: whether the problem took a step (``False`` for an optimistix problem that had
      already terminated while others ran on); ``True`` when not batched
    - ``batched``

    ``state`` is the state after this step's update, so it is what a checkpoint taken would hold.
    """

    def on_fit_start(self, state: FitState) -> None:
        pass

    def on_step_end(self, step: int, state: FitState, metrics: dict) -> bool | None:
        """Return ``True`` to stop the fit."""

    def on_fit_end(self, state: FitState) -> FitState | None:
        """Return a state to replace the one the fit ends on (``EarlyStopping`` restores here)."""


def _monitored(metrics: dict, monitor: str):
    """``metrics[monitor]``, else ``metrics["aux"][monitor]``, else the dotted path ``"aux.mse"``."""
    if monitor in metrics:
        value = metrics[monitor]
    elif monitor in metrics["aux"]:
        value = metrics["aux"][monitor]
    else:
        value = metrics
        for part in monitor.split("."):
            value = value[part]
    return np.asarray(value, dtype=float)


class Every(Callback):
    """Call ``fn(step, state, metrics)`` every ``n`` steps, e.g. a host evaluation on ``Run``s.

    A truthy return value stops the fit.
    """

    def __init__(self, n: int, fn: Callable):
        self.n, self.fn = int(n), fn

    def on_step_end(self, step, state, metrics):
        if step % self.n == 0:
            return self.fn(step, state, metrics)
        return None


class Best(Callback):
    def __init__(self, monitor: str = "loss"):
        self.monitor = monitor
        self.loss: Any = None
        self.step: Any = None
        self.raw = None
        self._constrain = lambda raw: raw

    def on_fit_start(self, state):
        self.loss = self.step = self.raw = None
        self._constrain = state.constrain

    @property
    def theta(self):
        return None if self.raw is None else self._constrain(self.raw)

    def problem(self, i: int) -> dict:
        """``{"loss", "step", "theta"}`` of problem ``i`` of a ``batched`` fit."""
        import jax.tree_util as jtu

        return {
            "loss": float(self.loss[i]),
            "step": int(self.step[i]),
            "theta": jtu.tree_map(lambda x: x[i], self.theta),
        }

    def on_step_end(self, step, state, metrics):
        import jax.numpy as jnp
        import jax.tree_util as jtu

        value = _monitored(metrics, self.monitor)
        value = np.where(np.isnan(value), np.inf, value)
        new = metrics["raw"]
        if self.loss is None:
            self.loss, self.raw = value, new
            self.step = np.full(value.shape, step) if value.ndim else step
            return

        better = value < self.loss
        self.loss = np.where(better, value, self.loss)
        if not value.ndim:
            if better:
                self.raw, self.step = new, step
            return

        self.step = np.where(better, step, self.step)

        def pick(n, o):
            mask = jnp.asarray(better).reshape(better.shape + (1,) * (jnp.ndim(n) - 1))
            return jnp.where(mask, n, o)

        self.raw = jtu.tree_map(pick, new, self.raw)
        return


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
        min_rel_delta: float = 0.0,
        stop_below: float | None = None,
    ):
        self.monitor, self.patience = monitor, int(patience)
        self.min_delta, self.restore_best = float(min_delta), restore_best
        self.min_rel_delta = float(min_rel_delta)
        self.stop_below = None if stop_below is None else float(stop_below)
        self.best = np.inf
        self.best_step = None
        self.stopped_step = None
        self._raw = None

    def on_fit_start(self, state):
        self.best, self.best_step, self.stopped_step, self._raw = (
            np.inf,
            None,
            None,
            None,
        )

    def on_step_end(self, step, state, metrics):
        value = float(np.mean(_monitored(metrics, self.monitor)))
        if np.isfinite(self.best):
            needed = max(self.min_delta, self.min_rel_delta * abs(self.best))
            improved = value < self.best - needed
        else:
            improved = value < self.best
        reached = self.stop_below is not None and value <= self.stop_below
        if improved or (reached and value < self.best):
            self.best, self.best_step = value, step
            self._raw = metrics["raw"]
        stale = step - (-1 if self.best_step is None else self.best_step)
        if reached or stale >= self.patience:
            self.stopped_step = step
            return True
        return None

    def on_fit_end(self, state):
        if self.restore_best and self._raw is not None:
            return state.replace(raw=self._raw)
        return None


class NaNGuard(Callback):
    """Stop on a failing loss or gradient, leaving the last good state as the checkpoint.

    ``on`` says what fails a step:

    - ``"loss"``: the loss is not finite
    - ``"grad"``: the gradient norm is not finite. ``grad_norm`` leaves NaN entries out, so this is
      an infinite gradient entry, not a NaN one
    - ``"nan_grad"``: any gradient entry is NaN
    """

    def __init__(self, checkpointer=None, on=("loss", "grad")):
        on = (on,) if isinstance(on, str) else tuple(on)
        unknown = set(on) - {"loss", "grad", "nan_grad"}
        if unknown:
            raise ValueError(
                f"NaNGuard watches 'loss', 'grad' and 'nan_grad', not {sorted(unknown)}"
            )
        self.checkpointer, self.on = checkpointer, on
        self.tripped_at = None
        self.reason = None
        self.nan_grad_steps = 0
        self._recent: deque = deque(maxlen=2)

    def on_fit_start(self, state):
        self.tripped_at = self.reason = None
        self.nan_grad_steps = 0
        self._recent.clear()

    def on_step_end(self, step, state, metrics):
        loss = np.asarray(metrics["loss"], float)
        norm = np.asarray(metrics.get("grad_norm", 0.0), float)
        nans = np.asarray(metrics.get("nan_grads", 0))
        self.nan_grad_steps += int(np.any(nans > 0))
        if "loss" in self.on and not np.all(np.isfinite(loss)):
            self.reason = f"loss {float(np.min(loss))}"
        elif "grad" in self.on and not np.all(np.isfinite(norm)):
            self.reason = f"gradient norm {float(np.max(norm))}"
        elif "nan_grad" in self.on and np.any(nans > 0):
            self.reason = f"{int(np.max(nans))} NaN gradient entries"
        else:
            self._recent.append(state)
            return None

        self.tripped_at = step
        target = self.checkpointer or state.checkpointer
        if target is not None and len(self._recent) == 2:
            good = self._recent[0]
            target.save(good.step, good)
        return True


def _jsonable(value):
    return np.asarray(value).tolist()


class JsonlLogger(Callback):
    """One JSON object per step appended to ``path``.

    Only process 0 writes.
    """

    def __init__(self, path, every: int = 1):
        self.path, self.every = Path(path), int(every)
        self._file = None

    def on_fit_start(self, state):
        import jax

        if jax.process_index() == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a")  # noqa: SIM115

    def on_step_end(self, step, state, metrics):
        if self._file is None or step % self.every:
            return
        row = {"step": step, "loss": _jsonable(metrics["loss"])}
        for name in ("grad_norm", "nan_grads"):
            if name in metrics:
                row[name] = _jsonable(metrics[name])
        row.update({k: _jsonable(v) for k, v in metrics["aux"].items()})
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()
        return

    def on_fit_end(self, state):
        if self._file is not None:
            self._file.close()
            self._file = None
        return


class History(Callback):
    """Every evaluation of the loss, as numpy.

    ``fit`` attaches one and returns :meth:`result` as ``Fit.history``, a dict with

    - ``loss``: the loss at each step's evaluated point, then one more entry for the final
      evaluation. An optimistix minimiser evaluates a line-search trial each step, so its entries
      include points the solver rejected; ``len(loss) == steps + 1``
    - ``params``: the constrained parameters at the same points
    - ``aux``: the loss's aux at the same points (for optimistix, at the last accepted point)
    - ``grad_norm``, ``nan_grads``: when the optimizer exposes the gradient (see :class:`Callback`);
      one entry per step
    - ``steps``: how many steps were taken: an int, or an array with one entry per problem when
      ``batched`` (an optimistix problem stops stepping once it terminates)
    - ``result``: for an optimistix minimiser, why each problem stopped, as the text of
      ``optimistix.RESULTS`` (a list with one entry per problem when ``batched``)

    When ``batched`` every array has the problem axis first. Leaves of the parameters with more
    than ``max_size`` elements are not recorded.
    """

    def __init__(self, params: bool = True, max_size: int = 1 << 16):
        self.params, self.max_size = params, int(max_size)
        self.batched = False
        self.termination = None
        self._reset()

    def _reset(self):
        self._loss, self._grad, self._nan, self._params, self._aux = (
            [],
            [],
            [],
            [],
            {},
        )
        self._steps = 0
        self.termination = None
        self._constrain = lambda raw: raw

    def on_fit_start(self, state):
        self._reset()
        self._constrain = state.constrain

    def on_step_end(self, step, state, metrics):
        self.record(
            metrics["raw"],
            metrics["loss"],
            metrics["aux"],
            metrics.get("grad_norm"),
            metrics.get("nan_grads"),
        )
        self._steps = self._steps + np.asarray(metrics.get("stepped", True), dtype=int)
        self.batched = bool(metrics.get("batched", False))
        return

    def record(self, raw, loss, aux, grad_norm=None, nan_grads=None):
        import jax
        import jax.tree_util as jtu

        self._loss.append(np.asarray(loss, dtype=float))
        if grad_norm is not None:
            self._grad.append(np.asarray(grad_norm, dtype=float))
        if nan_grads is not None:
            self._nan.append(np.asarray(nan_grads, dtype=int))
        for name, value in (aux or {}).items():
            self._aux.setdefault(name, []).append(np.asarray(value))
        if self.params:
            theta = self._constrain(raw)
            self._params.append(
                jtu.tree_map(
                    lambda x: (
                        np.asarray(jax.device_get(x))
                        if np.size(x) <= self.max_size
                        else None
                    ),
                    theta,
                )
            )

    def result(self) -> dict:
        import jax.tree_util as jtu

        axis = 1 if self.batched else 0

        def stack(rows):
            return np.stack(rows, axis=axis)

        out: dict = {"loss": stack(self._loss)}
        if self._grad:
            out["grad_norm"] = stack(self._grad)
        if self._nan:
            out["nan_grads"] = stack(self._nan)
        if self._aux:
            out["aux"] = {k: stack(v) for k, v in self._aux.items()}
        if self._params:
            out["params"] = jtu.tree_map(lambda *xs: stack(xs), *self._params)
        steps = np.asarray(self._steps, dtype=int)
        out["steps"] = int(steps) if steps.ndim == 0 else steps
        if self.termination is not None:
            out["result"] = self.termination
        return out


@runtime_checkable
class Checkpointer(Protocol):
    """Where a fit keeps its state so a later call can resume.

    ``every`` is the step interval of periodic saves. ``save`` takes the :class:`FitState` (or its
    :meth:`~FitState.payload`); ``restore`` takes the payload layout
    ``(raw, opt_state, step, key)`` as ``template` and
    returns the same layout filled from disk.
    """

    every: int

    def save(self, step: int, state) -> None: ...

    def latest(self) -> int | None: ...

    def restore(self, step: int, template): ...


_STEP = re.compile(r"step_(\d+)\.eqx$")


class EqxCheckpointer:
    """``eqx.tree_serialise_leaves`` of ``(theta, opt_state, step, key)``, atomically.

    Every save is written to a temporary file and moved into place with ``os.replace``, so a
    crash leaves the previous checkpoint intact. ``manifest.json`` lists the steps kept and holds
    ``document``, whatever the caller wants to travel with the run (a config, the command that
    made it).

    Only process 0 writes; every process may restore.

    Args:
        directory: where the checkpoints go
        every: steps between periodic saves
        keep: how many checkpoints to keep (the oldest are deleted)
        document: JSON-serialisable, stored in the manifest
    """

    def __init__(self, directory, every: int = 100, keep: int = 3, document=None):
        self.directory = Path(directory)
        self.every, self.keep, self.document = int(every), int(keep), document

    def _path(self, step: int) -> Path:
        return self.directory / f"step_{int(step):08d}.eqx"

    def steps(self) -> list[int]:
        if not self.directory.is_dir():
            return []
        found = (_STEP.match(p.name) for p in self.directory.iterdir())
        return sorted(int(m.group(1)) for m in found if m)

    def latest(self) -> int | None:
        steps = self.steps()
        return steps[-1] if steps else None

    def save(self, step: int, state) -> None:
        import equinox as eqx
        import jax

        payload = state.payload() if hasattr(state, "payload") else state
        if jax.process_index() != 0:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(step)
        tmp = path.with_suffix(".tmp")
        eqx.tree_serialise_leaves(tmp, payload)
        os.replace(tmp, path)

        steps = self.steps()
        for old in steps[: -self.keep] if self.keep > 0 else []:
            self._path(old).unlink(missing_ok=True)
        manifest = {
            "latest": int(step),
            "steps": self.steps(),
            "document": self.document,
        }
        tmp = self.directory / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, self.directory / "manifest.json")

    def restore(self, step: int, template):
        import equinox as eqx

        return eqx.tree_deserialise_leaves(self._path(step), template)

    def manifest(self) -> dict | None:
        path = self.directory / "manifest.json"
        return json.loads(path.read_text()) if path.is_file() else None


def initialize(comm=None) -> None:
    """Bring up ``jax.distributed`` for a multi-process job; a no-op for a single process.

    Under SLURM the coordinator address comes from the scheduler's environment. Given an mpi4py
    communicator of more than one rank, jax's own ``mpi4py`` cluster detection is used, which is
    how this coexists with livn's MPI. Call it before any other jax computation.
    """
    import jax

    if jax.distributed.is_initialized():
        return
    if comm is not None:
        if comm.Get_size() > 1:
            jax.distributed.initialize(cluster_detection_method="mpi4py")
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        jax.distributed.initialize()


def data_mesh(devices=None):
    """A one-axis ``"batch"`` mesh over every device of every process, with ``Auto`` axes.

    ``jax.make_mesh`` defaults to ``Explicit`` axes, under which an ordinary loss raises
    ``ShardingTypeError``; ``Auto`` lets XLA propagate the sharding and insert the all-reduce.
    """
    import jax
    from jax.sharding import AxisType

    return jax.make_mesh(
        (jax.device_count() if devices is None else len(devices),),
        ("batch",),
        axis_types=(AxisType.Auto,),
        devices=devices,
    )
