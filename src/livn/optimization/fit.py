import contextlib
import dataclasses
import functools
import inspect
import itertools
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optimistix as optx

from livn.optimization.train import Best, EqxCheckpointer, FitState, History
from livn.optimization.transforms import BIJECTORS, Bijector, bounded

__all__ = [
    "Fit",
    "Problem",
    "Step",
    "Stepper",
    "fit",
    "select",
    "stack",
]


def select(model, where):
    """A bool pytree selecting the trainable leaves of ``where(model)``."""
    mask = jtu.tree_map(lambda _: False, model)
    chosen = where(model)
    return eqx.tree_at(where, mask, jtu.tree_map(eqx.is_inexact_array_like, chosen))


class Step(NamedTuple):
    raw: Any
    state: Any
    evaluated: Any
    loss: Any
    aux: Any
    grad_norm: Any
    nan_grads: Any
    finished: Any
    stepped: Any = True
    result: Any = None


def _grad_stats(grads):
    """``(norm over the entries that are not NaN, number of NaN entries)``; inf stays inf."""
    import optax

    leaves = jtu.tree_leaves(grads)
    if not leaves:
        return jnp.zeros(()), jnp.zeros((), jnp.int32)
    nans = jnp.sum(jnp.stack([jnp.sum(jnp.isnan(g)) for g in leaves]))
    clean = jtu.tree_map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    return optax.tree.norm(clean), nans.astype(jnp.int32)


class Stepper(Protocol):
    """``fit``optimizer contract.

    ``args`` is ``(static_model, batch, key)``.
    """

    def init(self, raw, args): ...

    def step(self, raw, state, args) -> Step: ...

    def done(self, step: Step) -> bool: ...


def _tree_where(pred, a, b):
    """``a`` where ``pred`` else ``b``, on array leaves; ``b``'s statics are kept."""
    pa, _ = eqx.partition(a, eqx.is_array)
    pb, sb = eqx.partition(b, eqx.is_array)
    return eqx.combine(jtu.tree_map(lambda x, y: jnp.where(pred, x, y), pa, pb), sb)


def _compile(fn, n_lead, vmap_axes, jit):
    def run(*lead, args):
        static, batch, key = args

        def inner(*mapped):
            return fn(*mapped[:n_lead], static, *mapped[n_lead:])

        mapped = inner
        if vmap_axes is not None:
            mapped = eqx.filter_vmap(inner, in_axes=(0,) * n_lead + vmap_axes[1:])
        return mapped(*lead, batch, key)

    return eqx.filter_jit(run) if jit else run


class _Optax:
    def __init__(self, tx, objective, jit, vmap_axes):
        import optax

        self.tx = optax.with_extra_args_support(tx)

        def one(raw, state, static, batch, key):
            (value, aux), grads = jax.value_and_grad(objective, has_aux=True)(
                raw, static, batch, key
            )
            updates, state = self.tx.update(
                grads,
                state,
                raw,
                value=value,
                grad=grads,
                value_fn=lambda p: objective(p, static, batch, key)[0],
            )
            new = optax.apply_updates(raw, updates)
            norm, nans = _grad_stats(grads)
            return Step(new, state, raw, value, aux, norm, nans, jnp.array(False))

        self._step = _compile(one, 2, vmap_axes, jit)
        self._init = eqx.filter_vmap(self.tx.init) if vmap_axes else self.tx.init

    def init(self, raw, args):
        return self._init(raw)

    def step(self, raw, state, args):
        return self._step(raw, state, args=args)

    def done(self, step):
        return False


class _Probe(eqx.Module):
    search: Any  # the wrapped search's own state
    loss: Any  # the loss of the point the search was last shown


class _ProbeSearch(optx.AbstractSearch):
    inner: Any

    def init(self, y, f_info_struct):
        dtype = f_info_struct.as_min().dtype
        return _Probe(self.inner.init(y, f_info_struct), jnp.asarray(jnp.inf, dtype))

    def step(self, first_step, y, y_eval, f_info, f_eval_info, state):
        size, accept, result, inner = self.inner.step(
            first_step, y, y_eval, f_info, f_eval_info, state.search
        )
        return size, accept, result, _Probe(inner, f_eval_info.as_min())


class _Optimistix:
    def __init__(self, solver, objective, jit, vmap_axes):
        tags = frozenset()
        probed = hasattr(solver, "search")
        if probed:
            solver = eqx.tree_at(
                lambda s: s.search, solver, _ProbeSearch(solver.search)
            )

        def fn(y, args):
            return objective(y, *args)

        def init(raw, static, batch, key):
            args = (static, batch, key)
            f_struct, aux_struct = jax.eval_shape(lambda r: fn(r, args), raw)
            return solver.init(fn, raw, args, {}, f_struct, aux_struct, tags)

        def one(raw, state, static, batch, key):
            args = (static, batch, key)
            finished, _ = solver.terminate(fn, raw, args, {}, state, tags)
            new, new_state, _ = solver.step(fn, raw, args, {}, state, tags)
            new = _tree_where(finished, raw, new)
            new_state = _tree_where(finished, state, new_state)
            done, result = solver.terminate(fn, new, args, {}, new_state, tags)
            nan = jnp.zeros((), jnp.int32)
            norm = None
            grad = getattr(getattr(new_state, "f_info", None), "grad", None)
            if grad is not None:
                norm, nan = _grad_stats(grad)
            f_last = new_state.f_info.f
            if probed and hasattr(state, "y_eval"):
                evaluated = _tree_where(finished, new, state.y_eval)
                loss = jnp.where(finished, f_last, new_state.search_state.loss)
            else:
                evaluated, loss = new, f_last
            return Step(
                new,
                new_state,
                evaluated,
                loss,
                new_state.aux,
                norm,
                nan,
                done | (result != optx.RESULTS.successful),
                ~finished,
                result,
            )

        self._init = _compile(lambda raw, *a: init(raw, *a), 1, vmap_axes, jit)
        self._step = _compile(one, 2, vmap_axes, jit)

    def init(self, raw, args):
        return self._init(raw, args=args)

    def step(self, raw, state, args):
        return self._step(raw, state, args=args)

    def done(self, step):
        return bool(np.all(np.asarray(step.finished)))


@dataclasses.dataclass
class Problem:
    theta: Any
    history: dict
    best: dict | None
    model: Any


@dataclasses.dataclass
class Fit:
    model: Any
    theta: Any
    history: dict
    state: FitState
    best: Best | None = None
    _assemble: Callable | None = dataclasses.field(default=None, repr=False)
    _internal: Any = dataclasses.field(default=None, repr=False)
    _batched: bool = dataclasses.field(default=False, repr=False)

    def model_at(self, i: int):
        """The fitted model of problem ``i`` of a ``batched`` fit."""
        assert self._assemble is not None
        return self._assemble(jtu.tree_map(lambda x: x[i], self._internal))

    def problem(self, i: int) -> Problem:
        """Problem ``i`` of a ``batched`` fit with its theta, history, best point and model."""
        if not self._batched:
            raise ValueError(
                "problem(i) is for a batched fit; this one has one problem"
            )

        def pick(x):
            return x[i]

        history = {}
        for name, value in self.history.items():
            if name == "result":
                history[name] = value[i]
            else:
                history[name] = jtu.tree_map(pick, value)
        return Problem(
            jtu.tree_map(pick, self.theta),
            history,
            None if self.best is None else self.best.problem(i),
            self.model_at(i),
        )


def stack(models, params=None):
    """Stack the trainable leaves of identically structured ``models`` on a new leading axis."""
    models = list(models)
    if not models:
        raise ValueError("stack() needs at least one model")
    first, treedef = jtu.tree_flatten(models[0])
    layout = _Layout(models[0], params)
    columns = [first]
    for k, model in enumerate(models[1:], start=1):
        leaves, other = jtu.tree_flatten(model)
        if other != treedef:
            raise ValueError(f"model {k} is not structured like model 0")
        columns.append(leaves)
    stacked = list(first)
    for i in range(len(first)):
        values = [column[i] for column in columns]
        if i in layout.taken:
            stacked[i] = jnp.stack([jnp.asarray(_to_array(v)) for v in values])
            continue
        for k, v in enumerate(values[1:], start=1):
            if not _same(values[0], v):
                raise ValueError(
                    f"model {k} differs from model 0 in a leaf that is not trained, which would "
                    "be shared by every problem: select it in `params` or make them equal"
                )
    return jtu.tree_unflatten(treedef, stacked)


def _same(a, b) -> bool:
    if eqx.is_array_like(a) and eqx.is_array_like(b):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
    return bool(a == b)


def _is_selector(spec) -> bool:
    """A function, as opposed to a pytree (which may itself be callable, like a masked ``Env``)."""
    return callable(spec) and jtu.treedef_is_leaf(jtu.tree_structure(spec))


class _Slot:
    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index


_NOT_A_SELECTION = object()


class _Layout:
    def __init__(self, model, params):
        self.leaves, self.treedef = jtu.tree_flatten(model)
        self.grouped = isinstance(params, dict)
        self.train: dict = {}  # group -> indices of the leaves that train
        self.chosen: dict = {}  # group -> (treedef, indices) of what its selector returns
        self._slots = None

        if self.grouped:
            for label, spec in params.items():
                if _is_selector(spec):
                    self._add(label, self._select(spec, label))
                else:
                    self._add_flags(label, self._flags(spec, label))
        else:
            got = _NOT_A_SELECTION
            if _is_selector(params):
                try:
                    got = self._select(params, None)
                except Exception:
                    got = _NOT_A_SELECTION
            if got is _NOT_A_SELECTION:
                self._add_flags(None, self._flags(params, None))
            elif isinstance(got, dict):
                self.grouped = True
                self._add(None, got)
            else:
                self._add(None, got)
                self.train[None] = sorted(self.train[None])
                del self.chosen[None]

        taken: set = set()
        for indices in self.train.values():
            if taken & set(indices):
                raise ValueError("two parameter groups select the same leaf")
            taken |= set(indices)
        self.taken = taken

    def _select(self, spec, label):
        if self._slots is None:
            self._slots = jtu.tree_unflatten(
                self.treedef, [_Slot(i) for i in range(len(self.leaves))]
            )
        got = spec(self._slots)
        flat = jtu.tree_leaves(got)
        if not all(isinstance(x, _Slot) for x in flat) or (
            label is None and not flat and not isinstance(got, dict)
        ):
            if label is None:
                return _NOT_A_SELECTION
            raise ValueError(
                f"the selector of params[{label!r}] must return a leaf of the model, "
                "a tuple (or a module) of them, or a dict of those"
            )
        return got

    def _add(self, label, got):
        if isinstance(got, dict):
            for key, part in got.items():
                self._add_one(key, part, expanded=True)
        else:
            self._add_one(label, got)

    def _add_one(self, label, got, expanded=False):
        if label in self.train:
            raise ValueError(
                f"parameter group {label!r} is defined twice"
                + (" (a selector returned a dict with that key)" if expanded else "")
            )
        got_flat, got_def = jtu.tree_flatten(got)
        indices = [x.index for x in got_flat]
        self.chosen[label] = (got_def, indices)
        self.train[label] = [
            i for i in indices if eqx.is_inexact_array_like(self.leaves[i])
        ]

    def _add_flags(self, label, flags):
        if label in self.train:
            raise ValueError(f"parameter group {label!r} is defined twice")
        self.train[label] = [i for i, flag in enumerate(flags) if flag]

    def _flags(self, spec, label):
        if spec is None:
            spec = eqx.is_inexact_array_like
        if _is_selector(spec):
            return [bool(spec(x)) for x in self.leaves]
        if jtu.tree_structure(spec) != self.treedef:
            where = "params" if label is None else f"params[{label!r}]"
            raise ValueError(
                f"{where} is neither a selector (a function of the model returning its "
                "leaves) nor a bool pytree shaped like the model"
            )
        return [bool(x) for x in jtu.tree_leaves(spec)]

    def split(self):
        raw = {
            label: [_to_array(self.leaves[i]) for i in indices]
            for label, indices in self.train.items()
        }
        static = tuple(
            None if i in self.taken else leaf for i, leaf in enumerate(self.leaves)
        )
        if self.grouped:
            return static, raw
        return static, self._shaped(raw[None])

    def _shaped(self, part):
        return self._shaped_group(None, part)

    def assemble(self, static, theta):
        parts = theta if self.grouped else {None: jtu.tree_leaves(theta)}
        leaves = list(static)
        for label, indices in self.train.items():
            for i, value in zip(indices, parts[label], strict=True):
                leaves[i] = value
        return jtu.tree_unflatten(self.treedef, leaves)

    def present(self, theta):
        if not self.grouped:
            return theta
        shown = {}
        for label, part in theta.items():
            if label not in self.chosen:
                shown[label] = self._shaped_group(label, part)
                continue
            got_def, indices = self.chosen[label]
            value = dict(zip(self.train[label], part, strict=True))
            shown[label] = jtu.tree_unflatten(
                got_def, [value.get(i, self.leaves[i]) for i in indices]
            )
        return shown

    def _shaped_group(self, label, part):
        value = dict(zip(self.train[label], part, strict=True))
        return jtu.tree_unflatten(
            self.treedef, [value.get(i) for i in range(len(self.leaves))]
        )


def _to_array(x):
    return jnp.asarray(x, dtype=jnp.result_type(float)) if isinstance(x, float) else x


def _bijector(spec) -> Bijector:
    if isinstance(spec, Bijector):
        return spec
    if isinstance(spec, tuple):
        return bounded(*spec)
    if spec == "bounded":
        raise ValueError(
            "a 'bounded' transform needs its box: pass (lo, hi) or "
            "livn.optimization.transforms.bounded(lo, hi)"
        )
    return Bijector(*BIJECTORS[spec])


def _apply_transform(theta, spec, direction):
    if spec is None:
        return theta
    if isinstance(spec, dict):
        if not isinstance(theta, dict):
            raise TypeError("a transform dict needs a theta dict to name entries of")
        return {
            k: _apply_transform(v, spec.get(k), direction) for k, v in theta.items()
        }
    bijector = _bijector(spec)[direction]
    return jtu.tree_map(lambda x: bijector(jnp.asarray(x)), theta)


def _check_transform(theta, spec, path=""):
    if not isinstance(spec, dict):
        return
    if not isinstance(theta, dict):
        raise TypeError(
            f"transform names entries of {path or 'theta'}, which is not a dict"
        )
    unknown = sorted(set(spec) - set(theta))
    if unknown:
        raise KeyError(f"transform names parameters that are not fitted: {unknown}")
    for k, v in spec.items():
        _check_transform(theta[k], v, f"{path}{k}.")


def _kind(data):
    if data is None:
        return "none"
    if hasattr(data, "iterate"):
        return "pipeline"
    if isinstance(
        data, (list, tuple, dict, np.ndarray, np.generic, jax.Array, int, float)
    ):
        return "fixed"
    if callable(data):
        return "callable"
    if hasattr(data, "__next__") or hasattr(data, "__iter__"):
        return "iterable"
    return "fixed"


class _Source:
    def __init__(self, data, start, mesh=None):
        self.kind = _kind(data)
        self.data = data
        self.mesh = mesh
        self._cached = None
        if self.kind == "pipeline":
            process, processes = (
                (jax.process_index(), jax.process_count()) if mesh else (None, None)
            )
            self.iterator = iter(
                data.iterate(start_step=start, process=process, processes=processes)
            )
        elif self.kind == "iterable":
            self.iterator = itertools.islice(iter(data), start, None)

    @property
    def fixed(self):
        return self.kind in ("none", "fixed")

    def get(self, step):
        if self.kind == "none":
            return None
        if self.fixed:
            if self._cached is None:
                self._cached = self._place(self.data)
            return self._cached
        if self.kind == "callable":
            return self._place(self.data(step))
        try:
            return self._place(next(self.iterator))
        except StopIteration:
            raise ValueError(f"data ran out at step {step}") from None

    def _place(self, batch):
        return _place(self.mesh, batch, self.fixed) if self.mesh is not None else batch


def _place(mesh, batch, fixed):
    from jax.sharding import NamedSharding, PartitionSpec

    processes, process = jax.process_count(), jax.process_index()
    sharded = NamedSharding(mesh, PartitionSpec("batch"))
    replicated = NamedSharding(mesh, PartitionSpec())

    def leaf(x):
        if np.ndim(x) == 0:
            return jax.device_put(x, replicated)
        x = np.asarray(x)
        if fixed and processes > 1:
            size = x.shape[0] // processes
            x = x[process * size : (process + 1) * size]
        return jax.make_array_from_process_local_data(sharded, x)

    return jtu.tree_map(leaf, batch)


def _replicate(mesh, tree):
    from jax.sharding import NamedSharding, PartitionSpec

    replicated = NamedSharding(mesh, PartitionSpec())
    return jtu.tree_map(
        lambda x: jax.device_put(x, replicated) if eqx.is_array(x) else x, tree
    )


def _auto(mesh):
    from jax.sharding import AxisType, Mesh

    if all(t == AxisType.Auto for t in mesh.axis_types):
        return mesh
    return Mesh(
        mesh.devices, mesh.axis_names, axis_types=(AxisType.Auto,) * mesh.devices.ndim
    )


def _accepts_key(model) -> bool:
    """Whether the callable under ``model`` takes a ``key`` keyword."""
    fn = model.func if isinstance(model, eqx.Partial) else model
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (
            p.name == "key"
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for p in parameters
    )


def _prior_call(prior):
    if prior is None:
        return None
    try:
        parameters = list(inspect.signature(prior).parameters.values())
    except (TypeError, ValueError):
        return None
    for p in parameters[1:]:
        if p.name == "batch" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            return "keyword"
    positional = [
        p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2 and positional[1].default is p.empty:
        return "positional"
    return None


def fit(
    model,
    loss,
    *,
    data=None,
    params=None,
    transform=None,
    prior=None,
    optimizer=None,
    steps: int = 100,
    batched: bool = False,
    callbacks=(),
    checkpoint=None,
    resume: bool = True,
    mesh=None,
    key=None,
    jit: bool = True,
) -> Fit:
    """Fit a callable pytree by gradient descent through its output.

    ``model`` holds the trainable leaves and is the forward map that each step computes
    ``output = model(batch)`` (``model()`` without ``data``; ``model(batch, key=...)`` when ``key``
    is given and the callable takes one) and then ``loss(output, batch)``. A ``functools.partial``
    is converted on entry to the pytree ``equinox.Partial``.

    Args:
        model: the callable pytree being fitted: a partial of an ``Env``, an inverse model, an
            IO stack ...
        loss: ``loss(output, batch) -> scalar`` or ``(scalar, aux_dict)``; ``output`` is what
            ``model`` returned. ``aux`` reaches callbacks and the history.
        data: what ``model`` is called on and ``loss`` compares against, as ``batch``.

            - ``None``
            - a fixed pytree (list, tuple, dict, array...): used every step
            - a ``callable(step) -> batch``
            - a step-addressed pipeline exposing ``iterate(start_step, process, processes)``
              (:class:`livn.optimization.data.Pipeline`): it yields the batches of steps
              ``start_step``, ``start_step + 1``, ... as pure functions of ``(seed, step)``, only
              the ``process`` slice of each global batch (as numpy) when a mesh is given, so
              resuming needs no iterator state
            - any other iterable of batches, e.g. a generator; resuming skips the batches already
              consumed

            With ``batched`` every batch carries the leading problem axis, and with ``mesh`` its
            leading axis is the batch axis that is sharded.
        params:

            - ``None``: every inexact leaf, Python floats included (e.g. ``livn.io`` media keep their
              physical parameters as floats)
            - a predicate on leaves, or a bool pytree shaped like ``model`` (see :func:`select`)
            - ``{group: selector}``: labelled groups, for ``optimizer={group: tx}``. A selector
              is a function of the model returning its leaf or a tuple of leaves
              (``lambda m: m.func.module.E_L``, the getter of ``eqx.tree_at``), or a bool
              pytree. A selector that returns a ``dict`` of those makes one group per key, named
              by the key (``lambda m: m.theta`` for a model with a ``theta`` dict); a key that
              names another group is an error.
            - a selector on its own: the same, so one that returns a dict is a dict of groups.

        transform: bijectors the optimizer steps through, ``{name: "log"|"logit"|(lo, hi)|
            Bijector}``. Names are group labels (or entries of theta when ``params`` is not a
            dict of groups); ``"bounded"`` needs its box, ``(lo, hi)`` or
            :func:`livn.optimization.transforms.bounded`. A group takes one bijector for all its
            leaves.
        prior: ``callable(theta) -> scalar`` on the constrained theta (``{name: value}`` for
            groups), e.g. ``functools.partial(losses.param_prior, reference=..., weights=...)``.
            A prior that differs per problem takes the batch too: ``prior(theta, batch)`` when it
            has a required second positional parameter, or ``prior(theta, batch=batch)`` when it
            has a parameter named ``batch``. With ``batched`` it sees its own problem's slice.
        optimizer: an optax transformation; ``{group: tx}`` (see
            :func:`livn.optimization.train.group_chain`); or an optimistix minimiser, driven step by
            step. Default ``optax.adam(1e-2)``.
        steps: number of optimizer steps
        batched: ``theta`` (and ``data``) carry a leading axis of independent problems, each
            with its own optimizer state; the step is vmapped. History gains the problem axis.
        callbacks: :class:`livn.optimization.train.Callback` objects
        checkpoint: a :class:`~livn.optimization.train.Checkpointer`, or a directory (an
            :class:`~livn.optimization.train.EqxCheckpointer` with defaults)
        resume: continue from the checkpoint's latest step when there is one. The fit then runs
            to ``steps`` in total and its history covers the steps run in this call.
        mesh: a ``jax.sharding.Mesh``: batches are sharded on ``"batch"``, model and optimizer
            state replicated, the gradient all-reduced by XLA. Axes are forced to ``Auto``.
            Build one with :func:`livn.optimization.train.data_mesh`.
        key: PRNG key; ``model`` gets ``key=fold_in(key, step)`` (one split per problem when
            ``batched``) and must accept a ``key`` keyword
        jit: compile the step (``False`` to debug the objective)

    Returns:
        :class:`Fit`
    """
    import optax

    if isinstance(model, functools.partial):
        model = eqx.Partial(model.func, *model.args, **model.keywords)
    prior_call = _prior_call(prior)
    pass_key = key is not None
    if pass_key and not _accepts_key(model):
        raise ValueError("key= is given, but the model takes no `key` keyword")

    if mesh is not None and batched:
        raise ValueError("mesh and batched cannot be combined")
    if mesh is not None:
        mesh = _auto(mesh)

    layout = _Layout(model, params)
    static, raw = layout.split()

    if transform:
        _check_transform(raw, transform)

        def constrain(r):
            return _apply_transform(r, transform, 1)

        raw = _apply_transform(raw, transform, 0)
    else:

        def constrain(r):
            return r

    def objective(r, static, batch, key):
        theta = constrain(r)
        m = layout.assemble(static, theta)
        kwargs = {"key": key} if pass_key else {}
        output = m(**kwargs) if data is None else m(batch, **kwargs)
        result = loss(output, batch)
        value, aux = result if isinstance(result, tuple) else (result, {})
        if prior is not None:
            shown = layout.present(theta)
            if prior_call == "keyword":
                value = value + prior(shown, batch=batch)
            elif prior_call == "positional":
                value = value + prior(shown, batch)
            else:
                value = value + prior(shown)
        return jnp.asarray(value), dict(aux)

    # optimizer
    labels = None
    if layout.grouped:
        labels = {
            label: jtu.tree_map(lambda _, label=label: label, part)
            for label, part in raw.items()
        }
    if optimizer is None:
        optimizer = optax.adam(1e-2)
    if isinstance(optimizer, dict):
        if not layout.grouped or set(optimizer) != set(raw):
            raise ValueError(
                f"optimizer groups {sorted(optimizer, key=str)} must match params groups "
                f"{sorted(raw, key=str) if layout.grouped else '(none: params is not a dict)'}"
            )
        optimizer = optax.multi_transform(optimizer, lambda _: labels)

    source = _Source(data, 0, mesh)

    vmap_axes = None
    n_problems = 0
    if batched:
        n_problems = jtu.tree_leaves(raw)[0].shape[0]
        vmap_axes = (
            None,
            0 if data is not None else None,
            0 if key is not None else None,
        )

    is_optx = isinstance(optimizer, optx.AbstractMinimiser)
    if is_optx:
        if not source.fixed:
            raise ValueError(
                "an optimistix minimiser is full-batch: data must be a fixed pytree, "
                "not a stream of changing batches"
            )
        stepper = _Optimistix(optimizer, objective, jit, vmap_axes)
    else:
        stepper = _Optax(optimizer, objective, jit, vmap_axes)

    evaluate = _compile(objective, 1, vmap_axes, jit)

    if key is not None and jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)

    def step_key(step):
        if key is None:
            return None
        k = jax.random.fold_in(key, step)
        return jax.random.split(k, n_problems) if batched else k

    def theta_of(r):
        return layout.present(constrain(r))

    if isinstance(checkpoint, (str, Path)):
        checkpoint = EqxCheckpointer(checkpoint)

    context = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
    with context:
        if mesh is not None:
            raw = _replicate(mesh, raw)
            static = _replicate(mesh, static)

        # optimizer init only reads shapes; streams are not consumed for it
        opt_state = stepper.init(
            raw, (static, source.get(0) if source.fixed else None, step_key(0))
        )

        start = 0
        latest = checkpoint.latest() if checkpoint is not None and resume else None
        if checkpoint is not None and latest is not None:
            template = (raw, opt_state, np.asarray(0), key)
            raw, opt_state, restored, key = checkpoint.restore(latest, template)
            start = int(restored)
            if mesh is not None:
                raw, opt_state = _replicate(mesh, (raw, opt_state))
            source = _Source(data, start, mesh)

        history = History()
        history.batched = batched
        callbacks = [history, *callbacks]
        state = FitState(raw, opt_state, start, key, theta_of, checkpoint)
        for callback in callbacks:
            callback.on_fit_start(state)

        saved = start if start else None
        last = None
        stopped = False
        batch = source.get(start) if (start >= steps or source.fixed) else None
        for step in range(start, steps):
            batch = source.get(step)
            result = stepper.step(raw, opt_state, (static, batch, step_key(step)))
            raw, opt_state = result.raw, result.state
            state = FitState(raw, opt_state, step + 1, key, theta_of, checkpoint)

            loss_value = np.asarray(result.loss, dtype=float)
            metrics = {
                "step": step,
                "loss": loss_value if batched else float(loss_value),
                "aux": jax.device_get(result.aux),
                "raw": result.evaluated,
                "batched": batched,
                "stepped": np.asarray(result.stepped),
            }
            if result.grad_norm is not None:
                norm = np.asarray(result.grad_norm, dtype=float)
                nans = np.asarray(result.nan_grads, dtype=int)
                metrics["grad_norm"] = norm if batched else float(norm)
                metrics["nan_grads"] = nans if batched else int(nans)
            last = result

            stop = False
            for callback in callbacks:
                stop = bool(callback.on_step_end(step, state, metrics)) or stop

            if (
                checkpoint is not None
                and (step + 1) % checkpoint.every == 0
                and np.all(np.isfinite(loss_value))
            ):
                _save(checkpoint, step + 1, state)
                saved = step + 1
            stopped = stop
            if stop or stepper.done(result):
                break

        for callback in callbacks:
            replacement = callback.on_fit_end(state)
            if replacement is not None:
                state = replacement
        raw = state.raw

        if checkpoint is not None and saved != state.step and _finite(raw):
            _save(checkpoint, state.step, state)

        value, aux = evaluate(raw, args=(static, batch, step_key(state.step)))
        history.record(raw, np.asarray(value, dtype=float), jax.device_get(aux))
        if is_optx and last is not None:
            history.termination = _termination(last, batched, stopped)

    internal = constrain(raw)

    def assemble(t):
        return layout.assemble(static, t)

    fitted = None if batched else assemble(internal)
    best = next((c for c in callbacks if isinstance(c, Best)), None)
    return Fit(
        fitted,
        layout.present(internal),
        history.result(),
        state,
        best,
        assemble,
        internal,
        batched,
    )


def _termination(last, batched, stopped):
    """Why each problem of an optimistix fit ended, as the text of ``optimistix.RESULTS``."""
    finished = np.atleast_1d(np.asarray(last.finished))
    texts = []
    for i in range(finished.shape[0]):
        if finished[i]:
            code = (
                jtu.tree_map(operator.itemgetter(i), last.result)
                if batched
                else last.result
            )
            texts.append(str(code))
        elif stopped:
            texts.append("stopped by a callback")
        else:
            texts.append(str(optx.RESULTS.nonlinear_max_steps_reached))
    return texts if batched else texts[0]


def _finite(tree) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jtu.tree_leaves(tree))


def _save(checkpoint, step, state):
    checkpoint.save(step, state)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(f"fit-checkpoint-{step}")
