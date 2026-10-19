# Fit

`livn.optimization.fit.fit` differentiates a [loss](/optimization/losses) through a forward computation back to whatever you chose to train, and steps it. The forward computation is the model itself expressed as a callable pytree whose leaves are trainable.

```python
import functools
import optax

from livn.optimization.fit import fit
from livn.optimization.losses import voltage_mse

env = env.cells.set_params({"E_L": -70.0, "tau_m": 8.0})  # the starting values

result = fit(
    functools.partial(env, 200, stimulus, dt=0.1),          # env(200, stimulus, dt=0.1)
    lambda run, _: voltage_mse(run.voltage, target_voltage),
    params={
        "E_L": lambda m: m.func.module.E_L,
        "tau_m": lambda m: m.func.module.tau_m,
    },
    optimizer=optax.adam(0.5),
    steps=200,
)
result.theta        # {"E_L": ..., "tau_m": ...}
result.model        # the partial; result.model.func is the env with those parameters (yours is untouched)
result.history      # numpy: "loss", "params", "aux", "grad_norm", "nan_grads", "steps"
```

Each step computes `output = model(batch)` (`model()` without `data`) and `loss(output, batch)`. A `functools.partial` is converted to `equinox.Partial` automatically, so `functools.partial(env, decoding, encoding=encoding)` builds a differentiable forward map with livn's native call, `Env.__call__(decoding, inputs=batch, encoding=...)`.

See the [worked example](/examples/fitting) for a runnable version.

## Arguments

| argument | meaning |
|---|---|
| `model` | the callable pytree being fitted, e.g. a partial of an `Env`, an inverse model, an IO stack. It holds the trainable leaves |
| `loss` | `loss(output, batch) -> scalar` or `(scalar, aux_dict)`; `output` is what `model` returned. `aux` reaches callbacks and the history |
| `data` | `None`; a fixed pytree (used every step); a `callable(step)`; an object with `iterate(start_step, process, processes)` (step-addressed batches, e.g. the `Windows` of `livn.optimization.data.windows`, whose batches are `Window` conditions: it yields the batches of steps `start_step`, `start_step + 1`, ... as pure functions of `(seed, step)`, sliced per process under a mesh); or any other iterable of batches |
| `params` | see [Parameters](#parameters) |
| `transform` | see [Transforms](#transforms) |
| `prior` | `callable(theta) -> scalar` on the constrained theta, e.g. `functools.partial(losses.param_prior, reference=..., weights=...)`. A prior that differs per problem also takes the batch, see [Priors](#priors) |
| `optimizer` | an optax transformation, `{group: tx}`, or an optimistix minimizer. Default `adam(1e-2)` |
| `steps` | optimizer steps |
| `batched` | see [Batching](#batching) |
| `callbacks` | `livn.optimization.train.Callback` objects |
| `checkpoint`, `resume` | a `Checkpointer` or a directory; resume from its latest step |
| `mesh` | see [Meshes](#meshes-and-multiple-nodes) |
| `key` | PRNG key; the model gets `key=fold_in(key, step)` and must take a `key` keyword |
| `jit` | compile the step; `False` to debug the objective |

`Fit` carries `.model`, `.theta` (constrained; `{group: value}` for groups), `.history`, `.state` and `.best` (the `Best` callback you passed, else `None`), and for batched fits `.problem(i)`. See [History](#history) for what `.history` holds.

## Windows of a recording

`livn.optimization.data.windows(recording, context=..., batch_size=...)` returns `Windows` which are step-addressed batches for `data=`, each a `Window`. 

A `Window` provides `window.context` `(B, T, C)`, `window.prediction` (with `prediction > 0`), `window.spikes` (with `spikes=True`), `window.inputs` (with `inputs=True`), `window.channel_mask`, `window.recording`, `window.segment` and `window.start`:

```python
def loss(model, window, key):
    out = model.run(window)                     # membrane currents of the window
    pred = env.potential_recording(model.recording, window.replace(membrane_currents_uA=out), key)
    return mse(pred, window.context)
```

## Parameters

`params` denotes the trainable variables and how they are written back into `model`:

| `params` | trains |
|---|---|
| `None` | every inexact leaf of `model`, Python floats included (`livn.io` media keep their physical parameters as floats). Python floats among a partial's arguments count |
| a predicate, or a bool pytree | selecting leaves, e.g. `select(model, lambda m: m.medium.electrical.sigma_s_per_m)` builds the pytree for a getter |
| `{group: selector}` | labelled groups, for per-group optimizers |
| a selector, or `{group: selector}` where a selector returns a `dict` | one group per key, named by the key |

A selector is a function of the model returning its leaf, or a tuple (or module) of leaves. One that returns a `dict` of those makes a group of each entry, named by its key: with `params=lambda m: m.theta` and `m.theta = {"E_L": ..., "tau_m": ...}` the groups are `"E_L"` and `"tau_m"`, which is what `transform`, `prior` and `optimizer={group: tx}` are keyed by, and `result.theta` is `{"E_L": ..., "tau_m": ...}`. A key that is also the name of another group is an error. (A bare function is a selector when it can be applied to the model and returns its leaves, and a predicate on leaves otherwise.)

Freezing a component is leaving it out.

```python
from livn.optimization.train import group_chain

fit(
    model, loss, data=batch,
    params={"drive": lambda m: m.drive, "medium": lambda m: m.io.medium},
    optimizer={"drive": group_chain(1e-2, warmup=100, clip_norm=1.0),
               "medium": group_chain(1e-3, trust_region=0.05)},
)
```

## Transforms

`fit` optimizes raw values by default, and that is often wrong, say, if the value can only be positive. 

```python
fit(...,
    params={"tau_m": lambda m: m.func.module.tau_m, "g_L": lambda m: m.func.module.g_L},
    transform={"tau_m": "log", "g_L": (0.05, 500.0)})
```

| bijector | for |
|---|---|
| `"log"` | strictly positive |
| `"logit"` | inside `[0, 1]` |
| `(lo, hi)` or `bounded(lo, hi)` | inside a box |
| `"identity"`, or any `Bijector(fwd, inv)` | anything else |

## Priors

`prior(theta)` is added to the loss and sees the constrained theta (a `dict` of groups for grouped `params`). When the prior depends on the problem, say each cell's own reference, it takes the batch as well: `prior(theta, batch)` when its signature has a required second positional parameter, or `prior(theta, batch=batch)` when it has a parameter named `batch`. Under `batched=True` it sees its own problem's slice. A second parameter with a default (the `weights` of `functools.partial(param_prior, reference)`) is not the batch.

```python
fit(..., data={"stimulus": ..., "ref": starts},          # starts: one row per problem
    prior=lambda theta, batch: losses.param_prior(theta, batch["ref"]),
    batched=True)
```

## Optimizers

An optax transformation runs as usual, line-search optimizers (`optax.lbfgs`) included. An optimistix minimiser (`optx.BFGS(rtol=..., atol=...)`) is driven one step at a time, so callbacks, `Best` and checkpoints work exactly the same; it is full-batch, so `data` must be fixed. `fit` stops when the minimizer reports convergence.

An optimistix fit evaluates one point per step, and that point may be a line-search trial the solver then rejects. The step's `loss` and `raw` are those of the evaluated point, so `History` is a log of every evaluation and `Best` sees the trials a solver tried and discarded; `Fit.theta` is where the solver ended. A problem that has terminated stops being stepped (`batched` solvers each run to their own convergence test while the others go on) and repeats the point it stopped on.

## Batching

`batched=True` fits independent problems at once with the trained leaves of `model` (and `data`) carring a leading problem axis and the step being vmapped, each problem with its own optimizer state. It equals fitting them one after another, and `history` gains the leading axis. The rest of the model is shared; what differs per problem comes from the trained leaves and `data`. `Fit.model` is `None` for a batched fit; `Fit.model_at(i)` assembles one problem's.

`livn.optimization.fit.stack(models, params=None)` builds the model of a batched fit from one model per problem: the trained leaves (`params` as for `fit`) are stacked on a new leading axis, and everything else must be equal across the models, which is checked. `Fit.problem(i)` slices one problem back out: its constrained `theta`, its `history`, its `best` (`{"loss", "step", "theta"}` when a `Best` callback was passed) and its `model`.

```python
from livn.optimization.fit import fit, stack
from livn.optimization.train import Best

model = stack([cell_model(c) for c in cells], params=lambda m: m.theta)
result = fit(model, loss, data=batches, params=lambda m: m.theta, batched=True,
             callbacks=[Best()])
for i in range(len(cells)):
    cell = result.problem(i)       # .theta, .history, .best, .model
```

## History

`Fit.history` has the same layout whatever the optimizer, as numpy; when `batched` every array has the problem axis first.

| key | holds |
|---|---|
| `loss` | the loss at each step's evaluated point, then the final evaluation: `steps + 1` entries |
| `params` | the constrained parameters at the same points |
| `aux` | the loss's aux at the same points (for optimistix, at the last accepted point) |
| `grad_norm`, `nan_grads` | per step, when the optimizer exposes the gradient: the norm over the entries that are not NaN (an infinite entry still makes it infinite), and how many entries were NaN |
| `steps` | steps taken: an `int`, or an array with one entry per problem when batched |
| `result` | optimistix only: why each problem stopped, as the text of `optimistix.RESULTS`; `"stopped by a callback"` when a callback ended the fit |

`history["loss"][:-1]` with `history["params"]` is therefore the trajectory, evaluation by evaluation; there is no separate trace to ask for.

## Callbacks and checkpoints

`livn.optimization.train` has

- `Every(n, fn)`: host evaluations on `Run`s
- `EarlyStopping(monitor, patience, min_delta, min_rel_delta, stop_below, restore_best)`: stops when `monitor` (a key of the step metrics or of the aux, or a dotted path such as `"aux.mse"`) has not improved by `min_delta` and by the fraction `min_rel_delta` of the best for `patience` steps, or at once when it is at or below `stop_below`
- `NaNGuard(on=("loss", "grad"))`: stops on a non-finite loss (`"loss"`) or gradient norm (`"grad"`; NaN gradient entries are left out of the norm, so this is an infinite one) and leaves the last good state as the checkpoint. `"nan_grad"` also stops on any NaN gradient entry; without it the steps that had one are counted in `nan_grad_steps`, as an optimizer with `optax.zero_nans` goes on through them
- `Best`: best-so-far on the host, over every evaluated point; pass one and it is `Fit.best`
- `JsonlLogger(path)` and `History`

A callback is any object with `on_fit_start(state)`, `on_step_end(step, state, metrics) -> stop?` and `on_fit_end(state)`; `metrics` has `step`, `loss`, `raw` (the evaluated point), `aux`, `grad_norm`, `nan_grads`, `stepped` and `batched`.

`EqxCheckpointer(dir, every, keep)` writes `(theta, opt_state, step, key)` atomically plus a JSON manifest. `fit(..., checkpoint=dir)` resumes from the latest checkpoint and runs to `steps` in total; pipelines are addressed by step, so they resume with no iterator state.

## Meshes and multiple nodes

`fit(mesh=...)` replicates the model and optimizer state and shards each batch on the `"batch"` axis; XLA all-reduces the gradient. Build the mesh with `livn.optimization.train.data_mesh()` (note it uses `Auto` axes while `jax.make_mesh` defaults to `Explicit`). Across nodes, call `livn.optimization.train.initialize(comm)` first (SLURM, or an mpi4py communicator); each process then supplies only its own slice of the global batch, and only process 0 writes checkpoints and logs.
