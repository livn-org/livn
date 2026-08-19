# Fit

`optimization.fit.fit` differentiates a [loss](/optimization/losses) on `env.run` output all the way back to the cell parameters and steps them. It works for any differentiable livn model.

```python
from optimization.fit import fit
from optimization.losses import voltage_mse

theta, history = fit(
    env,
    target_voltage,
    lambda run, target: voltage_mse(run.voltage, target),
    {"E_L": -70.0, "tau_m": 8.0},
    duration=200.0,
    stimulus=stimulus,
    dt=0.1,
    steps=200,
    learning_rate=0.5,
)
```

See the [worked example](/examples/fitting) for a runnable version.

## Batching

Batching is **per-cell parameter arrays inside one `Env(N)`**, not a `vmap` over N separate
environments. Every cell parameter is already an `(n_cells,)` array:

```python
fit(env, target, loss, {"E_L": np.full(env.num_cells, -70.0)}, ...)
```

fits N distinct parameter sets in a single simulation per step. A scalar `init` value applies to
every cell and fits one shared parameter instead.

For the fit to recover N *different* answers, the loss must **sum** over cells rather than average
into one scalar target — a sum keeps each cell's gradient independent of the others. (In an
unconnected env, that is; with recurrent weights the cells are genuinely coupled and the fit is
joint.)

## Arguments

| argument | meaning |
|---|---|
| `env` | a differentiable env, already `init()`-ed, with the channels the loss needs enabled |
| `target` | whatever `loss` compares against — traces, spike times, a dict of both |
| `loss` | `loss(run, target) -> scalar`, called with each iteration's `Run` |
| `init` | `{name: value}` starting parameters; scalar or `(n_cells,)` |
| `duration`, `stimulus`, `dt` | the run to repeat each step; `dt` is the [recording grid](/optimization/losses#the-grid-contract) the target and loss must agree on |
| `optimizer`, `learning_rate`, `steps` | any optax optimizer; defaults to `adam(learning_rate)`. Line-search optimizers (`optax.lbfgs`, `scale_by_backtracking_linesearch`) work unchanged |
| `transform` | optimize in an unconstrained space — see [Transforms](#transforms). `True` uses the defaults, a `{name: bijector}` dict overrides |
| `prior`, `prior_weight`, `prior_weights` | the [`param_prior`](/optimization/losses#param-prior) term |
| `run_kwargs` | extra `env.run` arguments |
| `callback` | `callback(step, theta, value)` after each step |
| `jit` | compile the value-and-gradient step; set `False` to debug the inner objective |

## Return value

`(theta, history)`. `history["loss"]` has `steps + 1` entries — one per step plus a final evaluation at the returned `theta`, so the last entry is the loss of what you get back. `history["params"][name]` tracks each parameter over the same points, which is what you plot to see whether a fit converged or is still moving.

## Transforms

`fit` optimizes raw parameter values by default, and that is often wrong, say, if the value can only be positive. 

Pass `transform=True` to fit in a space where the constraints cannot be violated:

```python
theta, history = fit(env, target, loss, {"tau_m": 8.0, "V_threshold_base": 25.0},
                     duration=300.0, stimulus=stimulus, dt=0.05, transform=True)
```

By default, this is using:

| bijector | for | example |
|---|---|---|
| `log` | strictly positive, no natural ceiling | `sigma`, `alpha`, `b_v` |
| `logit` | fitted inside `[0, 1]` | `f_v`, the voltage-reset multiplier |
| `bounded` | positive and capped, or signed and capped | `tau_m`, `g_L`, `t_ref`, `E_L`, `V_threshold_base` |
| `identity` | genuinely unconstrained | anything unclassified |

`bounded` is a logit over a box from `optimization.transforms.BOUNDS`, e.g. `V_threshold_base: (1.0, 150.0)` mV. Override either per call:

```python
fit(..., transform={"tau_m": "log", "V_threshold_base": "identity"})
```
