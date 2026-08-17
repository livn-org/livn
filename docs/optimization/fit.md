# Fit

`optimization.fit.fit` is the Adam-through-simulation loop: it differentiates a
[loss](/optimization/losses) on `env.run` output all the way back to the cell parameters and steps
them. It works for any differentiable livn model — what it needs from `env` is a functional
parameter setter and a differentiable `run`, which today means the [diffrax
backend](/guide/backends).

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

## The functional-env contract

This is the single easiest thing to get wrong:

```python
env = env.cells.set_params(theta)   # returns a NEW env; the original is unchanged
```

`set_params` does **not** mutate in place. The fit loop rebuilds the env from `theta` on every
iteration and never touches the one you passed in — which is exactly what makes the objective a pure
function of `theta`, and so differentiable at all. If you write `env.cells.set_params(theta)` and
throw the return value away, nothing happens and no error is raised.

The other thing to get right is in the loss: read spike times from `run.spikes.padded`, not
`run.spike_times`. See [the losses page](/optimization/losses#take-spike-times-from-run-spikes-padded).

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
| `optimizer`, `learning_rate`, `steps` | any optax optimizer; defaults to `adam(learning_rate)` |
| `prior`, `prior_weight`, `prior_weights` | the [`param_prior`](/optimization/losses#param-prior) term |
| `run_kwargs` | extra `env.run` arguments |
| `callback` | `callback(step, theta, value)` after each step |
| `jit` | compile the value-and-gradient step; set `False` to debug the inner objective |

## Return value

`(theta, history)`. `history["loss"]` has `steps + 1` entries — one per step plus a final evaluation
at the returned `theta`, so the last entry is the loss of what you get back.
`history["params"][name]` tracks each parameter over the same points, which is what you plot to see
whether a fit converged or is still moving.

## Bounds

There are none. Adam will happily push `tau_m` negative and produce `NaN`. If a fit diverges, either
lower the learning rate or fit a transformed parameter (`log tau_m`, say) and exponentiate inside
your loss.
