# Generalized LIF (GLIF)

```python
from livn.models.glif import GLIF

GLIF(level=5, mechanism="hard")     # full Allen GLIF5
GLIF(level=1, mechanism="hard")     # leaky integrate-and-fire
GLIF(level=5, mechanism="escape")   # stochastic spike mechanism

GLIF.leaky_integrate_and_fire()     # the textbook LIF, parameters and all
```

`GLIF` is livn's point neuron. It is not one model among several but a space with two axes with a level (1–5, which parameters are in play) and a mechanism (`hard` or `escape`, how a spike is decided).

For fitting, a level is exactly a mask over the trainable parameter set:

```python
GLIF(level=3).trainable_params()
# ('tau_m', 'E_L', 'g_L', 'V_threshold_base', 'theta_decay_rate', 'asc_amp_1', ...)
```

## Units

Everything is in **mV / pA / ms**: `tau_m` and `t_ref` in ms, `E_L` and the threshold in mV, `g_L` in nS (so `I/g_L` is a voltage), decay rates in 1/ms. A stimulus reaching `env.run` on the diffrax backend is scaled by `current_scale` (nA by default) before it is added.

## Mechanisms

Both mechanisms watch the same quantity, `V − Θ`, and differ only in what they do with it.

| | `hard` | `escape` |
|---|---|---|
| a spike is | the root of `V − Θ` | the root of `S = ∫λ dt`, with `λ = exp((V − Θ)/σ)/τ_s` |
| deterministic | yes | no: `S` restarts from `log(U) − α` after every spike |
| backends | diffrax, brian2 | diffrax |
| extra parameters | — | `sigma`, `tau_s`, `alpha` |

```python
GLIF(mechanism="escape", params={"sigma": 1.0})
```

## Networks

```python
env = Env(predefined("S1"), model=GLIF(level=1)).init()
env.module.network        # (cells, cells), w[pre, post]; None when unconnected
```

Weights must be set before `init()` on the diffrax backend since the connectivity is baked into the module:

```python
env = Env(predefined("S1"), model=GLIF()).set_weights({"EXC_EXC": 400.0}).init()
```

## Noise

`sigma_v` is a membrane diffusion amplitude, available to either mechanism. The Brownian path is a structural part of the solve, so it has to be asked for:

```python
GLIF(diffusion=True, params={"sigma_v": 1.0})
env.set_noise({"sigma_v": 1.0})   # equivalently, per run
```

## Batching

`num_samples` runs independent realisations of the same stimulus and every returned array gains a leading sample axis:

```python
run = env.run(100.0, stimulus, dt=0.1, num_samples=8)

run.voltage                  # (8, cells, T)
run.spikes.padded.times      # (8, cells, k)
run.spikes.raster(0.1)       # (8, cells, T)
```

Spikes come back as a rectangle — one row per cell, `inf`-padded — because that is the shape the solver can allocate before it knows how many events there will be. A batched run has no ragged form at all, since the number of spikes differs per sample, so `run.spike_times` raises there and points you at `.padded` or `.raster()`.

## Recordable states

Beyond spikes and voltage, GLIF exposes the rest of its state to [`env.record()`](/guide/concepts/env#recording). Each lands on the run's own `dt` grid as a `Series` channel:

| signal | shape | what it is |
|---|---|---|
| `threshold` | `[n_cells, T]` | Θ(t) = Θ_inf + θ_s(t) + θ_v(t), the value the voltage is compared against |
| `theta_s` | `[n_cells, T]` | the spike-driven component θ_s, zero at levels 1 and 3 |
| `theta_v` | `[n_cells, T]` | the voltage-coupled component θ_v, level 5 only |
| `AScurrents` | `[n_cells, T, 2]` | the two after-spike currents, zero at levels 1 and 2 |

```python
env.record_voltage()
env.record("threshold")

run = env.run(100)
run["threshold"].values          # [n_cells, T]
```

```python
env.record_voltage(dt=dt)
env.record("threshold")
env.record_spikes()

run = env.run(duration, stimulus, dt=dt)
run.spikes.padded.times     # exact event times, differentiable
```
