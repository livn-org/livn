# Losses

Every loss is evaluated on the recording grid defined by the `dt` that is passed to `env.run`.

```python
run = env.run(duration, stimulus, dt=0.1)   # <- this dt
nll = spike_nll(log_rate, run.spike_times, dt=0.1)
```

Pick one `dt`, and use it for the run, for the target, and for the loss.

Spike times are the exception since they come out of the event root finder rather than off the grid, which is what makes them differentiable. `spike_nll` interpolates the log-intensity at each spike time instead of binning it, so parameters reach the loss through the spike times as well as through the trace.

::: danger Take spike times from `run.spikes.padded`
`run.spike_times` is the compact list with the padding dropped, ordered by time. Building it needs concrete values, so it cannot happen inside a compile-time trace; use `run.spikes.padded.times` instead:

```python
run.spike_times            # ragged, sorted  - reporting, decoding, plotting
run.spikes.padded.times    # (cells, k)      - losses, gradients
run.spikes.raster(dt)      # (cells, T) bool - binned, lossy, trace-safe
```
:::

## `spike_nll`

The point-process negative log likelihood defined as

```
NLL = ∫ λ dt − Σ_i log λ(t_i)
```

```python
from optimization.losses import log_intensity, spike_nll

run = env.run(duration, stimulus, dt=dt)
rate = log_intensity(run.voltage, run["threshold"].values, sigma, tau_s)
nll = spike_nll(rate, run.spikes.padded.times, dt)
```

It takes the log intensity, not the rate. Spikes may be given either as `(cells, k)` with one inf-padded row per cell or flat alongside `ids`.

### `log_intensity` / `intensity`

λ is never recorded but reconstructed via:

```
λ(t) = exp((V(t) − Θ(t)) / σ) / τ_s
```

so it relies on `record_voltage()` and `record("threshold")` (see [recordable states](/models/glif#recordable-states)).

## `voltage_mse`

Mean squared error between a simulated and a recorded voltage trace in mV^2.

```python
from optimization.losses import refractory_mask, voltage_mse

mask = refractory_mask(run.spikes.padded.times, times, t_ref)
mse = voltage_mse(run.voltage, target_voltage, mask)
```

`refractory_mask` returns the samples to keep, dropping `[t_spike, t_spike + t_ref)` around every spike. The mask materializes a `(cells, spikes, times)` intermediate, so keeping the spike count bounded is important for long runs.

## `param_prior`

The regularizer, `Σ ‖θ − θ₀‖^2` term:

```python
from optimization.losses import param_prior

penalty = param_prior(theta, reference, weights={"tau_m": 1.0, "E_L": 0.1})
```
