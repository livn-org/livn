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

The point-process negative log likelihood

```
NLL = ∫ λ dt − Σ_i log λ(t_i)
```

```python
from optimization.losses import log_intensity, spike_nll

sol = neurons.solve(..., record=frozenset({"voltage", "threshold", "spikes", "segments"}))
rate = log_intensity(v_at_nodes, threshold_at_nodes, sigma, tau_s)
nll = spike_nll(rate, sol.segment_ts, recorded_spike_times)
```

`log_rate` and `ts` are `(cells, nodes)` while spikes are `(cells, k)`, `inf`-padded.

### `log_intensity` / `intensity`

λ is never recorded but reconstructed via:

```
λ(t) = exp((V(t) − Θ(t)) / σ) / τ_s
```

so it relies on `record_voltage()` and `record("threshold")` (see [recordable states](/models/glif#recordable-states)).

## `spike_kernel`

Squared RKHS distance between two spike trains, computed from spike times alone.

```
d² = ΣΣ K(sᵢ,sⱼ) − 2 ΣΣ K(sᵢ,rⱼ) + ΣΣ K(rᵢ,rⱼ)
```

```python
from optimization.losses import spike_kernel

d2 = spike_kernel(run.spikes.padded.times, recorded_times, bandwidth=5.0)
```

This is `‖f − g‖²` for `f(t) = Σ K(t − sᵢ)` with a Gaussian `K`, so it is a genuine metric on spike trains and penalizes a count mismatch as well as a timing one.

Prefer `spike_nll` when fitting since it integrates the intensity rather than counting realized events, so the count discontinuity does not exist rather than being mitigated.

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
