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
from livn.optimization.losses import log_intensity, spike_nll

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
from livn.optimization.losses import spike_kernel

d2 = spike_kernel(run.spikes.padded.times, recorded_times, bandwidth=5.0)
```

This is `‖f − g‖²` for `f(t) = Σ K(t − sᵢ)` with a Gaussian `K`, so it is a genuine metric on spike trains and penalizes a count mismatch as well as a timing one.

Prefer `spike_nll` when fitting since it integrates the intensity rather than counting realized events, so the count discontinuity does not exist rather than being mitigated.

## `voltage_mse`

Mean squared error between a simulated and a recorded voltage trace in mV^2.

```python
from livn.optimization.losses import refractory_mask, voltage_mse

mask = refractory_mask(run.spikes.padded.times, times, t_ref)
mse = voltage_mse(run.voltage, target_voltage, mask)
```

`refractory_mask` returns the samples to keep, dropping `[t_spike, t_spike + t_ref)` around every spike. The mask materializes a `(cells, spikes, times)` intermediate, so keeping the spike count bounded is important for long runs.

## `param_prior`

The regularizer, `Σ ‖θ − θ₀‖^2` term:

```python
from livn.optimization.losses import param_prior

penalty = param_prior(theta, reference, weights={"tau_m": 1.0, "E_L": 0.1}, scale=1.0)
```

## `shape_logmag`

Scale-invariant loss for any simulated-vs-recorded `(*batch, time, channel)` trace. Each trace is divided by its RMS (over time, per batch and channel), and the loss is the shape error plus the squared log RMS ratio:

```
L = mean ‖V̂/RMS(V̂) − V/RMS(V)‖² + λ · mean (log RMS(V̂) − log RMS(V))²
```

```python
from livn.optimization.losses import events_to_mask, shape_logmag

loss = shape_logmag(pred, target, mask, log_mag_weight=1.0)
shape, logmag = shape_logmag(pred, target, mask, return_terms=True)
```

## `events_to_mask`

Builds the mask for `shape_logmag` from spike times, on device and jittable:

```python
mask = events_to_mask(batch["spikes"], (T, C), dt=0.1, window_ms=1.0, channel_radius=1, channels=peak_channel)
```

`events` is the `livn.optimization.data` batch format, `(*batch, unit, spike)` in ms relative to the window start, `inf`-padded. Each spike marks `±window_ms/2` (at least ±1 sample, inclusive) on its unit's channel and `channel_radius` neighbours either side. `channels` is `(unit,)`, or `(unit, m)` padded with `-1` for units seen on several channels; by default unit `u` is on channel `u`.

## `smoothness`

High-pass energy `mean((x − lowpass(x))²) / scale²` of a `(*batch, time, unit)` trace, with a Gaussian lowpass of `sigma_ms`. It suppresses sample-scale ringing in a fitted current while leaving slow structure alone.
