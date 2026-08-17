# Metrics

::: danger Hard metrics cannot be optimized against

- Hard metrics bin spike times onto a grid before comparing them. That binning has zero gradient, so a hard metric will happily return `0.0` gradients forever. They are the numbers to report.
- Soft metrics take a continuous raster or intensity and are differentiable, so they can be used as fitting terms.
:::

First, it is important to note that spikes are differentiable because livn's event solver root-finds every spike time and differentiates it exactly, including through a hard threshold and reset (see [Event-SDE formulation](/models/glif)).

```python
env = Env(1, model=GLIF.leaky_integrate_and_fire(mechanism="hard"))

def first_spike(tau):
    run = env.cells.set_params({"tau_m": tau}).run(
      duration, stimulus, dt=dt
    )
    # an exact, root-found event time
    return run.spikes.padded.times[0, 0]

jax.grad(first_spike)(jnp.asarray([12.0])) 
# >>> +0.3566, matching finite differences
```

The metric, however, has not gradient since `spikes_to_idx` computes:

```python
idx = jnp.round(spike_times_s / dt).astype(jnp.int32)
```
Quantizing a continuous time to an integer bin index is piecewise constant, so the gradient vanishes. Thus, while the exact spike-time gradients arrives at the metric, it is thrown away by the binning. So "hard metrics are not differentiable" means *not differentiable in the spike times* as a property of the metric, not of the simulator.

## `explained_variance_ratio` <Badge type="warning" text="hard" />

The Teeter et al. (2018) explained-variance ratio

## `soft_explained_variance` <Badge type="tip" text="soft" />

The same ratio over continuous traces, thus differentiable.

## `bits_per_spike` <Badge type="tip" text="soft" />

Co-smoothing bits per spike (co-BPS), the standard [Neural Latents Benchmark](https://neurallatents.github.io/) metric. Compares the model's Poisson log-likelihood against a null model a.k.a each neuron's mean firing rate, normalized by the total spike count and
expressed in bits. Higher is better; 0 means no better than the null model.

`rates` are firing rates, not log-rates, and both arguments are binned to `(batch, time,
neurons)`. `NaN` entries in `spikes` are treated as missing and dropped.
