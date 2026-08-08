# Spiking LIF (SLIF)

An event-driven spiking neural network model for the Diffrax backend, enabling differentiable simulation with JAX. Adapted from snnax.

```python
from livn.models.slif import SLIF

model = SLIF()
```

## Dynamics

Spike intensity follows:

```
intensity(v) = exp(β · min(v - v_th, 10)) / τ_s
```

| Parameter | Value |
|-----------|-------|
| β | 5 |
| v_th | 1 |
| v_reset | 1.2 |
| τ_s | 1 |
| α | 3×10⁻² |

## Features

- **Event-driven spike detection** with Marcus lift for precise spike time resolution
- **Optional diffusion** (Brownian motion)
- **Output**: spike times, spike neuron indices, state trajectories

SLIF runs on the shared [event loop](/guide/backends#memory-and-step-budgets), so its buffers are sized from a spike budget and its failures are loud.

## The spike budget

An event-driven solve has to preallocate where every spike costs a segment, and the number of segments must be a compile-time constant. `max_rate` is that budget, in spikes per millisecond for the network as a whole:

```python
env.run(duration, stimulus, dt=0.1, max_rate=20.0)   # 20 spikes/ms
```

The default is `1 / dt` so one spike per recording sample.

```python
solution = module(..., max_rate=5.0, throw=False)
solution.saturated   # per sample: the budget ran out before t1
solution.solver_ok   # per sample: every inner solve succeeded
```
