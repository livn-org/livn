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
- **Interleaved spike tracking** for temporal resolution
- **Output**: spike times, spike neuron indices, state trajectories
