# Leaky Integrate-and-Fire (LIF)

The simplest model, for the brian2 backend. A single-compartment leaky integrator with a hard threshold and reset.

```python
from livn.models.lif import LIF

model = LIF()
```

## Dynamics

```
τ dv/dt = (v_rest - v) + Rm * (I + I_noise) + stim
```

| Parameter | Value |
|-----------|-------|
| Resting potential (v_rest) | -70 mV |
| Threshold | -55 mV |
| Reset | -75 mV |
| Membrane time constant (τ) | 10 ms |
| Membrane resistance (Rm) | 100 MΩ |

## Background noise

Gaussian noise with configurable amplitude per population.
