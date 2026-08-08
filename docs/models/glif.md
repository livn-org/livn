# Generalized LIF (GLIF)

```python
from livn.models.glif import GLIF

GLIF(level=5, mechanism="hard")     # full Allen GLIF5
GLIF(level=1, mechanism="hard")     # leaky integrate-and-fire
```

For fitting, a level is exactly a mask over the trainable parameter set:

```python
GLIF(level=3).trainable_params()
# ('tau_m', 'E_L', 'g_L', 'V_threshold_base', 'theta_decay_rate', 'asc_amp_1', ...)
```

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


