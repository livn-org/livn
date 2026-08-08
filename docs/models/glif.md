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

