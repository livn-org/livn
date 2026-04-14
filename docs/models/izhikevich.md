# Izhikevich

A point-neuron model for the brian2 backend based on the Izhikevich formulation. It captures a wide range of spiking behaviors with a two-dimensional system (membrane potential $v$ and recovery variable $u$).

```python
from livn.models.izhikevich import Izhikevich

model = Izhikevich()
```

## Cell types

- **Excitatory cells**: Regular spiking with randomized recovery parameters
- **Inhibitory cells**: Fast spiking behavior with parameter variation

## Background noise

Noise is implemented as Gaussian current injection at 1 ms intervals:

- **Excitatory population**: 5 pA amplitude
- **Inhibitory population**: 2 pA amplitude
