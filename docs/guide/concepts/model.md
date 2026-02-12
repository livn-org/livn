# Models

A **model** in livn defines the neural dynamics: how neurons behave, how synapses transmit signals, and how the system responds to stimulation. Models implement the `Model` protocol and are paired with a [backend](/guide/backends) to simulate the system.

## The `Model` protocol

All models implement the following interface:

```python
class Model(Protocol):
    def stimulus_coordinates(self, neuron_coordinates) -> array:
        """Map neuron positions to stimulation target coordinates"""

    def recording_coordinates(self, neuron_coordinates) -> array:
        """Map neuron positions to recording source coordinates"""

    def apply_defaults(self, env, weights=True, noise=True):
        """Apply default weights and noise parameters to the environment"""

    def default_weights(self, system_name, backend=None) -> dict:
        """Return tuned synaptic weights for a given system"""

    def default_noise(self, system_name, backend=None) -> dict:
        """Return tuned noise parameters for a given system"""
```

The `stimulus_coordinates` and `recording_coordinates` methods are particularly important for multi-compartment models where stimulation and recording may target different locations on the neuron (e.g., soma vs. dendrite).

## Built-in models

### Reduced Calcium Soma-Dendrite (RCSD)

The default model for the NEURON and Diffrax backends. It implements two-compartment neuron models with calcium dynamics:

- **Excitatory cells**: Booth-Rinzel-Kiehn motoneuron model with Na⁺, K⁺, Ca²⁺ channels, and calcium-dependent potassium currents
- **Inhibitory cells**: Pinsky-Rinzel interneuron model

```python
from livn.models.rcsd import ReducedCalciumSomaDendrite

model = ReducedCalciumSomaDendrite()

# Access model parameters
params_exc = model.params("BoothRinzelKiehn-MN")
params_inh = model.params("PinskyRinzel-PVBC")
```

Because this is a two-compartment model, each neuron has a soma and dendrite compartment. The `stimulus_coordinates` method returns interleaved soma/dendrite coordinates, doubling the number of stimulation targets:

```python
coords = model.stimulus_coordinates(system.neuron_coordinates)
# Shape: [2 * n_neurons, 4] - soma0, dend0, soma1, dend1, ...
```

### Izhikevich

A point-neuron model for the brian2 backend. Models excitatory and inhibitory cells with different parameter distributions following the Izhikevich formulation:

```python
from livn.models.izhikevich import Izhikevich

model = Izhikevich()
```

- **Excitatory cells**: Regular spiking with randomized recovery parameters
- **Inhibitory cells**: Fast spiking behavior

### Leaky Integrate-and-Fire (LIF)

The simplest model, also for the brian2 backend:

```python
from livn.models.lif import LIF

model = LIF()
```

## Backend-specific implementations

Models are backend-aware. When you call `model.default_weights("EI2")`, the model dispatches to the appropriate backend-specific method (e.g., `neuron_default_weights` or `brian2_default_weights`) based on the active backend.

Each model provides backend-specific hooks that the `Env` implementation calls during initialization:

| Backend | Model hooks |
|---------|-------------|
| **brian2** | `brian2_population_group()`, `brian2_connection_synapse()`, `brian2_noise_op()`, `brian2_noise_configure()` |
| **NEURON** | `neuron_celltypes()`, `neuron_synapse_mechanisms()`, `neuron_synapse_rules()`, `neuron_noise_mechanism()`, `neuron_noise_configure()` |
| **Diffrax** | `diffrax_module()` - returns an Equinox module for differentiable simulation |

## Synaptic dynamics

The RCSD model defines detailed synapse types with distinct kinetics:

| Synapse | Type | Description |
|---------|------|-------------|
| AMPA | Excitatory | Fast glutamatergic (rise ~0.1ms, decay ~2ms) |
| NMDA | Excitatory | Slow, voltage-dependent glutamatergic |
| GABA_A | Inhibitory | Fast GABAergic |
| GABA_B | Inhibitory | Slow GABAergic |

Synaptic weights are specified per projection (pre-population → post-population) and per synapse type:

```python
weights = {
    "EXC_EXC-hillock-AMPA-weight": 0.001,
    "EXC_INH-hillock-AMPA-weight": 2.9,
    "INH_EXC-soma-GABA_A-weight": 9.4,
}
env.set_weights(weights)
```

## Background noise

Biological neural networks exhibit spontaneous activity driven by background synaptic input. livn models this through noise mechanisms:

```python
noise_params = {
    "g_e0": 1.0,       # mean excitatory conductance
    "g_i0": 1.2,       # mean inhibitory conductance
    "std_e": 0.33,      # excitatory conductance std
    "std_i": 0.36,      # inhibitory conductance std
    "tau_e": 33.0,      # excitatory time constant (ms)
    "tau_i": 28.5,      # inhibitory time constant (ms)
}
env.set_noise(noise_params)
```

## Custom models

To implement a custom model, create a class that satisfies the `Model` protocol and implement the necessary backend hooks. For example, a minimal brian2 model:

```python
from livn.types import Model
import brian2 as b2

class MyModel(Model):
    def brian2_population_group(self, name, n, offset, coordinates, prng):
        return b2.NeuronGroup(
            n,
            f"dv/dt = (-v + stim(t, i + {offset})) / tau : volt",
            threshold="v > -55*mV",
            reset="v = -70*mV",
            method="euler",
            name=name,
            namespace={"tau": 10 * b2.ms},
        )

    def brian2_connection_synapse(self, pre, post):
        return b2.Synapses(pre, post, "w : 1", on_pre="v += w*mV")
```
