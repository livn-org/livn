# Models

A **model** in livn defines the neural dynamics: how neurons behave, how synapses transmit signals, and how the system responds to stimulation. Models implement the `Model` protocol and are paired with a [backend](/guide/backends) to simulate the system.

For detailed documentation on the available built-in models, see the [Models reference](/models/).

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

## Backend-specific hooks

Models are backend-aware. When you call `model.default_weights("EI2")`, the model dispatches to the appropriate backend-specific method (e.g., `neuron_default_weights` or `brian2_default_weights`) based on the active backend.

Each model provides backend-specific hooks that the `Env` implementation calls during initialization:

| Backend | Model hooks |
|---------|-------------|
| **brian2** | `brian2_population_group()`, `brian2_connection_synapse()`, `brian2_noise_op()`, `brian2_noise_configure()` |
| **NEURON** | `neuron_celltypes()`, `neuron_synapse_mechanisms()`, `neuron_synapse_rules()`, `neuron_noise_mechanism()`, `neuron_noise_configure()` |
| **Diffrax** | `diffrax_module()` - returns an Equinox module for differentiable simulation |

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