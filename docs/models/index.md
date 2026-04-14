# Models

livn ships with several built-in neuron models spanning different levels of biophysical detail. Each model implements the [`Model` protocol](/guide/concepts/model) and is compatible with one or more [backends](/guide/backends).

| Model | Type | Backends | Description |
|-------|------|----------|-------------|
| [RCSD](/models/rcsd) | Two-compartment | brian2, NEURON, Diffrax | Booth-Rinzel-Kiehn motoneuron & Pinsky-Rinzel interneuron with calcium dynamics |
| [CA1](/models/ca1) | Multi-compartment | NEURON | Morphologically detailed hippocampal pyramidal neurons |
| [Izhikevich](/models/izhikevich) | Point neuron | brian2 | Quadratic integrate-and-fire with recovery variable |
| [LIF](/models/lif) | Point neuron | brian2 | Leaky integrate-and-fire |
| [SLIF](/models/slif) | Point neuron | Diffrax | Event-driven spiking LIF for differentiable simulation |

## Choosing a model

- For biophysically detailed simulations with the NEURON backend, use [RCSD](/models/rcsd) (the default) or [CA1](/models/ca1).
- For differentiable simulation with JAX/Diffrax, use [RCSD](/models/rcsd) or [SLIF](/models/slif).
- For fast prototyping with brian2, use [Izhikevich](/models/izhikevich) or [LIF](/models/lif).
