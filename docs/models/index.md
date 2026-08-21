# Models

livn ships with several built-in neuron models spanning different levels of biophysical detail. Each model implements the [`Model` protocol](/guide/concepts/model) and is compatible with one or more [backends](/guide/backends).

| Model | Type | Backends | Description |
|-------|------|----------|-------------|
| [RCSD](/models/rcsd) | Two-compartment | brian2, NEURON, Diffrax | Booth-Rinzel-Kiehn motoneuron & V1 Renshaw cell with calcium dynamics |
| [CA1](/models/ca1) | Multi-compartment | NEURON | Morphologically detailed hippocampal pyramidal neurons |
| [GLIF](/models/glif) | Point neuron | Diffrax, brian2 | Allen GLIF1–5, hard + escape mechanisms. Leaky integrate-and-fire is `GLIF(level=1)` |
| [Izhikevich](/models/izhikevich) | Point neuron | brian2 | Quadratic integrate-and-fire with recovery variable |

## Choosing a model

- For biophysically detailed simulations with the NEURON backend, use [RCSD](/models/rcsd) (the default) or [CA1](/models/ca1).
- For differentiable simulation with JAX/Diffrax, use [RCSD](/models/rcsd) or [GLIF](/models/glif).
- For fast prototyping with brian2, use [GLIF](/models/glif) or [Izhikevich](/models/izhikevich).

