# Standard Systems

livn includes predefined systems that cover a range of scales and biological models. These systems are ready to use and come with tuned parameters, default models, and MEA configurations.

::: tip
This section assumes familiarity with the core [concepts](/guide/concepts/env). If you haven't already, read through the Concepts guide first.
:::

## EXC-INH systems (EI1–EI4)

The EXC-INH systems are 2D flat cultures with excitatory and inhibitory neuronal populations at an 80/20 ratio. They are designed to reproduce the dynamics of in vitro cultures grown on multi-electrode arrays and are the recommended starting point for most users.

| System | Neurons | EXC | INH | MEA channels | Area (µm) | Typical use |
|--------|---------|-----|-----|-------------|-----------|-------------|
| **EI1** | 10 | 8 | 2 | 1 | 40 × 40 | Quick prototyping, unit tests |
| **EI2** | 100 | 80 | 20 | 16 | 400 × 400 | Development, RL experiments |
| **EI3** | 1,000 | 800 | 200 | 64 | 1000 × 1000 | Medium-scale experiments |
| **EI4** | 10,000 | 8,000 | 2,000 | 1,024 | 4000 × 4000 | Large-scale experiments |

### Architecture

Each EI system is a 2D flat culture generated with distance-dependent Gaussian connectivity. Synaptic connections include AMPA, NMDA, and GABA_A receptor types.

- **Excitatory (EXC)** neurons make up 80% of the population
- **Inhibitory (INH)** neurons make up the remaining 20%
- Connection probability decays with distance (sigma = 200 µm)

See [Generating 2D systems](/guide/systems/generate) for how to create custom 2D cultures.

### Tuned parameters

The synaptic parameters can be tuned (via surrogate-assisted optimization) to produce experimentally reported dynamics:
- Spontaneous mean firing rates around **3 Hz**
- Stimulated mean firing rates around **12 Hz**
- Branching ratio near **1.0** (critical dynamics)

See [Tuning](/guide/systems/tuning) for details.

```python
from livn import make

env = make("EI2")
# default_params() returns the tuned weights and noise
print(env.system.default_params())
```

## Hippocampal system (CA1)

The hippocampal system models the CA1 region of the rodent hippocampus, using 15 distinct cell types with biologically detailed morphologies and connectivity.

This system requires the NEURON backend with MPI and is designed for supercomputer-scale simulations.

```python
import os
os.environ["LIVN_BACKEND"] = "neuron"

from livn.system import predefined

system_path = predefined("CA1")
```

## Loading and using systems

All predefined systems can be loaded with `make()`:

```python
from livn import make

# Downloads the system on first use, caches locally
env = make("EI2")

env.record_spikes()
env.record_voltage()
it, t, iv, v, *_ = env.run(100)
```

Or individually:

```python
from livn.system import predefined, System

path = predefined("EI2")
system = System(path)

print(system.num_neurons)        # 100
print(system.populations)        # ['EXC', 'INH']
print(system.weight_names)       # tunable weight parameters
print(system.summary())          # neuron and projection counts
```

## The `systems` subpackage

The `systems/` subpackage provides tools for generating, tuning, and sampling custom systems. These tools are available via the `livn systems` CLI:

```sh
livn systems generate_2d --launch   # generate a 2D culture
livn systems tune --launch          # tune synaptic parameters
livn systems sample --launch        # generate a dataset
```

Under the hood, the CLI is powered by [machinable](https://machinable.org), a framework for reproducible computational experiments. You don't need to know much about machinable to use these tools - the CLI handles execution, configuration, and result storage automatically.

## What's next

- [Download datasets](/systems/datasets) - download datasets of the standard systems

If the predefined systems don't match your experimental setup, you can:

- [Generate systems](/systems/generate) - generate cultures with custom populations and connectivity
- [Tune systems](/systems/tuning) - optimize synaptic parameters for target dynamics
- [Generate datasets](/systems/sampling) - produce simulation datasets at scale
