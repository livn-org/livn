# Standard Systems

livn includes predefined systems that cover a range of scales and biological models. These systems are ready to use and come with tuned parameters and default models.

::: tip
This section assumes familiarity with the core [concepts](/guide/concepts/env). If you haven't already, read through the Concepts guide first.
:::

## Cultures

The cultures are 2D flat networks of excitatory and inhibitory neurons, built to reproduce the dynamics of in vitro preparations grown on multi-electrode arrays. They are the recommended starting point for most users.

| System | Neurons | EXC | INH | Inhibitory share | Ratio |
|--------|---------|-----|-----|------------------|-------|
| **`E`** | 2,600 | 2,600 | 0 | none | 1:0 |
| **`E5I`** | 2,600 | 2,167 | 433 | 17% | 5:1 |
| **`E3I`** | 2,600 | 1,950 | 650 | 25% | 3:1 |
| **`EI`** | 2,600 | 1,300 | 1,300 | 50% | 1:1 |

See [reading a name](/guide/concepts/system#reading-a-name).

Each culture has a replicate draw under `_b` (`EI_b`, `E3I_b`) so a result can be checked against a second sample of the same composition.

### Architecture

Every culture occupies the same 1.6 × 3.2 mm area with the same 2,600 cells, the same cell types, and the same per-projection in-degrees.

- Distance-dependent connectivity, exponential kernel with a 600 µm length constant
- `EXC→EXC` in-degree 20, `EXC→INH` 4, `INH→EXC` 40
- AMPA on excitatory targets, GABA_A on inhibitory ones

Holding the per-projection degrees fixed means each excitatory cell receives the same inhibitory convergence in every culture; what varies across the series is how concentrated that inhibition is in fewer, more divergent cells.

See [Generating 2D systems](/systems/generate) for how to create custom cultures.

### Tuned parameters

Synaptic parameters are fitted (via surrogate-assisted optimization) against measured culture recordings. See [Tuning](/systems/tuning) for details.

```python
from livn import make

# make() applies whatever the system ships for this model
env = make("EI")
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
env = make("EI")

env.record_spikes()
env.record_voltage()
it, t, iv, v, *_ = env.run(100)
```

Or individually:

```python
from livn.system import predefined, System

path = predefined("EI")
system = System(path)

print(system.num_neurons)        # 2600
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
