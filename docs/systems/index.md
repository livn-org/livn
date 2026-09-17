# Standard Systems

livn includes systems that cover a range of scales and biological models, ready to use and carrying their tuned parameters and default models.

::: tip
This section assumes familiarity with the core [concepts](/guide/concepts/env). If you haven't already, read through the Concepts guide first.
:::

## Cultures

The cultures are 2D flat networks of excitatory and inhibitory neurons, built to reproduce the dynamics of in vitro preparations grown on multi-electrode arrays. They are the recommended starting point for most users.

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

from livn.system import NeuroH5System, fetch

system = NeuroH5System(fetch("CA1"))   # downloads once, then reuses
```

## Loading and using systems

```python
from livn import make

env = make("EI")
env = make("runs/bursting/env.json")  # a configured env of your own

env.record_spikes()
env.record_voltage()
it, t, iv, v, *_ = env.run(100)
```

A hosted system is assembled explicitly, since fetching it goes to the network:

```python
from livn.env import Env
from livn.system import NeuroH5System, fetch

env = Env(NeuroH5System(fetch("CA1"))).init()
```

Or take the system on its own:

```python
from livn.system import predefined

system = predefined("EI")        # a Monolayer; ships with livn

print(system.num_neurons)        # 2608
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
