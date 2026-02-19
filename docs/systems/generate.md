# Generating Systems

livn supports two approaches for generating custom neural systems: **2D flat cultures** for rapid experimentation and **3D morphological networks** for biophysically detailed models.

::: tip Prerequisites
This section assumes familiarity with livn's core concepts - in particular [Systems](/guide/concepts/system), [Models](/guide/concepts/model), and [IO](/guide/concepts/io). The `systems/` subpackage is a separate workspace component that requires additional dependencies (`uv sync --package systems`).
:::

The system generation tools are available via the `livn systems` CLI as well as a Python API. Both are powered by the [machinable](https://machinable.org) framework, which handles configuration, execution, and result storage behind the scenes.

## 2D flat cultures

The `generate_2d` component creates systems where neurons are placed on a flat surface with distance-dependent Gaussian connectivity. This is the easiest way to create a custom system and is well suited for modelling dissociated cultures on MEAs.

### Quick start

Via the CLI:

```sh
livn systems generate_2d \
    total_cells=50 \
    output_directory=./my_system \
    --launch
```

Or equivalently via Python:

```python
from machinable import get

get("generate_2d", {
    "total_cells": 50,
    "populations": {
        "EXC": {"ratio": 0.8, "synapse_type": "excitatory"},
        "INH": {"ratio": 0.2, "synapse_type": "inhibitory"},
    },
    "output_directory": "./my_system",
}).launch()
```

This creates an HDF5-based system directory at `./my_system` that can be loaded directly into livn:

```python
from livn.env import Env

env = Env("./my_system").init()
env.apply_model_defaults()
```

### Configuration

The full configuration is specified via a Pydantic model:

#### Cell placement

Control how neurons are distributed in space using the `area` parameter:

```python
# Rectangular area (default)
config = {
    "area": "systems.generate_2d.rectangle",
    "area_kwargs": {
        "x_range": [0.0, 4000.0],
        "y_range": [0.0, 4000.0],
    },
    "total_cells": 100,
}

# Disk-shaped culture
config = {
    "area": "systems.generate_2d.disk",
    "area_kwargs": {
        "center": [0.0, 0.0],
        "radius": 500.0,
        "inner_radius": 0.0,   # for annular shapes
    },
    "total_cells": 100,
}
```

You can also provide a custom placement function:

```python
def my_placement(count, rng, **kwargs):
    """Return (xs, ys) arrays of neuron positions"""
    xs = rng.uniform(0, 1000, size=count)
    ys = rng.uniform(0, 1000, size=count)
    return xs, ys
```

#### Populations

Define cell populations using either a `ratio` (fraction of total cells) or an absolute `count`:

```python
"populations": {
    "EXC": {"ratio": 0.8, "synapse_type": "excitatory"},
    "INH": {"ratio": 0.2, "synapse_type": "inhibitory"},
}

# Or with explicit counts
"populations": {
    "EXC": {"count": 80, "synapse_type": "excitatory"},
    "INH": {"count": 20, "synapse_type": "inhibitory"},
}
```

#### Connectivity

Connection probability is governed by a Gaussian function of inter-neuron distance:

```
P(connect) = exp(-d² / (2σ²))
```

```python
"connectivity": {
    "sigma": 500.0,             # Gaussian width in µm
    "mean_degree": 10.0,        # average connections per neuron
    "cutoff": 3.0,              # cutoff at cutoff × sigma
    "allow_self_connections": False,
}
```

The `mean_degree` can also be specified per projection:

```python
"connectivity": {
    "mean_degree": {
        "EXC->EXC": 8.0,
        "EXC->INH": 12.0,
        "INH->EXC": 6.0,
        "INH->INH": 4.0,
    },
}
```

### MEA generation

After generating a system, create a matching MEA:

```sh
livn systems generate_2d output_directory=./my_system --mea
```

Or in Python:

```python
generated.mea(pitch=1000)  # creates mea.json in the output directory
```

The pitch parameter controls the inter-electrode spacing in micrometers.

### Visualization

Visualize the generated system in Python:

```python
generated.plot(
    show_connections=True,
    show_mea=True,
    max_edges=5000,
)
```

## 3D morphological networks

For biophysically detailed simulations with realistic neuron morphologies, use the `generate` component. This wraps the [MiV-Simulator](https://github.com/GazzolaLab/MiV-Simulator) network generation and requires the NEURON backend.

The 3D morphological systems (S1–S4, CA1) are organized in layered architectures mimicking hippocampal organization, with multi-compartment neuron models and biologically detailed morphologies. They are the 3D counterparts of the EI1–EI4 2D cultures.

### Configuration

3D systems are configured via YAML files. livn includes example configurations in `systems/config/`:

| Config | Description |
|--------|-------------|
| `S1.yml` – `S4.yml` | EXC-INH systems of increasing scale |
| `CA1.yml` | Hippocampal CA1 system |

Example YAML structure:

```yaml
Definitions: !include _definitions.yml
Cell Types: !include _cell_types.yml
Synapses: !include _synapses.yml

Geometry:
  Cell Distribution:
    EXC:
      Layer: SP
      Count: 7
    INH:
      Layer: SO, SP, SR
      Count: 3

  Axon Extent:
    EXC: 1000.0
    INH: 500.0
```

### Generation

Via the CLI:

```sh
livn systems generate config=systems/config/S1.yml --launch
livn systems generate config=systems/config/S1.yml --mea
livn systems generate config=systems/config/S1.yml --export
```

Or in Python:

```python
from machinable import get

gen = get("generate", {"config": "systems/config/S1.yml"})
gen.launch()  # runs MiV-Simulator network generation
gen.mea()     # generates electrode array
gen.export()  # copies to systems/data/
```

This requires MPI and the NEURON backend dependencies. For large systems (S3, S4, CA1), use the `mpi` or `slurm` execution module to run on HPC infrastructure:

```sh
livn systems slurm generate \
    config=systems/config/S3.yml \
    **resources='{"--nodes": 4, "--ntasks-per-node": 56, "-p": "normal", "-t": "4:00:00"}' \
    --launch
```

See the [machinable execution docs](https://machinable.org/guide/execution) for details on available execution modules.

## Output format

Both 2D and 3D generators produce a system directory with the following structure:

```
my_system/
├── cells.h5          # Neuron coordinates and synapse attributes (NeuroH5)
├── connections.h5    # Synaptic projections
├── graph.json        # System metadata and configuration
└── mea.json          # MEA electrode coordinates (optional)
```

This directory can be passed directly to `Env()` or `System()`:

```python
from livn.system import System
system = System("./my_system")
print(system.num_neurons, system.populations)
```