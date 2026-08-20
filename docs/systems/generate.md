# Generating Systems

livn supports two approaches for generating custom neural systems: **2D flat cultures** for rapid experimentation and **3D morphological networks** for biophysically detailed models.

::: tip Prerequisites
This section assumes familiarity with livn's core concepts - in particular [Systems](/guide/concepts/system), [Models](/guide/concepts/model), and [IO](/guide/concepts/io). The `systems/` subpackage is a separate workspace component that requires additional dependencies (`uv sync --package systems`).
:::

The system generation tools are available via the `livn systems` CLI as well as a Python API. Both are powered by the [machinable](https://machinable.org) framework, which handles configuration, execution, and result storage behind the scenes.

## 2D flat cultures

The `generate_2d` component creates systems where neurons are placed on a flat surface and wired with a distance-dependent kernel. This is the easiest way to create a custom system and is well suited for modelling dissociated cultures on MEAs.

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

This creates an HDF5-based system directory at `./my_system` — `cells.h5`, `connections.h5`, a `graph.json` describing the architecture, and a `provenance.json` recording the full configuration it was generated from.

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

Every population must also appear in `population_definitions`, which assigns it the integer type id stored in the H5 file (generation refuses a population that is missing one):

```python
"population_definitions": {"EXC": 10, "INH": 11}
```

Two further per-population options decide what a projection out of it looks like:

| Option | Default | Effect |
|--------|---------|--------|
| `transmitter` | from `synapse_type` | `glutamatergic`, `cholinergic`, `gabaergic` or `glycinergic` — picks the receptor, its kinetics, and the number of release sites |
| `soma_only` | true for inhibitory populations | Single-compartment cells, which can only receive on the soma |

The transmitter sets the mechanism a projection uses and its default weight (the release-site count):

| Transmitter | Mechanisms | Sites |
|-------------|-----------|-------|
| `glutamatergic` | AMPA (tau_decay 3 ms) + NMDA (80 ms) | 1 |
| `cholinergic` | AMPA (tau_decay 7 ms) | 7 |
| `gabaergic` | GABA_A (E = -60 mV, 6 ms) | 1 |
| `glycinergic` | GABA_A (E = -70 mV, 5 ms) | 5 |

Where a projection lands follows from the two. An inhibitory or glycinergic input, or any input onto a `soma_only` population, goes to the `soma` while everything else goes to `dend`.

`systems.naming` turns a ratio into the [name a culture of that composition carries](/guide/concepts/system#reading-a-name), and back, so a generated system can follow the same convention as the shipped ones:

```python
from systems.naming import name_for, ratios

name_for(3)             # 'E3I' -- three parts excitatory to one inhibitory
ratios("E3I")           # {'EXC': 0.75, 'INH': 0.25}
```

#### Connectivity

Connection probability falls off with inter-neuron distance. `kernel` picks the shape:

```
exponential (default)   P ∝ exp(-d / σ)
gaussian                P ∝ exp(-d² / (2σ²))
```

The exponential kernel is heavy-tailed, matching the long reach of axons in a
free-growing 2D culture; the Gaussian is the more tissue-like, local option.

```python
"connectivity": {
    "kernel": "exponential",    # or "gaussian"
    "sigma": 600.0,             # length constant in µm (lambda, or Gaussian width)
    "mean_degree": 20.0,        # expected incoming connections per neuron
    "cutoff": None,             # optional probability floor in [0, 1]
    "allow_self_connections": False,
}
```

`mean_degree` fixes the expected in-degree by normalising the kernel amplitude, so `sigma` controls only how far each neuron reaches, not how many partners it ends up with. `cutoff` is a probability threshold where edges whose connection probability falls below it are discarded, and `None` keeps the full tail.

The `mean_degree` can also be specified per projection, with `default` covering the pairs not listed:

```python
"connectivity": {
    "mean_degree": {
        "EXC->INH": 4.0,
        "INH->EXC": 40.0,
        "default": 0.0,
    },
}
```

A projection with degree `0` is not created at all, so `default: 0.0`. The default config lists `EXC->EXC` (20), `EXC->INH` (4) and `INH->EXC` (40), no `INH->INH` synapses.

::: warning
Configuration is merged with the defaults, not replaced by them, and that applies inside `mean_degree` too. Passing `{"EXC->EXC": 8.0, "default": 0.0}` leaves the default `EXC->INH` and `INH->EXC` entries in place, so the culture still gets both. Set a pathway to `0.0` explicitly to remove it, and read `provenance.json` in the output directory to double check.
:::

#### Depth and seed

`z_range` (default `(0, 10)` µm) is the slab cells are scattered through, and `random_seed` is the draw.

### MEA generation

Cultures ship without an array, because which electrodes a culture is read out through belongs to the experiment. After generating a system you can write one into the directory, where `MEA.from_directory` (and `Env`'s default IO) finds it:

```sh
livn systems generate_2d output_directory=./my_system --mea
```

Or in Python:

```python
generated.mea(pitch=1000)  # creates mea.json in the output directory
```

`pitch` is the inter-electrode spacing in micrometers; the grid is the largest centred power-of-two array that fits the culture area. `input_radius` (default 50 µm) and `output_radius` (default 80 µm) bound which cells an electrode can stimulate and record from, and `coordinates=[[id, x, y, z], ...]` places the electrodes explicitly instead.

For optical stimulation, `generated.lightarray()` writes a `lightarray.json` the same way. A single fibre above the centre of the culture by default. See [optical stimulation](/guide/advanced/optical-stimulation).

### Visualization

Visualize the generated system in Python:

```python
generated.plot(
    max_edges=5000,   # None draws every edge
    sample=None,      # or a cell count, to thin the scatter
    mea=True,         # overlay mea.json if the directory has one
    filename="system.png",
)
```

## 3D morphological networks

For biophysically detailed simulations with realistic neuron morphologies, use the `generate` component. This wraps the [MiV-Simulator](https://github.com/GazzolaLab/MiV-Simulator) network generation and requires the NEURON backend.

The 3D morphological systems (S1–S4, CA1) are organized in layered architectures mimicking hippocampal organization, with multi-compartment neuron models and biologically detailed morphologies. They are the 3D counterparts of the 2D cultures.

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