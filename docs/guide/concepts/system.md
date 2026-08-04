# Systems

A **system** in livn defines the physical architecture of an in vitro neural network: the neuron positions, cell populations, connectivity, and synaptic structure. It is the static substrate on which [models](/guide/concepts/model) define dynamics and [IO](/guide/concepts/io) devices interface with the outside world.

## Predefined systems

livn ships with six predefined systems hosted on [Hugging Face](https://huggingface.co/datasets/livn-org/livn) that can be loaded by name:

| Name | Neurons | Populations | Description |
|------|---------|-------------|-------------|
| EI1 | 10 | 8 EXC / 2 INH | Minimal 2D excitatory-inhibitory culture |
| EI2 | 100 | 80 EXC / 20 INH | Small-scale 2D EXC-INH culture |
| EI3 | 1,000 | 800 EXC / 200 INH | Medium-scale 2D EXC-INH culture |
| EI4 | 10,000 | 8,000 EXC / 2,000 INH | Large-scale 2D EXC-INH culture |
| CA1 | ~100,000 | 15 cell types | Hippocampal CA1 model |

The EXC-INH systems (EI1–EI4) are 2D flat cultures with an 80/20 excitatory-to-inhibitory ratio. The hippocampal system (CA1) model the CA1 region and are suitable for simulating brain slices.

### Loading a system

```python
from livn.system import predefined, make

# Download and return the path to a predefined system
system_path = predefined("EI2")

# Or use make() which returns a System object directly
system = make("EI2")
```

Systems are cached locally in `./systems/graphs/` after the first download.

## The `System` class

The `System` class provides access to all structural properties of a neural system:

```python
from livn.system import System

system = System("./systems/graphs/EI2")

# Cell populations
system.populations          # ['EXC', 'INH']
system.num_neurons          # 10
system.cells_meta_data      # CellsMetaData with population ranges

# Spatial layout
system.neuron_coordinates   # [n_neurons, 4] array of [gid, x, y, z]
system.bounding_box         # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
system.center_point         # [x, y, z] midpoint

# Connectivity
system.weight_names         # List of tunable weight parameter names
system.connectivity_matrix()  # [n, n] weight matrix
```

### Coordinates

Each neuron has a unique integer ID (GID) and a 3D position. The coordinate array has shape `[n_neurons, 4]` where each row is `[gid, x, y, z]` in micrometers:

```python
coords = system.neuron_coordinates
print(coords[0])  # [0, 125.3, 450.7, 175.0]
```

### Populations

Neurons are organized into named populations (e.g., `"EXC"`, `"INH"`). Each population has a contiguous GID range:

```python
meta = system.cells_meta_data
meta.population_names        # ['EXC', 'INH']
meta.population_ranges       # {'EXC': (0, 7), 'INH': (7, 3)}
meta.population_count("EXC") # 7
```

### Connectivity

Access synaptic projections between populations:

```python
# Iterate over post-synaptic neuron projections
for post_gid, (pre_gids, projection) in system.projections("EXC", "INH"):
    print(f"Neuron {post_gid} receives from {len(pre_gids)} EXC neurons")

# Full connectivity weight matrix
W = system.connectivity_matrix()  # shape [num_neurons, num_neurons]
```

## ParallelSystem

`ParallelSystem` describes N unconnected cells, e.g. a single cell, or N copies of one cell that never interact.

```python
from livn.env import Env

env = Env(64).init()   # 64 independent cells, no graph required
env.record_voltage()
it, t, iv, v, *_ = env.run(100)
```

Passing an `int` as the system is shorthand for constructing one, so the two lines below are equivalent:

```python
from livn.system import ParallelSystem

env = Env(64)
env = Env(ParallelSystem(64))
```

### Populations

`num_neurons` may instead be a `{population: count}` mapping, so a plain count is shorthand for a single `"EXC"` population, e.g. `3` translates to `{"EXC": 3}`:

```python
system = ParallelSystem({"EXC": 3, "INH": 5})

system.populations       # ['EXC', 'INH']
system.num_neurons       # 8
system.population_counts # {'EXC': 3, 'INH': 5}
```

GIDs are assigned contiguously in the order the populations are given, so `EXC` owns 0-2 and `INH` owns 3-7. `Env` accepts the mapping directly too:

```python
env = Env({"EXC": 3, "INH": 5}).init()
```

Because models key their cell factories by population name, every name has to be one the [model](/guide/concepts/model) defines.

::: warning
A model may handle some populations implicitly. For example, `ReducedCalciumSomaDendrite(implicit_inhibition=True)` puts `"INH"` in `ignored_populations()`, so those cells are never instantiated. Pass `ReducedCalciumSomaDendrite(implicit_inhibition=False)` to simulate them explicitly.
:::

### Coordinates

The cells are unconnected, so their geometry only reaches the simulation through the [IO](/guide/concepts/io) transforms. `coordinates` accepts three forms:

```python
ParallelSystem(64)                    # default: every cell at the origin
ParallelSystem(64, coordinates=25.0)  # 25 um apart along x, in gid order

# an explicit [n_neurons, 3] of x, y, z (or [n_neurons, 4] of gid, x, y, z)
ParallelSystem(3, coordinates=[[0, 0, 0], [10, 0, 0], [0, 10, 0]])

# or a callable taking the total cell count
ParallelSystem(64, coordinates=lambda n: rng.normal(size=(n, 3)))
```

## Storage format

Systems are stored on disk as a directory containing:

| File | Contents |
|------|----------|
| `cells.h5` (or `graph.h5`) | Neuron coordinates and synapse attributes in NeuroH5 format |
| `connections.h5` (or `graph.h5`) | Synaptic projections between populations |
| `graph.json` | System metadata (architecture, connectivity config, element provenance) |
| `mea.json` | Default IO device configuration (optional) |
| `model.json` | Default model configuration (optional) |
| `params.json` | Tuned default parameters (optional) |


## Default model and IO

Each system can specify default configurations:

```python
system = System("./systems/graphs/EI2")

model = system.default_model()       # e.g., ReducedCalciumSomaDendrite
io = system.default_io()             # e.g., MEA with stored electrode layout
params = system.default_params()     # Tuned weights and noise parameters
```

These are used automatically by `livn.make()`.

## Custom systems

For generating your own systems - including 2D flat cultures and 3D morphological networks - see the [Systems](/systems/) guide.
