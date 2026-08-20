# Systems

A **system** in livn defines the physical architecture of an in vitro neural network: the neuron positions, cell populations, connectivity, and synaptic structure. It is the static substrate on which [models](/guide/concepts/model) define dynamics and [IO](/guide/concepts/io) devices interface with the outside world.

## Predefined systems

livn ships a series of 2D cultures, plus a hippocampal slice model, hosted on [Hugging Face](https://huggingface.co/datasets/livn-org/livn) and loaded by name:

| Name | Neurons | EXC / INH | Description |
|------|---------|-----------|-------------|
| `E` | 2,600 | 2600 / 0 | Excitatory only |
| `E5I` | 2,600 | 2167 / 433 | 17% inhibitory |
| `E3I` | 2,600 | 1950 / 650 | 25% inhibitory |
| `EI` | 2,600 | 1300 / 1300 | Balanced |
| `CA1` | ~10,000 | 15 cell types | Hippocampal CA1 model |

The cultures differ only in composition using the same area, same cell types, same per-projection in-degrees.

### Reading a name

A culture's name states its excitatory-to-inhibitory ratio, with inhibition fixed at 1:

| Name | Ratio | Inhibitory share |
|------|-------|------------------|
| `E` | 1:0 | none |
| `E5I` | 5:1 | 17% |
| `E3I` | 3:1 | 25% |
| `EI` | 1:1 | 50% |

The number always follows `E`, never `I`, so `E3I` is three parts excitatory to one part inhibitory.

`systems.naming` parses this convention:

```python
from systems.naming import composition_of, ratios

composition_of("E3I")   # (3.0, 1.0)
ratios("E3I")           # {'EXC': 0.75, 'INH': 0.25}
```

### Replicates

Each culture has a second draw under a `_b` suffix (e.g. `EI_b`) built from the same configuration with a different RNG seed.

### Scale

To allow running cheaply, every culture carries three nested spatial subselections, `e1`/`e2`/`e3`, cut from 250 / 500 / 1000 um boxes at its centre:

| Rung | Box | Cells (`EI`) |
|------|-----|--------------|
| `e1` | 250 um | ~30 |
| `e2` | 500 um | ~120 |
| `e3` | 1000 um | ~490 |

### Loading a system

```python
from livn.system import predefined, make

# Download and return the path to a predefined system
system_path = predefined("EI")

# Or use make() which returns a System object directly
system = make("EI")
```

Systems are cached locally in `./systems/graphs/` after the first download.

## The `System` class

The `System` class provides access to all structural properties of a neural system:

```python
from livn.system import System

system = System("./systems/graphs/EI")

# Cell populations
system.populations          # ['EXC', 'INH']
system.num_neurons          # 2600
system.population_ranges    # {'EXC': (0, 1300), 'INH': (1300, 1300)}

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
system.populations             # ['EXC', 'INH']
system.population_ranges       # {'EXC': (0, 1300), 'INH': (1300, 1300)}
system.population_count("EXC") # 1300
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
A model may declare that some populations should not be built. `ignored_populations()` returns the names backends skip when instantiating cells and connections; it is empty by default, so every population in the system is simulated.

Override it to ablate one:

```python
class ExcitatoryOnly(ReducedCalciumSomaDendrite):
    def ignored_populations(self):
        return {"INH"}
```
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
| `selection/<name>.json` | Stored subselections, as resolved cell ids (optional) |
| `params/<selection>.json` | Tuned parameters per selection, `default.json` for the whole system (optional) |


## Default model and IO

Each system can specify default configurations:

```python
system = System("./systems/graphs/EI")

model = system.default_model()       # e.g., ReducedCalciumSomaDendrite
io = system.default_io()             # e.g., MEA with stored electrode layout
```

These are used automatically by `livn.make()`.

## Subselections

`env.selection(name)` builds only part of a system, which is how the `e1`/`e2`/`e3` rungs are used:

```python
env = Env("./systems/graphs/EI")
env.selection("e2")   # ~120 cells instead of 2600
env.init()
```

A stored selection holds the resolved cell ids rather than a rule to recompute, so it names the same cells however the selection code changes. An ad-hoc selection takes a count, a fraction, or a spatial patch instead:

```python
env.selection(100)                        # 100 cells, proportional across populations
env.selection(0.25)                       # a quarter of each population
env.selection(0.25, method="patch")       # a centred region, keeping neighbours together
```

`method="patch"` matters when connectivity is distance-dependent. Thinning at random keeps an edge only where both endpoints survive, so in-degree collapses in proportion to the thinning. A contiguous patch keeps each cell's nearest partners and drops the distant ones, which are the weakest under a distance kernel.

Even so, a subselection is a different network. For example, on `EI`, the rungs retain 3%, 11% and 37% of each cell's in-degree, so parameters fitted on the whole system do not describe a rung and vice versa. That is why each carries its own parameter file.

## Tuned parameters

A system ships its tuned parameters under `params/`, keyed by the model and by which selection is in force. Applying them is a method on the environment, because the choice depends on what was actually built:

```python
env = Env("./systems/graphs/EI").init()
env.apply_default_params()                  # the whole system
```

```python
env = Env("./systems/graphs/E")
env.selection("e3")                          # a stored subselection
env.init()
env.apply_default_params()                   # -> params/e3.json
```

Each file holds one block per model and one named group per promoted solution:

```json
{
  "ReducedCalciumSomaDendrite": {
    "default": {"params": {"EXC_EXC-hillock-AMPA-weight": 0.31, "noise-g_e0": 1.0},
                "meta": {"loc": 7, "retained_in_degree": 0.624}}
  }
}
```

Pass `group=` to pick a different one. When a system ships nothing, the model's own built-in defaults apply instead.

## Custom systems

For generating your own systems - including 2D flat cultures and 3D morphological networks - see the [Systems](/systems/) guide.
