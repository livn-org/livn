# Simulation Environment

The `Env` class is the central object in livn. It represents a configured simulation environment that combines a [system](/guide/concepts/system) (the neural architecture), a [model](/guide/concepts/model) (the dynamics), and an [IO](/guide/concepts/io) device (the interface) into a runnable simulation.

## Creating an environment

The simplest way to create an environment is through `livn.make()`, which fetches a predefined system and initializes an environment with sensible defaults:

```python
from livn import make

env = make("EI2")
```

This downloads the EI2 system definition, selects the default model and IO for the active [backend](/guide/backends), and applies tuned parameters.

For more control, you can construct an `Env` directly:

```python
from livn.env import Env
from livn.system import predefined
from livn.io import MEA
from livn.models.rcsd import ReducedCalciumSomaDendrite

system = predefined("EI2")
model = ReducedCalciumSomaDendrite()
io = MEA()

env = Env(system, model=model, io=io, seed=42).init()
env.apply_model_defaults()
```

The `system` argument accepts a system directory (as above), any object implementing the [`System` protocol](/guide/concepts/system#the-system-protocol), or a cell count. A count is shorthand for a [`ParallelSystem`](/guide/concepts/system#parallelsystem) of that many unconnected cells, which needs no system graph on disk:

```python
env = Env(64).init()   # 64 independent cells
env = Env(1).init()    # a single cell

env = Env({"EXC": 3, "INH": 5}).init()  # or per population
```

## Recording

Before running a simulation, specify what to record. Every backend provides at least these three modalities:

```python
# Record spike times for all populations
env.record_spikes()

# Record voltage traces (membrane potential)
env.record_voltage()

# Record membrane currents (for LFP estimation)
env.record_membrane_current()
```

Each recording method accepts an optional `population` argument to restrict recording to specific cell populations:

```python
env.record_spikes("EXC")        # only excitatory cells
env.record_voltage(["EXC", "INH"])  # multiple populations
```

The set of modalities is open and the methods above are wrappers over `env.record(what, population, **kwargs)`, which dispatches on the signal name. Signal-specific options are passed as keyword arguments, so the calls above are equivalent to:

```python
env.record("spikes")
env.record("voltage", dt=0.1)
env.record("membrane_current", ["EXC", "INH"], dt=0.5)
```

Which signals an environment supports depends on its backend and on its model, and is reported by `recordable()`:

```python
env.recordable()   # ['membrane_current', 'spikes', 'voltage']
```

Recording an unknown signal raises an `AttributeError` naming the available ones, and options a signal does not accept raise a `TypeError` rather than being ignored.

### Model-defined states

A model can expose its own internal states for recording, and they appear in `recordable()` alongside the three above. [GLIF](/models/glif), for instance, carries a threshold that moves:

```python
env = Env(64, model=GLIF(level=5)).init()

env.recordable()   # [..., 'AScurrents', 'theta_s', 'theta_v', 'threshold', ...]

env.record("threshold")
run = env.run(100)

run["threshold"].values    # [n_neurons, timestep], on the run's own dt
```

::: tip Recording is what makes a signal exist
On the diffrax backend a signal that was never recorded is never sampled and allocates no buffer, so `run.voltage` is `None` until `record_voltage()` has been called. Anything you intend to read back has to be asked for first.
:::

## Running a simulation

Use `env.run()` to advance the simulation by a given duration (in milliseconds):

```python
it, t, iv, v, im, mp = env.run(duration=100)
```

The return value holds six arrays, reachable both by unpacking (as above) and by attribute:

| Variable | Attribute | Type | Description |
|----------|-----------|------|-------------|
| `it` | `spike_ids` | `int[]` | Neuron IDs of cells that spiked |
| `t` | `spike_times` | `float[]` | Corresponding spike times (ms) |
| `iv` | `voltage_ids` | `int[]` | Neuron IDs with voltage recordings |
| `v` | `voltage` | `float[n_neurons, timestep]` | Voltage traces |
| `im` | `current_ids` | `int[]` | Neuron IDs with membrane current recordings |
| `mp` | `current` | `float[n_neurons, timestep]` | Membrane current traces |

Arrays are `None` if the corresponding recording was not enabled.

Keeping the result whole gives you a `Run`, which knows its own time base and composes:

```python
run = env.run(duration=100)

run.spike_times            # same array as t above
run.voltage_dt             # the sampling interval of run.voltage

run.slice(50, 100)         # window into [50, 100) ms
run.select(gids=[1, 2, 3]) # keep only these cells
```

`select()` can also take population names, resolved against the `{name: (start, count)}` ranges the [system](/guide/concepts/system) carries:

```python
run.select(
    population="EXC",
    population_ranges=env.system.population_ranges,
)
```

`concat()` joins successive runs along the time axis, re-applying the offset so the spike times of every chunk stay on one continuous axis:

```python
run = env.run(100)
for _ in range(9):
    run = run.concat(env.run(100))   # equivalent to a single 1000 ms run
```

`merge()` is the counterpart for the id axis, joining results that cover the same window over disjoint cells. `gather()` is the [distributed](/guide/advanced/distributed) shorthand for it: it collects the per-rank runs onto the root rank and merges them, returning `None` elsewhere. Every operation returns a new `Run`, leaving the original untouched.

### Channels and the time base

A `Run` is a set of named channels. Spikes are an `Events` channel (`ids`, `times`) while voltage and current are `Series` channels (`ids`, `values` of shape `[n_ids, T]`, plus the `dt` they were sampled at). The model-defined states above are `Series` channels too, and they are reachable by name:

```python
run["voltage"].dt          # 0.1
run.channels.keys()        # dict_keys(['spikes', 'voltage', 'current', 'threshold'])
```

The three standard channels are also reachable by name:

```python
run.drop_voltage()                          # everything but the voltage trace
run.drop_current().drop_spikes()            # composes, and returns a new Run

run.drop("threshold")                       # anything else, by name
```

A sample may itself be a vector, in which case the channel's values are `[n_ids, T, ...]`. Every channel operation (`slice`, `select`, `concat`, `merge`) works the same on those.

A run also records the window it covers where `t0` is the absolute simulation time at which it starts and `duration` is its length. The arrays are stored relative to that origin, so spike times fall in `[0, duration)` and sample `k` of a series is at `t0 + k * dt`:

```python
first, second = env.run(100), env.run(100)

second.spike_times.max()   # < 100.0 - the times restart at zero every run
second.t0                  # 100.0 on backends that track simulation time
```

### Providing stimulus

To stimulate the system, pass a [`Stimulus`](/guide/concepts/stimulus) object:

```python
from livn.stimulus import Stimulus
import numpy as np

# Direct current injection: 100 timesteps, 10 neurons
stim = Stimulus(array=np.random.randn(100, 10) * 0.1, dt=1.0)

it, t, *_ = env.run(100, stimulus=stim)
```

### Optical stimulation

For optogenetic stimulation, pass an irradiance [`Stimulus`](/guide/concepts/stimulus):

```python
stim = Stimulus.from_irradiance(irradiance_array, dt=0.1)
it, t, *_ = env.run(100, stimulus=stim)
```

See the [Optical Stimulation](/guide/advanced/optical-stimulation) guide for full details.

### Continuing simulation

The simulation state persists between `run()` calls, allowing you to implement closed-loop experiments:

```python
# First 100ms: no stimulus
env.run(100)

# Next 100ms: with stimulus
env.run(100, stimulus=stim)

# Reset to t=0
env.clear()
```

## The `__call__` interface

For more structured workflows, `Env` can be called directly with an [Encoding](/guide/concepts/encoding) and [Decoding](/guide/concepts/decoding):

```python
from livn.decoding import MeanFiringRate

result = env(
    decoding=MeanFiringRate(duration=1000),
    inputs=some_features,
    encoding=some_encoding,
)
```

This is equivalent to:

1. `encoding(env, duration, inputs)` → produces a [Stimulus](/guide/concepts/stimulus)
2. `env.run(duration, stimulus)` → a `Run`
3. `decoding(signal, env)` → processed output

This pattern is used extensively in [dataset generation](/systems/sampling) and the Gymnasium RL integration.

## Parameters

An environment carries parameters at two levels. First, the network level, which is the synaptic weights and the background noise, and secondly, the cell level, which is the physical properties of the individual cells (conductances, time constants, passive properties). They are set through separate methods, and `set_params()` routes to both by key prefix.

### Weights and noise

Synaptic weights and background noise can be configured after initialization:

```python
# Apply tuned defaults for the system
env.apply_model_defaults()

# Or set explicitly
env.set_weights({"EXC_EXC-hillock-AMPA-weight": 0.5, ...})
env.set_noise({"g_e0": 1.0, "std_e": 0.3, ...})

# Combined via params dict
env.set_params({"weight-EXC_EXC-hillock-AMPA-weight": 0.5, "noise-g_e0": 1.0})
```

### Cell parameters

The simulated cells are reachable through `env.cells`, which maps a population name to its cells and a gid directly to one cell:

```python
env.cells["EXC"]        # {gid: Cell}
env.cells[7]            # the Cell with gid 7
env.cells.gids          # every gid, ascending
```

Every cell reports and accepts its physical parameters as a flat dict. Which names it exposes depends on what backs it so read them back before setting them:

```python
env.cells[7].get_params()          # {'soma.g_pas': 1e-05, 'soma.cm': 3.0, ...}
env = env.cells[7].set_params({"soma.g_pas": 3e-5})
```

On the NEURON backend a name is `"<section type>.<name>"`, where the section type is the same one weight keys select on (`soma`, `hillock`, `basal`, ...) and the name is a section attribute (`cm`, `Ra`) or a mechanism parameter under its suffixed NEURON name (`g_pas`, `gnabar_hh`). Reads report the value at the middle of that section type's first section, writes reach every segment of every section of that type.

The whole population is addressed at once by the registry's own `get_params()` / `set_params()`, which work in `env.cells.gids` order. A scalar applies to every cell, an array holds one value per cell:

```python
env.cells.gids                                    # [0, 1, 2]
env.cells.get_params()                            # {'soma.g_pas': array([1e-5, 1e-5, 1e-5]), ...}

env = env.cells.set_params({"soma.g_pas": 3e-5})            # all cells
env = env.cells.set_params({"soma.g_pas": [1e-5, 2e-5, 3e-5]})  # one each
```

The same reaches `set_params()` under the `cells-` prefix, so cell parameters can be searched alongside weights and noise by anything that drives an env through one dict:

```python
env = env.set_params({"cells-soma.g_pas": 3e-5, "noise-g_e0": 1.0})
```

::: warning Keep the returned env
Every parameter setter returns the environment to continue with. On NEURON and brian2 that is the same object, but a diffrax `Env` holds an immutable equinox module and state cannot be mutated inside a `jit` or `grad` trace, so its setters return a new env and leave the original untouched:

```python
env = env.cells.set_params(theta)     # not env.cells.set_params(theta)
```
:::

On the diffrax backend the parameters are the module's per-cell arrays, so they differentiate:

```python
import jax

def loss(theta):
    return objective(env.cells.set_params(theta).run(100.0))

gradients = jax.grad(loss)(env.cells.get_params())
```

A model only exposes what it holds as an array field of shape `[n_cells, ...]`. A parameter kept as a Python float, or captured in a closure, is a compile-time constant to JAX and carries no gradient.

Under MPI the cells are distributed over the ranks. Indexing the registry reaches only the rank's own cells (`env.cells.local_gids`), while `gids` and `get_params()` cover all of them and are collective. Every rank has to reach them. An array passed to `set_params()` is in global gid order, and each rank picks out the entries for the cells it owns.

## Seed

The random seed controls noise generation and is set during construction:

```python
env = Env(system, seed=42).init()
```

## MPI parallelism

For the NEURON backend, MPI communicators can be passed for distributed simulation:

```python
from mpi4py import MPI

env = Env(system, comm=MPI.COMM_WORLD, subworld_size=4).init()
```

::: tip 
For running many simulations in parallel across MPI workers, checkout the [Distributed Environment](/guide/advanced/distributed)
:::

## Cleanup

Always close the environment when done to free resources:

```python
env.close()
```
