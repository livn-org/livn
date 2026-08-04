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

Which signals an environment supports depends on its backend and is reported by `recordable()`:

```python
env.recordable()   # ['membrane_current', 'spikes', 'voltage']
```

Recording an unknown signal raises an `AttributeError` naming the available ones, and options a signal does not accept raise a `TypeError` rather than being ignored.

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

A `Run` is a set of named channels. Spikes are an `Events` channel (`ids`, `times`) while voltage and current are `Series` channels (`ids`, `values` of shape `[n_ids, T]`, plus the `dt` they were sampled at). Models may record further channels, and they are reachable by name:

```python
run["voltage"].dt          # 0.1
run.channels.keys()        # dict_keys(['spikes', 'voltage', 'current'])
```

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
2. `env.run(duration, stimulus)` → raw recordings
3. `decoding(env, *recordings)` → processed output

This pattern is used extensively in [dataset generation](/systems/sampling) and the Gymnasium RL integration.

## Weights and noise

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
