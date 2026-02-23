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

## Recording

Before running a simulation, specify what to record. livn supports three recording modalities:

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

## Running a simulation

Use `env.run()` to advance the simulation by a given duration (in milliseconds):

```python
it, t, iv, v, im, mp = env.run(duration=100)
```

The return value is a tuple of six arrays:

| Variable | Type | Description |
|----------|------|-------------|
| `it` | `int[]` | Neuron IDs of cells that spiked |
| `t` | `float[]` | Corresponding spike times (ms) |
| `iv` | `int[]` | Neuron IDs with voltage recordings |
| `v` | `float[n_neurons, timestep]` | Voltage traces |
| `im` | `int[]` | Neuron IDs with membrane current recordings |
| `mp` | `float[n_neurons, timestep]` | Membrane current traces |

Arrays are `None` if the corresponding recording was not enabled.

### Providing stimulus

To stimulate the system, pass a [`Stimulus`](/guide/concepts/stimulus) object:

```python
from livn.stimulus import Stimulus
import numpy as np

# Direct current injection: 100 timesteps, 10 neurons
stim = Stimulus(array=np.random.randn(100, 10) * 0.1, dt=1.0)

it, t, *_ = env.run(100, stimulus=stim)
```

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

## Configuration

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

### Seed

The random seed controls noise generation and is set during construction:

```python
env = Env(system, seed=42).init()
```

### MPI parallelism

For the NEURON backend, MPI communicators can be passed for distributed simulation:

```python
from mpi4py import MPI

env = Env(system, comm=MPI.COMM_WORLD, subworld_size=4).init()
```

## Cleanup

Always close the environment when done to free resources:

```python
env.close()
```
