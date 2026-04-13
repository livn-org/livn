# Stimulus

A **Stimulus** is the input signal delivered to neurons during simulation. It represents the time-varying stimulation pattern - whether electrical current, conductance changes, or MEA-mediated channel inputs - that drives neural activity.

## The `Stimulus` class

The `Stimulus` class wraps a 2D array of stimulation values along with timing metadata:

```python
from livn.stimulus import Stimulus
import numpy as np

stim = Stimulus(
    array=np.random.randn(100, 10) * 0.1,  # [timesteps, n_targets]
    dt=1.0,                                  # timestep in ms
    gids=np.arange(10),                      # target neuron IDs (optional)
)

print(stim.duration)  # 100.0 ms
print(len(stim))      # 10 targets
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `array` | `float[T, N]` | Stimulation values over time |
| `dt` | `float` | Timestep in milliseconds |
| `gids` | `int[N]` | Target neuron IDs (optional) |
| `duration` | `float` | Total duration in ms (`T × dt`) |
| `meta_data` | `dict` | Additional metadata |

## Creating stimuli

### From arrays

Pass a NumPy array directly:

```python
# Current injection: 200ms at 0.05ms resolution, 10 neurons
stim = Stimulus(array=np.zeros((4000, 10)), dt=0.05)
```

### From channel inputs

When using an [MEA](/guide/concepts/io), you typically specify inputs per electrode channel rather than per neuron. The MEA's `cell_stimulus` method transforms channel inputs into per-neuron stimulation:

```python
from livn.io import MEA

mea = MEA()
channel_inputs = np.zeros((100, mea.num_channels))
channel_inputs[:, 5] = 0.5  # stimulate channel 5

cell_stim = mea.cell_stimulus(system.neuron_coordinates, channel_inputs)
stim = Stimulus(array=cell_stim, dt=1.0)
```

### Biphasic pulses

For MEA-style experiments, create charge-balanced biphasic electrical pulses:

```python
stim = Stimulus.biphasic_pulse(
    n_channels=16,
    channels=[5, 6],          # channels to stimulate
    amplitude=1.5,            # µA
    phase_duration=0.2,       # ms per phase
    interphase_gap=0.05,      # ms between phases
    pulse_times=[0.0, 50.0],  # onset times
    dt=0.05,                  # timestep resolution
    cathodic_first=True,      # cathodic phase first (standard)
)
```

### From conductance values

For synaptic conductance-based inputs:

```python
stim = Stimulus.from_conductance(
    conductance=np.random.rand(100, 10) * 0.01,  # µS
    dt=0.1,
)
```

### From current injection

For direct current injection:

```python
stim = Stimulus.from_current(
    current=np.ones((100, 10)) * 0.5,  # nA
    dt=0.1,
)
```

### From current density

For current density injection:

```python
stim = Stimulus.from_current_density(
    current_density=np.ones((100, 10)) * 0.1,  # mA/cm²
    dt=0.1,
)
```

### From extracellular voltage

For explicit extracellular voltage stimulation:

```python
stim = Stimulus.from_extracellular(
    voltage=cell_stim_array,  # mV, [timesteps, n_gids]
    dt=0.1,
)
```

### From irradiance (optical stimulation)

For optical stimulation via opsin-expressing neurons. The opsin model is part of the neuron model and does not need to be specified on the stimulus:

```python
stim = Stimulus.from_irradiance(
    irradiance=light_at_neurons,  # mW/mm^2, [timesteps, n_gids]
    dt=0.1,
)
```

## Stimulus types

Each stimulus carries an `input_mode` that tells the backend what physical quantity the array represents:

| Type | Factory | Units | Description |
|------|---------|-------|-------------|
| `extracellular` | `Stimulus()`, `from_extracellular()` | mV | Extracellular voltage (MEA default) |
| `current` | `from_current()` | nA | Direct intracellular current injection |
| `current_density` | `from_current_density()` | mA/cm² | Current density normalized to membrane area |
| `conductance` | `from_conductance()` | µS | Synaptic conductance (requires model `E_rev`) |
| `irradiance` | `from_irradiance()` | mW/mm² | Light intensity for optogenetic stimulation |

Access the mode via `stim.input_mode`:

```python
stim = Stimulus.from_current(np.ones((100, 10)), dt=0.1)
print(stim.input_mode)  # "current"
```

## Using stimuli

Pass a stimulus to `env.run()`:

```python
it, t, *_ = env.run(duration=100, stimulus=stim)
```

Or use it with the [encoding](/guide/concepts/encoding) interface for structured input pipelines:

```python
result = env(
    decoding=MeanFiringRate(duration=100),
    inputs=features,
    encoding=my_encoding,
)
```

## Automatic conversion

`Stimulus.from_arg()` automatically converts common types into a single `Stimulus`:

```python
# From array
stim = Stimulus.from_arg(np.zeros((100, 10)))

# From tuple (array, dt)
stim = Stimulus.from_arg((np.zeros((100, 10)), 0.05))

# From dict
stim = Stimulus.from_arg({"array": np.zeros((100, 10)), "dt": 0.05})

# Pass through
stim = Stimulus.from_arg(existing_stimulus)

# None = no stimulus
stim = Stimulus.from_arg(None)
```
