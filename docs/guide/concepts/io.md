# IO

**IO** (Input/Output) in livn models the physical interface between the neural system and the outside world. In a real experiment, this would be a multi-electrode array (MEA) or optical stimulation device. In livn, IO transformations translate between per-channel signals and per-neuron effects, bridging the gap between the neuronal level and what an experimenter controls.

## The `IO` class

The base `IO` class defines the interface for input-output transformations:

```python
class IO:
    @property
    def num_channels(self) -> int:
        """Number of IO channels (e.g., electrodes)"""

    @property
    def channel_ids(self) -> array:
        """Array of channel identifiers"""

    def cell_stimulus(self, neuron_coordinates, channel_inputs) -> array:
        """Transform per-channel inputs into per-neuron stimulation"""

    def channel_recording(self, neuron_coordinates, neuron_ids, *recordings):
        """Transform per-neuron recordings into per-channel observations"""

    def potential_recording(self, distances, membrane_currents) -> array:
        """Estimate extracellular potentials from membrane currents"""
```

The two core operations mirror the two directions of information flow:

- **Stimulation** (`cell_stimulus`): `In: R^channels → R^neurons` - maps channel-level input commands to neuron-level effects
- **Recording** (`channel_recording`): `Out: R^neurons → R^channels` - maps neuron-level activity to channel-level observations

## Multi-Electrode Array (MEA)

The `MEA` class is the default IO device, implementing a grid of electrodes that can both stimulate and record neurons based on spatial proximity:

```python
from livn.io import MEA

mea = MEA(
    electrode_coordinates=None,   # uses default 4x4 grid
    input_radius=250,             # stimulation radius in µm
    output_radius=250,            # recording radius in µm
)
```

### Electrode layout

By default, livn generates a 4×4 regular electrode grid with 1000 µm pitch:

```python
from livn.io import electrode_array_coordinates

coords = electrode_array_coordinates(
    pitch=1000,    # inter-electrode spacing in µm
    xs=4,          # columns
    ys=4,          # rows
    xoffset=500,
    yoffset=500,
    z=175,         # electrode depth
)
# Returns: [16, 4] array of [id, x, y, z]
```

Each electrode has a unique integer ID and 3D coordinates in micrometers.

### Stimulation model

When stimulating through an MEA, the effect on each neuron depends on its distance to the activated electrode. livn models this using a point-source volume conductor model:

```
v = ρ · I / (4π · r)
```

where `ρ` is the tissue resistivity, `I` is the electrode current, and `r` is the distance between the neuron and the electrode. Only neurons within `input_radius` of an electrode receive meaningful stimulation.

```python
# Apply per-channel input to the system
import numpy as np

channel_inputs = np.zeros((100, mea.num_channels))  # [timesteps, channels]
channel_inputs[:, 5] = 0.5  # stimulate channel 5

# Transform to per-neuron stimulus
cell_stim = mea.cell_stimulus(system.neuron_coordinates, channel_inputs)
# Returns: [timesteps, n_neurons]
```

### Recording model

For spike recording, the MEA assigns each neuron to the nearest electrode within `output_radius`:

```python
it, t, *_ = env.run(100)

# Map neuron-level spikes to electrode channels
cit, ct = mea.channel_recording(system.neuron_coordinates, it, t)

# cit[channel_id] → array of neuron IDs detected at this channel
# ct[channel_id]  → array of spike times detected at this channel
```

### Extracellular potential estimation

For membrane current recordings, the MEA can estimate the extracellular field potential (local field potential) at each electrode using the point-source model:

```python
distances = mea.distances(system.neuron_coordinates)
lfp = mea.potential_recording(distances, membrane_currents)
# Returns: [n_channels, timesteps] in µV
```

This is used by decodings like [`LFP`](/guide/concepts/decoding) for spectral analysis of the extracellular signal.

### Loading from a system

Predefined systems include a saved MEA configuration matched to their spatial layout:

```python
mea = MEA.from_directory("./systems/graphs/EI2")
```

### Biphasic pulse stimulation

The `Stimulus` class provides a helper for generating charge-balanced biphasic pulses, the standard stimulation waveform for MEA experiments:

```python
from livn.stimulus import Stimulus

stim = Stimulus.biphasic_pulse(
    n_channels=mea.num_channels,
    channels=[5, 6],         # electrodes to stimulate
    amplitude=1.5,           # µA
    phase_duration=0.2,      # ms per phase
    interphase_gap=0.05,     # ms gap between phases
    pulse_times=[0.0, 50.0], # pulse onset times in ms
)
```

## Custom IO

You can implement custom IO transformations by subclassing `IO`:

```python
from livn.io import IO

class OptogeneticIO(IO):
    def __init__(self, light_sources):
        self.light_sources = light_sources

    @property
    def num_channels(self):
        return len(self.light_sources)

    def cell_stimulus(self, neuron_coordinates, channel_inputs):
        # Map light source intensities to opsin-mediated currents
        ...
```

livn also integrates with the [cleo](https://cleosim.readthedocs.io/) library for modelling optogenetic stimulation with detailed laser and opsin models.
