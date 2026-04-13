# IO

**IO** (Input/Output) in livn models the physical interface between the neural system and the outside world, whether through a multi-electrode array (MEA) for electrical stimulation, a fiber optic array for optical stimulation, or a combination of both. IO transformations translate between per-channel signals and per-neuron effects, bridging the gap between the neuronal level and what an experimenter controls.

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

class MyCustomIO(IO):
    def __init__(self, sources):
        self.sources = sources

    @property
    def num_channels(self):
        return len(self.sources)

    def cell_stimulus(self, neuron_coordinates, channel_inputs):
        # Map source inputs to per-neuron effects
        ...
```

## Light Array (Fiber Optic)

The `LightArray` class models a fiber optic array for optical stimulation. It maps per-fiber light power (mW) to per-neuron irradiance (mW/mm^2) using a Kubelka-Munk scattering propagation model:

```python
from livn.io import LightArray
import numpy as np

fibers = LightArray(
    fiber_coordinates=np.array([
        [0, 500, 500, 0],    # fiber 0 at (500, 500, 0)
        [1, 1500, 500, 0],   # fiber 1 at (1500, 500, 0)
    ]),
    numerical_aperture=0.37,
    fiber_radius_um=100.0,
    wavelength_nm=473.0,
    scattering_coefficient=11.2,  # mm⁻¹, brain tissue at 473nm
)
```

`cell_stimulus()` returns a `Stimulus` object with `input_mode='irradiance'`:

```python
# Per-fiber power trace: 100 timesteps, 2 fibers
fiber_power = np.zeros((100, 2))
fiber_power[10:50, 0] = 5.0  # 5 mW on fiber 0

stim = fibers.cell_stimulus(system.neuron_coordinates, fiber_power, dt=0.1)
# stim is a Stimulus with input_mode='irradiance'
env.run(100, stimulus=stim)
```

### Light propagation model

The transmittance from each fiber to each neuron is computed using the Kubelka-Munk model (Aravanis et al. 2007), accounting for:

- Fiber numerical aperture and cone geometry
- Tissue scattering at the specified wavelength
- 3D distance between fiber tip and neuron

livn also integrates with the [cleo](https://cleosim.readthedocs.io/) library for modelling optogenetic stimulation with detailed laser and opsin models.
