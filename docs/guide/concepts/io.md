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

    def potential_recording(self, distances, membrane_currents_uA, dt=None, key=None) -> array:
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

When stimulating through an MEA, the effect on each neuron depends on its distance to the activated electrode. livn models this through the [medium](#medium) the array sits in, i.e. a point source whose field is `ρ · I / (4π · h_eff)` at the electrode surface, falling as `1 / sqrt(r² + h_eff²)`, where `ρ` is the stimulation resistivity, `I` is the electrode current, `r` is the distance between the neuron and the electrode and `h_eff` combines the electrode radius (`MEA(electrode_radius=15.0)`) with the culture height. Only neurons within `input_radius` of an electrode receive meaningful stimulation.

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

#### Physical detection

Given the recording's noise level, the MEA decides which cells an electrode actually sees the way a spike detector does. A cell counts on a channel when its spike there reaches `threshold_sigma` times the noise, and spikes closer together than `dead_time` on one channel are dropped.

```python
from livn.io import MEA, ComposedIO, Noise

io = ComposedIO(
    inputs=None,   # the same MEA stimulates
    outputs=[MEA(threshold_sigma=5.0, dead_time=3.0, output_radius=150.0), Noise(3.6)],
)
cit, ct = env.channel_recording(it, t)   # read through the env
```

The noise, threshold and dead time belong to the recording. How large a cell's spike is at a given distance belongs to the cell, so `env.channel_recording` attaches the model's `extracellular_profile()` when it reads the array, and populations with smaller cells come out weaker (`extracellular_amplitude`).

### Extracellular potential estimation

For membrane current recordings, the MEA can estimate the extracellular field potential (local field potential) at each electrode using the point-source model:

```python
distances = mea.distances(system.neuron_coordinates)
lfp = mea.potential_recording(distances, membrane_currents_uA)
# Returns: [n_channels, timesteps] in µV
```

On an env, `env.potential_recording(conditions=run.conditions())` projects a run where `run.conditions()` gives its membrane currents, their gids and `dt`, while the env supplies the recording geometry itself; see [Recording compositions](#recording-compositions).

This is used by decodings like [`LFP`](/guide/concepts/decoding) for spectral analysis of the extracellular signal.

### Fitting an array to a system

The cultures ship without an array so that the recording setup stays a choice of the experiment rather than a property of the network. `electrode_array_coordinates_for_area` lays out a grid over a system's extent at a given pitch:

```python
from livn.io import MEA, electrode_array_coordinates_for_area
from livn.system import NeuroH5System

system = NeuroH5System("./systems/graphs/EI")
(xmin, ymin, _), (xmax, ymax, _) = system.bounding_box

mea = MEA(electrode_array_coordinates_for_area(200, ((xmin, ymin), (xmax, ymax))))
mea.num_channels
```

A system that does ship one (`mea.json`) is loaded with `MEA.from_directory(path)`, which is what `make()` uses for its default IO.

### Biphasic pulse stimulation

The `Stimulus` class provides a helper for generating charge-balanced biphasic pulses, the standard stimulation waveform for MEA experiments:

```python
from livn.stimulus import Stimulus

stim = Stimulus.biphasic_pulse(
    n_channels=mea.num_channels,
    channels=[5, 6],         # electrodes to stimulate
    amplitude_uA=1.5,           # µA
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
    fiber_radius=100.0,
    wavelength=473.0,
    # the tissue it shines into is a Medium(optical=KubelkaMunk(scattering_per_mm=11.2))
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
- Tissue scattering and refractive index, from the [medium](#medium)'s optical part
- 3D distance between fiber tip and neuron

livn also integrates with the [cleo](https://cleosim.readthedocs.io/) library for modelling optogenetic stimulation with detailed laser and opsin models.

## Patterned light (spots)

`PatternedLight` models light spots projected onto a monolayer. Each channel is a spot, a disc `[id, x, y, radius]` in sample µm, and a command is the fraction of the laser level the spot is lit at, in `[0, 1]` (`input_units == "normalized"`). `irradiance` is the peak at the laser level (mW/mm²).

```python
from livn.io import GaussianBlur, Medium, PatternedLight

light = PatternedLight(
    spots=[[0, 200.0, 150.0, 40.0], [1, 600.0, 150.0, 40.0]],
    irradiance=5.0,     # mW/mm² at the laser level
    wavelength=470.0,
    medium=Medium(optical=GaussianBlur(width=5.0)),  # the default
)

commands = np.zeros((1000, 2)); commands[100:150, 0] = 1.0  # spot 0 on for 5 ms
stim = light.cell_stimulus(env.stimulus_coordinates(), commands, dt=0.1)  # irradiance
```

## Medium

A `Medium` is the preparation between the devices and the tissue defining how fields and light propagate.

```python
from livn.io import DispersivePointSource, KubelkaMunk, Medium, PointSource

medium = Medium(
    electrical=PointSource(
        sigma_s_per_m=0.3,                   # conductivity
        height=50.0,                      # culture height
        min_distance=5.0,                 # floor on the source distance
        stimulation_resistivity_ohm_m=3.5,   # None: 1 / sigma
    ),
    optical=KubelkaMunk(scattering_per_mm=11.2, refractive_index=1.36),
)
mea = MEA(medium=medium)
```

## Transfer stacks

A transfer acts on a channel's signal after the device that measured it.

| transfer | what it models |
|---|---|
| `ElectrodeRC(r_seal, c_pF)` | the electrode-electrolyte interface as a first-order RC low-pass; learnable seal resistance, shared or per channel |
| `FIR(taps, per_channel=False)` | a learned causal impulse response |
| `Bandpass(low, high, order)` | the acquisition filter (a causal Butterworth) |
| `Noise(sigma_uV, seed=0)` | additive Gaussian noise |

```python
from livn.io import FIR, MEA, ComposedIO, ElectrodeRC, Noise

io = ComposedIO(
    inputs=None,
    outputs=[MEA(), ElectrodeRC(50.0, 100.0), FIR([1.0, 0.3]), Noise(6.0)],
)
v = io.potential_recording(distances, membrane_currents_uA, dt=0.1)             # deterministic
v = io.potential_recording(distances, membrane_currents_uA, dt=0.1, key=key)    # with noise
```

`[device, *transfers]` is shorthand for the recording composition `Chain(extracellular=ExtracellularPotential(), *transfers)` (`io.recording`). A stage is named after its class (`electrode_rc`, `fir`, `noise`; a repeated class is numbered, `noise_2`), and its parameters are addressed by that name, `io.set_params({"recording-electrode_rc-r_seal": 30.0})`.

## Recording compositions

What an electrode records is not only a function of the simulated cells but can have many contributors such as a photo-electric effect, stimulation, or temperature. These dependencies are specified as `Conditions`:

```python
from livn.conditions import Conditions
from livn.io import (
    MEA, Bandpass, Chain, ComposedIO, ElectrodeRC, ExtracellularPotential, LightArtifact,
    Noise, PatternedLight, Sum,
)

recording = Chain(
    Sum(extracellular=ExtracellularPotential(), artifact=LightArtifact(n_channels=60)),  # sources, added
    rc=ElectrodeRC(50.0, 100.0), band=Bandpass(300.0, 3000.0), noise=Noise(6.0),  # transfers, in order
)
io = ComposedIO(inputs=PatternedLight(spots, irradiance=5.0), outputs=MEA(), recording=recording)

v = env.potential_recording(
    conditions=run.conditions(inputs=commands),
    key=key,
)
```

The conditions can provide the following dependencies:

| field | what it is | set by |
|---|---|---|
| `membrane_currents_uA`, `dt` | the run's currents `[n_neurons, T]` and their interval (ms) | the caller (`dt`: the env's recording interval when unset) |
| `inputs`, `inputs_dt`, `stim_markers` | the stimulating device's commands `[T_in, n_input_channels]` on their own grid, and marker times (ms) | the caller |
| `neuron_coordinates`, `gids` | the recording coordinate of each current row (`gids` selects them, a cell with several sections repeating) | the env |
| `sections` | the section of each current row | the caller |
| `t0`, `segment`, `start` | where the window is | the caller (a `Window` of `livn.optimization.data`) |

```python
v = env.potential_recording(conditions=run.conditions(inputs=commands))
v = env.potential_recording(Chain(extracellular=ExtracellularPotential(), rc=ElectrodeRC()), run.conditions())  # dark
v = env.potential_recording(Chain(artifact=io.recording["artifact"], rc=ElectrodeRC()), Conditions(inputs=commands))  # empty dish
```


To define custom conditions, use a subclass:

```python
class RigConditions(Conditions):
    temperature: Float[Array, " timestep"] | None = None

class Drift(eqx.Module):
    gain: Array
    def __call__(self, *, temperature, io): ...   # io.num_channels, io.medium, ...

recording = Chain(Sum(extracellular=ExtracellularPotential(), drift=Drift(...)), rc=ElectrodeRC())
env.potential_recording(recording, run.conditions(RigConditions, temperature=T))
```

**Names and parameters.** An unnamed block takes its class name in snake case (or its class's `default_name`, `extracellular` for `ExtracellularPotential`); two unnamed blocks of one class raise and ask for names that have to be unique across nested blocks. A composition serializes as `{"cls": "livn.io.Chain", "kwargs": {name: block, ...}}` and round-trips with its `ComposedIO`.

### Photoelectric artifact

Light on an electrode induces a voltage that can be modeled using `LightArtifact`. The `inputs` are translated into the irradiance that reaches each electrode `I[c, t]` (mW/mm²). It is optionally saturated (`I / (1 + I / saturation)`) and passes through a causal kernel per channel on a basis of exponentials with time constants `taus` (ms).

## Composed IO

When stimulation and recording use different physical modalities (for example, optical stimulation via a `LightArray` and electrical recording via an `MEA`), use `ComposedIO` to pair them into a single IO object:

```python
from livn.io import ComposedIO, LightArray, MEA, Medium

io = ComposedIO(
    inputs=LightArray(fibers),
    outputs=MEA(),
    medium=Medium(),
)
env = Env(system, io=io)
```

`ComposedIO` delegates `cell_stimulus` to the `inputs` IO and `channel_recording`, `potential_recording`, and `distances` to the `outputs` IO. Channel properties (`num_channels`, `channel_ids`) are taken from the output IO. `outputs` may be a list `[device, *transfers]` (see [transfer stacks](#transfer-stacks)), or the device with a full [recording composition](#recording-compositions) as `recording=Chain(...)`.

`inputs` may also be a list of stimulating devices, e.g. two light sources, or an MEA that stimulates (`"outputs"`, the recording device itself) beside a light:

```python
io = ComposedIO(inputs=[PatternedLight(spots_a), PatternedLight(spots_b, irradiance=2.0)], outputs=MEA())
stimulus = io.cell_stimulus(coordinates, commands, dt=0.1)   # a ComposedStimulus of the two
```

