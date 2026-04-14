# Optical Stimulation

livn supports optogenetic stimulation through built-in opsin models. Neurons expressing light-sensitive opsins (channelrhodopsins) generate photocurrents in response to light, enabling precise optical control of neural activity.

## Overview

Optical stimulation in livn works in three layers:

1. **Opsin model** — a kinetic model of the light-sensitive ion channel, embedded in each neuron
2. **Stimulus** — an irradiance signal (mW/mm²) delivered to target neurons
3. **IO device** — an optional `LightArray` that maps fiber-optic power to per-neuron irradiance

Both the **NEURON** and **Diffrax** backends support optical stimulation natively. The brian2 backend can use optical stimulation through the [Cleo integration](/guide/advanced/cleo).

## Quick start

```python
import os
os.environ["LIVN_BACKEND"] = "neuron"  # or "diffrax"

import numpy as np
from livn import make
from livn.stimulus import Stimulus

env = make("EI2")
env.record_spikes()
env.record_voltage()

# Create a light pulse: 5 mW/mm² for 50ms on all neurons
n_neurons = env.system.num_neurons
dt = 0.1
duration = 200  # ms
timesteps = int(duration / dt)

irradiance = np.zeros((timesteps, n_neurons))
irradiance[500:1000, :] = 5.0  # light ON from 50ms to 100ms

stim = Stimulus.from_irradiance(irradiance, dt=dt)
it, t, iv, v, *_ = env.run(duration, stimulus=stim)
```

## Opsin models

### RhO3c (3-state)

The default opsin model. A three-state Markov kinetic model (Nikolic et al. 2009) with states:

- **C** (closed) — resting state, photon absorption opens the channel
- **O** (open) — conducting state, allows cation current
- **D** (dark-adapted/desensitized) — inactivated state, recovers to C in the dark

The transitions are governed by light-dependent activation (Ga) and fixed rates for closing (Gd) and recovery (Gr):

```
C --Ga(φ)--> O --Gd--> D --Gr--> C
```

where `Ga(φ) = ka · φᵖ / (φᵖ + φmᵖ)` depends on photon flux φ.

### RhO6c (6-state)

An alternative six-state model (Grossman et al. 2011) available in the NEURON backend. It has two open states and two closed states, providing more accurate kinetics for sustained illumination:

```
C1 <-> I1 <-> O1 <-> O2 <-> I2 <-> C2
```

To use RhO6c, override the model's opsin configuration:

```python
from livn.models.rcsd import ReducedCalciumSomaDendrite

class MyModel(ReducedCalciumSomaDendrite):
    def opsin_config(self):
        return {
            "mechanism": "RhO6c",
            "sections": ["soma"],
            "wavelength_nm": 473.0,
        }
```

## Using the `LightArray` IO

For spatially structured illumination, use `LightArray` to model fiber-optic light delivery. It computes per-neuron irradiance from fiber positions and power using a Kubelka-Munk scattering model (Aravanis et al. 2007).

```python
from livn.io import LightArray
import numpy as np

# Define fiber positions: [id, x, y, z] in micrometers
fibers = np.array([
    [0, 0.0, 0.0, -100.0],    # fiber above center
    [1, 200.0, 0.0, -100.0],  # fiber offset to the right
])

light = LightArray(
    fiber_coordinates=fibers,
    numerical_aperture=0.37,
    fiber_radius_um=100.0,
    wavelength_nm=473.0,
    scattering_coefficient=11.2,  # mm^-1
)

# Power trace: [timesteps, n_fibers] in mW
power = np.zeros((2000, 2))
power[500:1000, 0] = 1.0  # 1 mW from fiber 0

# Convert to per-neuron irradiance stimulus
stim = light.cell_stimulus(system.neuron_coordinates, power, dt=0.1)

it, t, *_ = env.run(200, stimulus=stim)
```

The `LightArray` automatically:
- Computes distance-dependent transmittance using tissue scattering
- Converts fiber power (mW) to irradiance (mW/mm²) at each neuron
- Returns a `Stimulus` with `input_mode="irradiance"`

## Unit conversion

The `Stimulus.convert_to()` method converts between irradiance and photon flux units using Planck's equation:

```
φ = I / Eγ,    Eγ = hc / λ
```

where I is irradiance (mW/mm²), φ is photon flux (photons/s/mm²), h is Planck's constant, c is the speed of light, and λ is the wavelength.

```python
stim = Stimulus.from_irradiance(irradiance, dt=0.1)

stim_flux = stim.convert_to("photon_flux")

stim_irr = stim_flux.convert_to("mW/mm2")
```

The wavelength is read from `meta_data["wavelength_nm"]` (default: 473.0 nm). When using `LightArray`, the wavelength propagates automatically.

## Backend details

### NEURON

Each opsin POINT_PROCESS:
- Is attached to the specified cell section (e.g. soma at midpoint 0.5)
- Receives photon flux (`phi`) via CVode scatter/gather callbacks each timestep
- Injects photocurrent into the host section

The irradiance stimulus is automatically converted to photon flux before delivery.

### Diffrax

In the Diffrax backend, the opsin dynamics are integrated directly into the ODE system. The RhO3c state variables (C, O) are appended to the neuron state vector and solved alongside the membrane dynamics. This makes the full optical stimulation pathway end-to-end differentiable.

```python
import os
os.environ["LIVN_BACKEND"] = "diffrax"

import jax
from livn import make

env = make("EI2")
# Gradients through opsin dynamics are available
```

### brian2

The brian2 backend does not have built-in opsin models. Use the [Cleo integration](/guide/advanced/cleo) for optogenetic experiments with brian2.
