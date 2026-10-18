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

env = make("EI")
env = env.record_spikes()
env = env.record_voltage()

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

### ChR2(H134R) (4-state)

The four-state model of Williams et al. (2013), *Computational optogenetics: empirically-derived voltage- and light-sensitive channelrhodopsin-2 model*, PLoS Comput Biol 9(9): e1003220, as implemented for NEURON in ModelDB 244414. Two open states (O1, O2) and two closed states (C1, C2), with an activation variable `p` that follows the light with a 1.3 ms time constant:

```
C1 --ε1 F p--> O1 <--e12/e21--> O2 <--ε2 F p-- C2 --Gr--> C1
     O1 --Gd1--> C1,            O2 --Gd2--> C2
```

The current is `gmax (A + B e^{V/C})/V (O1 + γ O2)(V − E)`, i.e. inwardly rectifying, with the closing rate `Gd1` and recovery `Gr` voltage dependent. All rates scale by Q10 from 22 °C to the model's `celsius`. It desensitizes, so the second of two pulses is weaker, which matters for pulse trains. Select it with `ReducedCalciumSomaDendrite(opsin="ChR2H134R")`; see [RCSD](/models/rcsd#opsin-configuration) for its parameters and per-cell expression. It is available on the NEURON, native and diffrax backends.

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
            "wavelength": 473.0,
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
    fiber_radius=100.0,
    wavelength=473.0,
    # tissue: Medium(optical=KubelkaMunk(scattering_per_mm=11.2, refractive_index=1.36))
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

The wavelength is read from `meta_data["wavelength"]` (default: 473.0 nm). When using `LightArray`, the wavelength propagates automatically.

## Backend details

### NEURON

RhO3c is a POINT_PROCESS at the midpoint of the section; ChR2H134R a density mechanism on every segment. Either receives the light each timestep through a CVode scatter/gather callback: RhO3c as photon flux (`phi`, converted from the irradiance), ChR2H134R as irradiance (`Irradiance`, mW/mm²).

### Diffrax

In the Diffrax backend, the opsin dynamics are integrated directly into the ODE system. The opsin's state variables (C, O for RhO3c; O1, O2, C1, p for ChR2H134R) are appended to the neuron state vector and solved alongside the membrane dynamics. The ChR2H134R conductance is one value per cell (`opsin_gmax`), differentiable like the light. This makes the full optical stimulation pathway end-to-end differentiable.

```python
import os
os.environ["LIVN_BACKEND"] = "diffrax"

import jax
from livn import make

env = make("EI")
# Gradients through opsin dynamics are available
```

### brian2

The brian2 backend does not have built-in opsin models. Use the [Cleo integration](/guide/advanced/cleo) for optogenetic experiments with brian2.
