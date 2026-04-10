# Cleo Integration

[Cleo](https://cleosim.rtfd.io) (Closed-Loop, Electrophysiology, and Optogenetics experiment simulation testbed) enables optogenetic and electrophysiology experiments on livn neural cultures. By injecting opsins and light sources into the brian2 simulation, you can optically stimulate specific neuron populations and study their response.

::: warning
Cleo integration requires the **brian2** backend. Set `LIVN_BACKEND=brian2` or configure it programmatically before creating the environment.
:::


## Quick start

Cleo wraps a brian2 `Network` in a `CLSimulator` that manages device injection (opsins, light sources, electrodes). livn provides a `cleosim()` convenience method that automatically assigns 3D coordinates from the livn system and returns the `CLSimulator` ready for device injection. Once configured, use the livn env instead of `CLSimulator.run`:

```python
import os
os.environ["LIVN_BACKEND"] = "brian2"

import numpy as np
import brian2 as b2
from cleo.light import Light, fiber473nm
from cleo.opto import chr2_4s
from livn.env import Env

# Create and initialize the environment
env = Env("path/to/system").init()
env.apply_model_defaults()
env.record_spikes()

# Wrap with cleo (assigns coordinates automatically)
sim = env.cleosim()

# Inject ChR2 opsin into excitatory neurons
opsin = chr2_4s()
exc = env._populations["EXC"]
sim.inject(opsin, exc, Iopto_var_name="I")

# Place a fiber optic light source above the culture
light = Light(
    name="fiber",
    light_model=fiber473nm(),
    coords=np.array([[0, 0, -100]]) * b2.um,
    direction=np.array([0, 0, 1]),
    wavelength=473 * b2.nmeter,
    max_value=10 * b2.mwatt / b2.mm**2,
)
sim.inject(light, exc)

# Run with a light pulse
env.run(200)                               # baseline
light.update(5 * b2.mwatt / b2.mm**2)      # light ON
it, t, iv, v, _, _ = env.run(100)
light.update(0.0 * b2.mwatt / b2.mm**2)    # light OFF
env.run(200)                               # recovery

print(f"Spikes during light: {len(t)}")
```

## `env.cleosim()`

Returns a `cleo.CLSimulator` wrapping the environment's brian2 network. Coordinates from the system graph are automatically assigned to each population's `NeuronGroup` via `cleo.coords.assign_coords`.

```python
sim = env.cleosim()
```

The returned simulator's `run()` method is intentionally disabled. Always use `env.run()` to advance the simulation to ensure that livn's time tracking, stimulus delivery, and monitor collection work correctly.


## Coordinates

`cleosim()` reads coordinates from the system graph via `system.coordinate_array(name)`, which returns an `[N, 4]` array of `(gid, x, y, z)` values. The x, y, z columns (in micrometers) are assigned to the corresponding brian2 `NeuronGroup`.

If you need custom coordinates, assign them before calling `cleosim()`:

```python
from cleo.coords import assign_coords

pop = env._populations["EXC"]
my_coords = np.random.randn(len(pop), 3) * 50 * b2.um
assign_coords(pop, my_coords)

# cleosim() will skip populations that already have coordinates
sim = env.cleosim()
```

## Combining with electrical stimulation

Optical and electrical stimulation can run simultaneously. Pass a stimulus array to `env.run()` as usual while the light source is active:

```python
# Electrical stimulus on IO channels
inputs = np.zeros([100, env.io.num_channels])
inputs[50:80, 0] = 1.0  # pulse on channel 0
stimulus = env.cell_stimulus(inputs)

# Light is already on via light.update(...)
it, t, iv, v, _, _ = env.run(100, stimulus=stimulus)
```

This lets you study how optical and electrical inputs interact in the culture.
