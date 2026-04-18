# WebUI

The livn WebUI is a browser-based 3D visualization and Python console for interacting with neural systems. It renders neuron populations, electrode arrays, and bounding boxes in a Threlte/Three.js scene, driven entirely through a Python REPL powered by [Pyodide](/installation/pyodide).

## Starting the UI

The WebUI requires two servers:

1. **Backend server** — serves system data files (and optionally HSDS)
2. **Frontend dev server** — serves the SvelteKit application

```bash
# Start the backend file server (port 5102)
livn ui server --launch

# In another terminal, start the frontend
livn ui web --launch
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

::: tip
If you also want to use [HSDS](/systems/hsds) for HDF5 access, start the HSDS server first:
```bash
livn ui server --launch  # starts both HSDS on :5101 and file server on :5102
```
The WebUI auto-detects HSDS on startup and configures h5pyd automatically. Without HSDS, pyfive is used as a fallback.
:::

## Using the Console

The right panel is a Python REPL. Type standard livn Python code to load and inspect systems:

```python
from livn.env import Env
from livn.system import predefined

env = Env(predefined('EI1'))
```

After each execution, the 3D scene updates automatically to reflect the current `env` state.

### Loading electrodes

```python
from livn.io import MEA
env.io = MEA.from_directory(env.system.uri)
```

Electrode positions appear as yellow wireframe cubes in the scene.

## 3D Scene

The left panel shows a 3D scene with:

- **Neuron populations** — colored spheres (blue for excitatory, red for inhibitory)
- **Electrode array** — yellow wireframe cubes at electrode positions
- **Bounding box** — wireframe outline of the system extent

Click on a neuron to see its GID, population, coordinates, and distance to the nearest electrode.

Use your mouse to orbit, zoom, and pan the camera.

## View Controls

An overlay in the top-left of the scene provides visualization controls:

| Control | Description |
|---------|-------------|
| Population toggles | Show/hide individual populations |
| Point size | Adjust neuron sphere size (0.2×–3.0×) |
| Opacity | Adjust neuron transparency (10%–100%) |
| Bounding box | Toggle system bounding box wireframe |
| Electrodes | Toggle electrode markers |

These controls affect only the visualization — they do not modify the Python `env` object.

## Status Bar

The bottom bar shows:
- **Backend type** — "HSDS" or "pyfive (local)"
- **Env state** — which attributes are currently set (system, io, model)
- **Execution time** — duration of the last console command
