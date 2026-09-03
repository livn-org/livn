# In the Browser via Pyodide

livn can run directly in the browser using [Pyodide](https://pyodide.org/), a Python distribution for WebAssembly. This may be useful for quick experimentation without any local installation or browser-based visualization.

::: tip
You can try this below or in any Pyodide-powered environment such as [JupyterLite](https://jupyterlite.readthedocs.io/) or the [REPL](https://pyodide.org/en/stable/console.html).
:::


## Setup

Install livn and its dependencies using `micropip` (Pyodide 0.29.3 or later is required):

```python
import micropip
await micropip.install(['livn', 'pyodide-http'])

import pyodide_http
pyodide_http.patch_all()
```

## Simulating in the browser

Since livn publishes a [PyEmscripten wheel](https://peps.python.org/pep-0783/) carrying a WebAssembly build of the `native` backend, simulation works in the browser with no extra steps:

```python
import numpy as np
from livn.backend import backend
from livn.env import Env
from livn.stimulus import Stimulus

backend()  # 'native', picked up automatically

env = Env({"EXC": 1}).init()
env.record_spikes()

current = np.zeros((2400, 1))
current[80:, 0] = 1.0  # 1 nA from 2 ms
run = env.run(60.0, stimulus=Stimulus.from_current(current, dt=0.025, gids=np.array([0])))
list(run.spike_times)
```

Performance is roughly that of the native build divided by the browser's wasm overhead, but everything is single-threaded, so keep in-browser networks small (e.g. a few dozen cells).

## Usage

Once installed, you can use livn as usual:

```python
from livn.env import Env
from livn.system import predefined

env = Env(predefined('EI'))
env.io.electrode_coordinates
```

Note that `predefined()` downloads system files into the in-browser filesystem. Since Pyodide uses an in-memory filesystem by default, downloaded systems are not persisted and will need to be re-downloaded on each page load.

## Interactive Demo

Try it right here! Click the button to load Pyodide, download the EI system, and visualize neuron and electrode positions directly in your browser.

<PyodideWidget />

::: details Click me to toggle the code
<<< @/.vitepress/theme/PyodideWidget.vue

:::