# In the Browser via Pyodide

livn can run directly in the browser using [Pyodide](https://pyodide.org/), a Python distribution for WebAssembly. This may be useful for quick experimentation without any local installation or browser-based visualization.

::: tip
You can try this below or in any Pyodide-powered environment such as [JupyterLite](https://jupyterlite.readthedocs.io/) or the [REPL](https://pyodide.org/en/stable/console.html).
:::


## Setup

Install livn and its dependencies using `micropip` (Pyodide 0.29.3 or later is required):

```python
import micropip
await micropip.install(['livn', 'fsspec', 'huggingface_hub', 'httpcore'])
```

::: warning
Simulation backends like brian2 are not available in Pyodide unless you provide a custom Wasm build. Out of the box, you can load and inspect systems and datasets but not run simulations.
:::

## Usage

Once installed, you can use livn as usual:

```python
from livn.env import Env
from livn.system import predefined

env = Env(predefined('EI1'))
env.io.electrode_coordinates
```

Note that `predefined()` downloads system files into the in-browser filesystem. Since Pyodide uses an in-memory filesystem by default, downloaded systems are not persisted and will need to be re-downloaded on each page load.

## Interactive Demo

Try it right here! Click the button to load Pyodide, download the EI1 system, and visualize neuron and electrode positions directly in your browser.

<PyodideWidget />

::: details Click me to toggle the code
<<< @/.vitepress/theme/PyodideWidget.vue

:::