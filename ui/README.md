# livn UI

Browser-based 3D visualization and Python console for livn neural systems.

## User Documentation

If you're looking to use the UI, see the main docs:

- **[WebUI Guide](../docs/ui.md)** — starting the UI, using the console, 3D scene controls, view options
- **[HSDS Setup](../docs/systems/hsds.md)** — configuring HSDS for serving system data to the browser
- **[Installation (Pyodide)](../docs/installation/)** — running livn in the browser

---

## Developer Guide

The rest of this document is for contributors working on the UI codebase. It covers architecture, component design, and the reactive data flow between Pyodide and Threlte.

Built on [Threlte](https://threlte.xyz/) (Svelte 5 + Three.js) and [Pyodide](https://pyodide.org/).

## Quick Start

```bash
# 1. Start the HSDS server (optional — pyfive fallback if skipped)
livn ui server --launch

# 2. Build the livn wheel and start the Vite dev server
livn ui web --launch

# Open http://localhost:5173
```

The dev server watches for file changes. The livn wheel is rebuilt each time you run `--launch`.

## How It Works

The UI is a reactive visualization of a livn `Env` object. Users interact with livn through the standard Python API in a browser-based console; the Threlte scene automatically reflects the `Env` state. Svelte components are thin wrappers that mirror the livn package structure: `System` renders neurons, `io.MEA` renders electrodes, and read data directly from Pyodide Python objects via typed array transfer.

There is no UI-specific API. Everything in the console is standard livn Python:

```python
from livn.env import Env
from livn.system import predefined
from livn.io import MEA

env = Env(predefined('EI1'))                 # → scene shows EI1 neurons
env.io = MEA.from_directory(env.system.uri)  # → electrodes appear
```

The Threlte layer adds visualization-only affordances (tooltips, population toggles, point size) that have no Python equivalent.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser                                                         │
│                                                                  │
│  ┌────────────────────────────────┐                              │
│  │  Pyodide (Wasm)               │                              │
│  │                               │                              │
│  │  env = Env(predefined('EI1')) │  ← Python console input      │
│  │  env.system  →  System        │                              │
│  │  env.io      →  MEA           │                              │
│  │  env.model   →  Model         │                              │
│  │  env.run(…)  →  decoding    │  (future)                   │
│  └──────────┬────────────────────┘                              │
│             │ typed array transfer (toJs)                        │
│             │ + change notifications                             │
│             ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Threlte / Svelte 5                                      │   │
│  │                                                          │   │
│  │  EnvScene.svelte          ← reactive root                │   │
│  │    ├─ System.svelte       ← env.system                   │   │
│  │    │    ├─ populations     (instanced spheres per pop)   │   │
│  │    │    └─ bounding box    (wireframe edges)             │   │
│  │    ├─ IO.svelte           ← env.io                       │   │
│  │    │    └─ electrodes      (instanced wireframe cubes)   │   │
│  │    ├─ Decoding.svelte     ← env.decoding (future)        │   │
│  │    │    ├─ spike raster                                  │   │
│  │    │    └─ voltage traces                                │   │
│  │    └─ Tooltip.svelte      ← click interaction            │   │
│  │                                                          │   │
│  │  Console.svelte           ← Python REPL                  │   │
│  │  StatusBar.svelte         ← backend info                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                         │
               ┌─────────▼─────────┐
               │  livn ui server   │  (optional — HSDS + file server)
               └───────────────────┘
```

**Startup flow:**
1. `livn ui server --launch` — starts HSDS on `:5101` + file server on `:5102` (optional)
2. `livn ui web --launch` — builds livn wheel into `web/static/livn.whl`, starts Vite on `:5173`

## Navigation

The UI has two top-level tabs in the nav bar:

| Tab | First view | Second view (detail) |
|-----|-----------|----------------------|
| **Sim** | System selection cards (EI1, EI2, CA1d) | 3D scene + Python console + info panels |
| **Bio** | Recording selection cards (placeholder) | Waveform panel + Setup / Data Info panels |

On first load the app navigates directly to the **EI1 Sim detail** view and auto-loads the system once Pyodide is ready.

### Sim detail

The Sim detail page preserves the full existing visualization workflow:

```python
env = Env(predefined('EI1'))          # loaded automatically on navigation
env.io = MEA.from_directory(...)      # type in the console to add electrodes
```

Three collapsible overlay panels appear in the top-right corner of the 3D scene:

| Panel | Content |
|-------|---------|
| **Setup** | Culture shape, bounding-box dimensions, population counts, total neurons |
| **Neuron Info** | First 10 neurons (GID, population, x/y/z) from `envSystem` |
| **Stim Protocol** | Placeholder electrode, timing, and amplitude values per system |

The existing viz controls (population toggles, point size, opacity, bounding box, electrodes) remain in the top-left overlay.

---

## Directory Structure

```
ui/
├── pyproject.toml              # Python deps
├── server.py                   # HSDS server + file server launcher
├── web.py                      # Wheel builder + Vite dev server launcher
└── web/                        # SvelteKit frontend
    ├── package.json
    ├── svelte.config.js        # Static adapter (SPA, no SSR)
    ├── vite.config.ts          # COOP/COEP headers for SharedArrayBuffer
    ├── tsconfig.json
    ├── static/
    │   └── livn.whl            # Built by `livn ui web --launch`
    └── src/
        ├── app.html
        ├── app.css
        ├── lib/
        │   ├── pyodide.ts      # Pyodide loader, HSDS probe, executeCode(), snapshotEnv()
        │   ├── stores.ts       # Svelte stores (reactive env state) + updateStores() diff
        │   ├── types.ts        # TypeScript interfaces (EnvSnapshot, SystemData, IOData, …)
        │   └── components/
        │       ├── NavBar.svelte         # Top nav bar — Bio / Sim tabs
        │       ├── SimSystemList.svelte  # System selection cards (EI1, EI2, CA1d)
        │       ├── BioRecordingList.svelte  # Recording selection cards (placeholder)
        │       ├── BioRecordingDetail.svelte # Waveform + Setup / Data Info panels
        │       ├── EnvScene.svelte       # Root 3D scene — camera, lights, rendering
        │       ├── System.svelte         # env.system → instanced neurons + bounding box
        │       ├── IO.svelte             # env.io → instanced wireframe electrode cubes
        │       ├── Tooltip.svelte        # Click → neuron info panel (GID, population, coords)
        │       ├── Console.svelte        # xterm.js Python REPL connected to Pyodide
        │       └── StatusBar.svelte      # Backend indicator, env status, execution time
        └── routes/
            ├── +layout.ts      # Disables SSR/prerender (client-only SPA)
            └── +page.svelte    # Navigation state machine + all page views
```

### Component → Python Mapping

| Svelte component       | Renders                     | Python source          |
|------------------------|-----------------------------|------------------------|
| `System.svelte`        | Neurons + bbox              | `env.system`           |
| `IO.svelte`            | Electrodes                  | `env.io` (MEA)         |
| `Console.svelte`       | Python REPL                 | Pyodide interpreter    |
| `SimSystemList.svelte` | System selection cards      | triggers `predefined()`|
| `BioRecordingList.svelte` | Recording cards (stub)   | —                      |

## Reactive Data Flow

Every console execution triggers an env snapshot:

```
Console input → runPythonAsync() → snapshotEnv() → diff → update stores → Threlte re-renders
```

1. User types Python code in `Console.svelte` → sent to `pyodide.runPythonAsync()`
2. After execution, `snapshotEnv()` extracts the current `env` object state as typed arrays
3. The snapshot is diffed against the previous state
4. Changed stores update → Svelte reactivity propagates to Threlte components

Stores are split per env attribute (`envSystem`, `envIO`, `envModel`) so that setting `env.io` does not re-render the neuron cloud.

### Env Snapshot

The `snapshotEnv()` function (in `lib/pyodide.ts`) runs Python inside Pyodide to extract the env state. NumPy arrays transfer to JS as `Float64Array`/`Int32Array` via `toJs()` — no JSON serialization overhead. See `lib/types.ts` for the `EnvSnapshot` interface.

### Console Execution

`executeCode()` in `lib/pyodide.ts` redirects stdout/stderr, runs the code, captures output, and always calls `snapshotEnv()` after execution. The result includes output text, any error, and the new snapshot.

### HSDS Auto-Detection

On initialization, the Pyodide bridge probes `http://localhost:5101/about` (or the `?hsds=` query param). If HSDS responds, it installs h5pyd and sets `LIVN_HSDS`. Otherwise, pyfive is used.

## Scene Components

### `System.svelte`

Renders `env.system` as one `InstancedMesh` per population.

- **Data:** `pop_coords[pop]` → `Float64Array` interleaved `[gid, x, y, z]`
- **Bounding box:** `EdgesGeometry` wireframe from `bounding_box` `[xmin, ymin, zmin, xmax, ymax, zmax]`
- **Coordinate mapping:** system `[x, y, z]` → Three.js `[x, z, y]` (z-axis becomes Y/up)
- **Colors:** `EXC` → `#4fc3f7`, `INH` → `#ef5350`, fallback → `#aaaaaa`
- **Sizing:** `SphereGeometry(span * 0.02 * pointSize, 12, 12)`
- **Interaction:** raycaster click → populate `Tooltip` with GID, population, coordinates

### `IO.svelte`

Renders `env.io` electrode positions as instanced wireframe cubes.

- **Data:** `electrode_coordinates` → `Float64Array` interleaved `[id, x, y, z]`
- **Geometry:** `BoxGeometry(span * 0.03)` wireframe, color `#fdd835`
- **Coordinate mapping:** same `[x, z, y]` transform as System

### `Tooltip.svelte`

Fixed-position panel at top-center, shown on neuron click:
- GID, population, coordinates
- Nearest electrode + distance (if IO data available)
- Dismissible via close button

### `Console.svelte`

xterm.js-based Python REPL connected to Pyodide:
- **Enter:** execute, **Shift+Enter:** newline (multiline)
- **Up/Down:** command history
- **Ctrl+C:** cancel current input
- stdout in white, errors in red (last traceback line)
- Triggers `snapshotEnv()` + `updateStores()` after every execution

### `StatusBar.svelte`

Displays backend type, env attribute status indicators (system/io/model), and last execution time.

### `+page.svelte`

Main layout with CSS Grid:
- Left: 3D scene with overlay view controls (population toggles, point size, opacity, bbox/electrode toggles)
- Right: Console panel
- Bottom: StatusBar spanning full width
- Responsive: stacks vertically on narrow screens (<900px)

## View Config

The `viewConfig` store holds visualization-only state that never touches Python:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `popVisibility` | `Record<string, boolean>` | `{}` | Per-population visibility toggles |
| `pointSize` | `number` | `1.0` | Neuron sphere size multiplier (0.2–3.0) |
| `opacity` | `number` | `0.85` | Neuron sphere opacity (0.1–1.0) |
| `showBoundingBox` | `boolean` | `true` | Show system bounding box wireframe |
| `showElectrodes` | `boolean` | `true` | Show electrode markers |

7. **SvelteKit with static adapter.** Client-only SPA, no SSR. `@sveltejs/adapter-static` produces a bundle servable from any HTTP server.

## Possible extension (PRs welcome)

### Decoding Visualization

The `Decoding.svelte` component (not yet implemented) will render simulation results as 2D overlays:

- **Spike raster:** scatter plot of `(spike_times, spike_ids)` colored by population
- **Voltage traces:** line plots for selected neurons
- **Electrode recordings:** LFP / channel recording line plots

The intended workflow uses livn's Decoding-based replay pattern where an `Encoding` loads pre-saved Arrow or H5 files and feeds them through `env.__call__()`, keeping the browser lightweight (no simulation in Wasm).


### System designer

Allow users to modify system specs like the MEA visually on the fly. This can integrate with the `livn systems generate_2d`` API or be standalone.


### No-code UI

Add a visual mode that allows to configure the Env from a UI which automatically executes the Python commands behind the scene.


### Jupyter Integration

Allow running the application directly in Jupyter Notebooks as a Widget.