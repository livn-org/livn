# Custom Backends

livn ships with built-in [backends](/guide/backends), but you can use any Python module that exports an `Env` class implementing the livn Env protocol.

## Usage

Set `LIVN_BACKEND` to the fully qualified import path of your backend module:

```sh
export LIVN_BACKEND=my_package.backend
```

Then use livn as normal as the custom `Env` is resolved automatically:

```python
from livn import make

env = make("EI")  # uses my_package.backend.Env
```

## Writing a custom backend

A backend module must export an `Env` class that implements the [`Env` protocol](/guide/concepts/env). At minimum:

```python
# my_package/backend.py

from livn.types import Env as EnvProtocol
from livn.run import Run
from livn.stimulus import Stimulus

class Env(EnvProtocol):
    def __init__(self, system, model=None, io=None, seed=123, comm=None, subworld_size=None):
        ...

    def init(self):
        # Load cells, connections, etc.
        ...
        return self

    def run(self, duration, stimulus=None, dt=0.025, **kwargs):
        # Run the simulation and assemble the recordings into a Run
        return (
            Run(t0=t_start, duration=duration)
            .add_spikes(spike_ids, spike_times)
            .add_voltage(voltage_ids, voltages, dt=self.voltage_recording_dt)
            .add_current(current_ids, currents, dt=self.membrane_current_recording_dt)
        )

    def _record_spikes(self, population):
        ...
        return self

    def _record_voltage(self, population, dt=0.1):
        ...
        return self

    # ... other Env protocol methods
```

::: tip
Your backend can reuse `livn.types`, `livn.stimulus`, `livn.system`, etc., only the simulation engine needs to be custom.
:::

### Recording

`_record_<name>` is the extension point for recordings. The public `env.record(what, population, **kwargs)` normalizes `population` (`None` means [all active populations](/guide/concepts/env#recording), a list is iterated) and then dispatches to `_record_<what>` on your class, so an implementation only ever sees a single population name plus the signal's own options:

```python
class Env(EnvProtocol):
    def _record_threshold(self, population, dt=0.1):
        ...
        return self
```

Defining that method is all it takes: `env.record("threshold", dt=0.5)` now works, `recordable()` lists `threshold`, and `run` can return the trace via `run.add("threshold", ids, values, dt=dt)`. `record_spikes` / `record_voltage` / `record_membrane_current` are wrappers over the same dispatcher; override `_record_spikes` and friends, instead of the public wrappers, so the population normalization is kept.

### What `run` must return

`run` returns a [`Run`](/guide/concepts/env#running-a-simulation). `add_spikes` / `add_voltage` / `add_current` cover the three standard channels and are a no-op when the recording was never enabled, so there is nothing to branch on. They are thin wrappers over `add`, which is what you use for anything else your model records:

```python
from livn.run import Run

run = Run(t0=t_start, duration=duration)
run = run.add_spikes(spike_ids, spike_times)
run = run.add("threshold", threshold_ids, thresholds, dt=0.1)
```

A channel that was never added reads back as `None`, so it allocates nothing. Times are stored relative to `t0`, meaning the spike times of a continued run start at zero again.

### Exposing the cells

`env.cells` is a [`CellRegistry`](/guide/concepts/env#cell-parameters) that is mapping from population name to `{gid: Cell}` that also resolves a gid directly. Fill it as `init()` constructs the cells, one `add()` per population:

```python
from livn.cells import CellRegistry

class Env(EnvProtocol):
    def __init__(self, system, ...):
        self.cells = CellRegistry(self)

    def init(self):
        for population in self.system.populations:
            self.cells.add(population, {gid: Cell(self, population, gid) for gid in ...})
        return self
```

A `Cell` is the handle to one cell's physical parameters. Subclass [`livn.types.Cell`](/guide/concepts/env#cell-parameters), which carries the identity (`env`, `gid`, `population`), the `repr` and the `unknown_param()` error, and implement the two methods over whatever holds the values:

```python
from livn.types import Cell

class MyCell(Cell):
    def get_params(self) -> dict[str, float]:
        return {name: ... for name in ...}

    def set_params(self, params) -> "Env":
        for name, value in params.items():
            if name not in ...:
                raise self.unknown_param(name, ...)
            ...
        return self.env
```

A handle that needs its own state (the NEURON one wraps a cell object, the brian2 and diffrax ones keep a row index) takes it in `__init__` and passes the identity up:

```python
    def __init__(self, env, population, gid, index):
        super().__init__(env, population, gid)
        self._index = int(index)
```

That is all `env.cells.get_params()`, `env.cells.set_params()` and the `cells-` prefix of `Env.set_params()` need as the registry implements them over the cells it holds. Pass it the `comm` your cells are distributed over and it spans the ranks; subclass it (as the diffrax backend does) when a backend can do better than cell by cell, for instance to set a whole parameter array in one operation.

## What you may rely on from `system`

The `system` argument is not necessarily a `livn.system.System`: users may pass a system directory, a neuron count, or their own object. Normalize it first with `livn.system.resolve`, which turns a `str` into a `System`, an `int` into a [`ParallelSystem`](/guide/concepts/system#parallelsystem), and passes anything else through:

```python
from livn.system import resolve

class Env(EnvProtocol):
    def __init__(self, system, model=None, io=None, seed=123, comm=None, subworld_size=None):
        self.system = resolve(system, comm=comm)
        self.model = model if model is not None else self.system.default_model()
        self.io = io if io is not None else self.system.default_io()
```

What the resolved object guarantees is the [`System` protocol](/guide/concepts/system):

| member | |
|---|---|
| `name`, `num_neurons`, `gids` | identity and size |
| `populations`, `population_ranges`, `population_count(p)` | population layout |
| `neuron_coordinates`, `coordinate_array(p)`, `transform_coordinates(f)` | geometry |
| `connections_config`, `projection_array(pre, post)`, `connectivity_matrix()` | connectivity, empty for an unconnected system |
| `default_io()`, `default_model()` | what `Env` falls back to |
| `selection(spec)` | cell subselection |

Anything else a concrete system exposes is an implementation detail of that class and not something to rely on.

