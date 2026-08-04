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

env = make("EI1")  # uses my_package.backend.Env
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

    def record_spikes(self, population=None):
        ...
        return self

    def record_voltage(self, population=None, dt=0.1):
        ...
        return self

    # ... other Env protocol methods
```

::: tip
Your backend can reuse `livn.types`, `livn.stimulus`, `livn.system`, etc., only the simulation engine needs to be custom.
:::

### What `run` must return

`run` returns a [`Run`](/guide/concepts/env#running-a-simulation). `add_spikes` / `add_voltage` / `add_current` cover the three standard channels and are a no-op when the recording was never enabled, so there is nothing to branch on. They are thin wrappers over `add`, which is what you use for anything else your model records:

```python
from livn.run import Run

run = Run(t0=t_start, duration=duration)
run = run.add_spikes(spike_ids, spike_times)
run = run.add("threshold", threshold_ids, thresholds, dt=0.1)
```

A channel that was never added reads back as `None`, so it allocates nothing. Times are stored relative to `t0`, meaning the spike times of a continued run start at zero again.

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

