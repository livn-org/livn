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
from livn.stimulus import Stimulus

class Env(EnvProtocol):
    def __init__(self, system, model=None, io=None, seed=123, comm=None, subworld_size=None):
        ...

    def init(self):
        # Load cells, connections, etc.
        ...
        return self

    def run(self, duration, stimulus=None, dt=0.025, **kwargs):
        # Run simulation, return (spike_ids, spike_times, voltage_ids, voltages, current_ids, currents)
        ...

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
