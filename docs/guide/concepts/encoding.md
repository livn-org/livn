# Encoding

**Encoding** translates external inputs (features, observations, actions) into neural [stimuli](/guide/concepts/stimulus). It is the input side of the encoding-decoding pipeline that allows livn to interface with ML workflows and RL environments.

## The `Encoding` base class

All encodings extend `Encoding` and implement a `__call__` method:

```python
from livn.types import Encoding

class MyEncoding(Encoding):
    def __call__(self, env, t_end, inputs):
        """
        Transform external inputs into a Stimulus.

        Args:
            env: The simulation environment
            t_end: Simulation duration in ms
            inputs: External input data (features, actions, etc.)

        Returns:
            A Stimulus object or None (for spontaneous activity)
        """
        ...
```

## How encodings are used

Encodings plug into the `env()` call interface alongside a [decoding](/guide/concepts/decoding):

```python
result = env(
    decoding=MeanFiringRate(duration=1000),
    inputs=my_features,
    encoding=my_encoding,
)
```

The flow is:

1. `encoding(env, duration, inputs)` → produces a `Stimulus`
2. `env.run(duration, stimulus)` → raw recordings
3. `decoding(env, *recordings)` → processed output

## Built-in encodings

### H5Inputs

Loads stimulus patterns from HDF5 files, used primarily with the NEURON backend for replaying recorded or pre-computed spike train inputs:

```python
from livn.encoding import H5Inputs

encoding = H5Inputs(
    filepath="path/to/inputs.h5",
    namespace="",
    attribute="Spike Train",
    onset=0,
)
```

### Null encoding

For spontaneous activity simulations (no external stimulus), return `None`:

```python
class NoStimulus(Encoding):
    def __call__(self, env, t_end, inputs):
        return None
```

## Custom encodings

Encodings are where you implement the neural coding strategies that translate ML-level features into biologically meaningful stimulation patterns. Here's an example that encodes scalar features as firing rates across MEA channels:

```python
import numpy as np
from livn.types import Encoding
from livn.stimulus import Stimulus

class RateEncoding(Encoding):
    """Encode feature values as stimulation rates across channels"""
    n_channels: int = 16
    max_rate: float = 100.0  # max stimulation frequency in Hz

    def __call__(self, env, t_end, inputs):
        # inputs: array of shape [n_channels] with values in [0, 1]
        rates = np.array(inputs) * self.max_rate  # Hz

        dt = 1.0  # ms
        n_steps = int(t_end / dt)
        array = np.zeros((n_steps, self.n_channels))

        for ch, rate in enumerate(rates):
            if rate > 0:
                interval = 1000.0 / rate  # ms between pulses
                pulse_times = np.arange(0, t_end, interval)
                pulse_indices = (pulse_times / dt).astype(int)
                pulse_indices = pulse_indices[pulse_indices < n_steps]
                array[pulse_indices, ch] = 1.0

        # Transform channel inputs to per-neuron stimulus via IO
        cell_stim = env.io.cell_stimulus(
            env.model.stimulus_coordinates(env.system.neuron_coordinates),
            array,
        )
        return Stimulus(array=cell_stim, dt=dt)
```

## Gymnasium integration

When using livn as an RL environment, the encoding maps the agent's actions to neural stimulation:

```python
class ActionEncoding(Encoding):
    """Map RL actions to neural stimulus"""

    @property
    def input_space(self):
        import gymnasium
        return gymnasium.spaces.Box(low=0.0, high=1.0, shape=(16,))

    def __call__(self, env, t_end, inputs):
        # inputs = agent's action vector
        ...
```

The `input_space` property defines the Gymnasium-compatible action space for RL agents.
