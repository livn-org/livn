# Decoding

**Decoding** transforms raw simulation recordings into structured, meaningful outputs. Think of it as the post-processing step that extracts information from the neural activity - computing firing rates, extracting LFP signals, detecting avalanches, or simply selecting a time window.

## The `Decoding` base class

All decodings extend `Decoding` and require a `duration` (in ms) that determines how long the simulation runs:

```python
from livn.types import Decoding

class MyDecoding(Decoding):
    duration: int = 1000

    def setup(self, env):
        """Optional: configure the environment before simulation (e.g., enable recordings)"""
        env.record_spikes()

    def __call__(self, env, it, tt, iv, vv, im, mp):
        """Process raw recordings and return structured output"""
        return {"n_spikes": len(tt) if tt is not None else 0}
```

The six arguments to `__call__` correspond to the raw recording outputs from [`env.run()`](/guide/concepts/env):

| Argument | Type | Description |
|----------|------|-------------|
| `it` | `int[]` | Spiking neuron IDs |
| `tt` | `float[]` | Spike times (ms) |
| `iv` | `int[]` | Voltage-recorded neuron IDs |
| `vv` | `float[n, T]` | Voltage traces |
| `im` | `int[]` | Membrane-current-recorded neuron IDs |
| `mp` | `float[n, T]` | Membrane current traces |

## Using decodings

Decodings are passed to `env()` (the callable interface) rather than `env.run()`:

```python
from livn.decoding import MeanFiringRate

result = env(decoding=MeanFiringRate(duration=1000))
# result = {"rate_hz": 3.2, "total_spikes": 320, "n_units": 10, ...}
```

## Built-in decodings

### Slice

Extracts a time window from the recordings. Useful for discarding transient onset responses:

```python
from livn.decoding import Slice

# Keep only activity between 100ms and 500ms
decoding = Slice(start=100, stop=500)
```

### GatherAndMerge

Collects recordings from all MPI ranks and merges them. Essential for distributed simulations where each rank holds a subset of neurons:

```python
from livn.decoding import GatherAndMerge

decoding = GatherAndMerge(
    duration=1000,
    spikes=True,
    voltages=True,
    membrane_currents=False,
)
```

### ChannelRecording

Maps neuron-level recordings to per-electrode-channel recordings using the environment's [IO device](/guide/concepts/io):

```python
from livn.decoding import ChannelRecording

decoding = ChannelRecording(duration=1000)
# Returns: (channel_spike_ids, channel_spike_times, voltage_ids, voltages, channel_ids, potentials)
```

### MeanFiringRate

Computes the population mean firing rate in Hz:

```python
from livn.decoding import MeanFiringRate

result = env(decoding=MeanFiringRate(duration=5000))
# {"rate_hz": 3.1, "total_spikes": 155, "n_units": 10, "duration_s": 5.0}
```

### ActiveFraction

Computes what fraction of neurons are active (fired at least `min_spikes` spikes):

```python
from livn.decoding import ActiveFraction

result = env(decoding=ActiveFraction(duration=5000, min_spikes=1))
# {"active_fraction": 0.8, "active_units": 8, "total_units": 10}
```

### Stability

Detects pathological states (quiescence or runaway firing) by analyzing binned spike rates:

```python
from livn.decoding import Stability

result = env(decoding=Stability(duration=5000))
```

### LFP

Computes local field potential features including band power (delta, theta, alpha, beta, gamma) via Welch power spectral density:

```python
from livn.decoding import LFP

result = env(decoding=LFP(duration=5000))
# Includes downsampled LFP, band powers, and spectral features
```

### AvalancheAnalysis

Detects neuronal avalanches and computes criticality metrics including the branching ratio and power-law exponents:

```python
from livn.decoding import AvalancheAnalysis

result = env(decoding=AvalancheAnalysis(duration=5000))
# Includes branching_ratio, power_law_exponent, mean_size, mean_duration, ...
```

## Composing decodings

The `Pipe` decoding allows sequential composition of multiple decodings:

```python
from livn.decoding import Pipe, Slice, GatherAndMerge

decoding = Pipe(
    duration=5000,
    stages=[
        GatherAndMerge(duration=5000),
        Slice(start=1000, stop=5000),  # discard first second
    ],
)
```

Each stage receives the output of the previous stage, allowing you to build multi-step processing pipelines.

::: tip Advanced usage
For pipelines that need to share intermediate results between stages or carry state across multiple simulation steps, see [Decoding pipelines](/guide/advanced/decoding-pipelines).
:::
