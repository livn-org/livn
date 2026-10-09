# Standard Systems

livn includes systems that cover a range of scales and biological models, ready to use and carrying their tuned parameters and default models.

::: tip
This section assumes familiarity with the core [concepts](/guide/concepts/env). If you haven't already, read through the Concepts guide first.
:::

## Cultures

The cultures are 2D flat networks of motoneurons, alone or mixed with inhibitory Renshaw cells, on a 128-electrode multi-electrode array (MEA). Each one is tuned to a recording of a real culture with the same composition, and they are the recommended starting point for most users.

```python
from livn import make

env = make("EI")          # the 50/50 culture, tuned
env = make("E-1000")      # the all-excitatory culture, 1000-cell version
```

### The predefined cultures

There are three compositions, each at three sizes:

| Culture | Composition | 2600 cells | 1000 cells | 470 cells |
|---------|-------------|------------|------------|-----------|
| `E` | 100% motoneurons | `E` | `E-1000` | `E-470` |
| `E3I` | 75% motoneurons / 25% Renshaw | `E3I` | `E3I-1000` | `E3I-470` |
| `EI` | 50% motoneurons / 50% Renshaw | `EI` | `EI-1000` | `EI-470` |

| Size | Area | Electrodes | Notes |
|------|------|------------|-------|
| 2600 cells | 1.6 x 3.2 mm | 128 (8 x 16 grid) | the tuned network |
| 1000 cells | 1.0 x 2.0 mm | 40 (4 x 10) | centred cut-out of the full culture |
| 470 cells | 0.7 x 1.4 mm | 24 (4 x 6) | centred cut-out of the full culture |

Common to all of them:

- **Density** is ~500 cells/mm², the cells the array can see; the rest of a real culture enters as background synaptic noise.
- **Electrodes** sit on a 200 µm grid. A channel id means the same electrode in every culture at a given size, so recordings from `E-1000` and `EI-1000` line up channel for channel.
- **Boundaries are periodic**: every cell has a full set of inputs, so there are no edge effects. The smaller sizes keep every cell's number of inputs and shorten the connection range to fit their box.
- **The model** is `ReducedCalciumSomaDendrite`: a reduced spinal motoneuron (soma, dendrite, axon initial segment) with depressing AMPA/NMDA synapses, and a spherical V1 Renshaw interneuron for inhibition.
- **The array records physically.** Each cell's extracellular spike is computed from the model cell's own membrane currents, and a spike counts on a channel when it clears 5 times the recording's noise; only motoneurons within ~50 µm of an electrode are detected.

The smaller sizes are cut-outs of the full culture with the same parameters, not separate fits, so they behave like their 2600-cell parent at rest. Check the table under [Known limitations](#known-limitations) before using them for stimulation experiments.

### What they were tuned to

Each culture was fitted (surrogate-assisted, see [Tuning](/systems/tuning)) against a spike-sorted recording of a real motoneuron culture with the same composition. The fit compares single cells to single cells with the largest sorted unit on each culture channel against the strongest detected model cell on each electrode, through the physical readout above.

Measured on those units, in 20 s windows, with bands taken from the recording's own variability:

- **Firing**: mean rate, the spread of rates across units, the share carried by the fastest 10%, spike-time irregularity (ISI CV), the fastest unit's rate, the share of electrodes with an active unit.
- **Coordination**: pair correlation of units, whether multi-unit events exceed chance, synchrony peak, population timescale, burst rate, branching ratio, Fano factor.
- **Bursts** (all-excitatory culture): how many units join each burst, how many spikes each fires, how spread out their recruitment is, and activity between bursts.
- **Stimulus response**: the fraction of the array answering electrode pulses of four amplitudes (a recruitment curve, 8 pulses per amplitude), with response latency and duration.

The tuned networks reproduce dynamic differences like the fact that the all-excitatory culture fires in network bursts while the two mixed cultures fire asynchronously. 

### Known limitations

| Culture | Matches | Does not match |
|---------|---------|----------------|
| `EI` | every measured feature, including the (weak) stimulus response | Inhibitory activity is unconstrained as different inhibitory strengths fit equally well |
| `E` | burst rate, firing rate, population timescale, average stimulus response | inside a burst: in the culture each unit joins ~70% of bursts, recruited over ~40 ms in a stable order; in the model every cell fires in every burst within ~8 ms. Bursts are therefore too regular (ISI CV) and units somewhat too correlated. Stimulation ignites a burst ~2/3 of the time regardless of amplitude, where the culture's response grows with amplitude |
| `E3I` | resting activity (rate, rate spread, correlation, coordination) | stimulus response: the culture is quiet but excitable, and a pulse recruits most of the array; the model answers only near the electrode (4-7% of the array). Spiking is slightly too regular (ISI CV) |

The smaller sizes inherit these, plus:

| Config | Use for |
|--------|---------|
| `EI-1000` | spontaneous and evoked activity |
| `E-1000`, `E-470`, `E3I-1000`, `EI-470` | spontaneous activity |
| `E3I-470` | spontaneous activity, approximately as it fires faster and more evenly than `E3I` |

See [Generating 2D systems](/systems/generate) for how to create custom cultures.

## Hippocampal system (CA1)

The hippocampal system models the CA1 region of the rodent hippocampus, using 15 distinct cell types with biologically detailed morphologies and connectivity.

This system requires the NEURON backend with MPI and is designed for supercomputer-scale simulations.

```python
import os
os.environ["LIVN_BACKEND"] = "neuron"

from livn.system import NeuroH5System, fetch

system = NeuroH5System(fetch("CA1"))   # downloads once, then reuses
```

## Loading and using systems

```python
from livn import make

env = make("EI")
env = make("runs/bursting/env.json")  # a configured env of your own

env.record_spikes()
env.record_voltage()
it, t, iv, v, *_ = env.run(100)
```

A hosted system is assembled explicitly, since fetching it goes to the network:

```python
from livn.env import Env
from livn.system import NeuroH5System, fetch

env = Env(NeuroH5System(fetch("CA1"))).init()
```

Or take the system on its own:

```python
from livn.system import predefined

system = predefined("EI")        # a Monolayer; ships with livn

print(system.num_neurons)        # 2600
print(system.populations)        # ['EXC', 'INH']
print(system.weight_names)       # tunable weight parameters
print(system.summary())          # neuron and projection counts
```

## The `systems` subpackage

The `systems/` subpackage provides tools for generating, tuning, and sampling custom systems. These tools are available via the `livn systems` CLI:

```sh
livn systems generate_2d --launch   # generate a 2D culture
livn systems tune --launch          # tune synaptic parameters
livn systems sample --launch        # generate a dataset
```

Under the hood, the CLI is powered by [machinable](https://machinable.org), a framework for reproducible computational experiments. You don't need to know much about machinable to use these tools - the CLI handles execution, configuration, and result storage automatically.

## What's next

- [Download datasets](/systems/datasets) - download datasets of the standard systems

If the predefined systems don't match your experimental setup, you can:

- [Generate systems](/systems/generate) - generate cultures with custom populations and connectivity
- [Tune systems](/systems/tuning) - optimize synaptic parameters for target dynamics
- [Generate datasets](/systems/sampling) - produce simulation datasets at scale
