# Generating Datasets

Once you have a [generated](/systems/generate) and [tuned](/systems/tuning) system, you can produce datasets of simulation recordings at scale. These datasets capture the stimulus-response dynamics of your system and can serve as training data for machine learning models.

::: tip Prerequisites
This section requires the `systems/` subpackage and its dependencies (`uv sync --package systems`). Familiarity with [Encoding](/guide/concepts/encoding), [Decoding](/guide/concepts/decoding), and [Stimulus](/guide/concepts/stimulus) is assumed.
:::

## Overview

Dataset generation follows a straightforward pipeline:

1. **Configure** the system, encoding, and decoding
2. **Sample** by running many simulations with varying inputs
3. **Merge** the individual samples into a unified dataset
4. **Publish** (optionally) to Hugging Face Hub

Each sample in the dataset is a single simulation run - typically 5 seconds of physical time - under a specific stimulus configuration. The resulting dataset captures the full range of the system's dynamic responses.

## Quick start

Via the CLI:

```sh
livn systems sample \
    system=./systems/graphs/EI \
    decoding='["systems.sample.Raw", {"duration": 5000}]' \
    samples=1000 \
    output_directory=./my_dataset \
    --launch

# Merge individual samples into a Hugging Face Dataset
livn systems sample \
    system=./systems/graphs/EI \
    output_directory=./my_dataset \
    --merge
```

Or equivalently in Python:

```python
from machinable import get

sampler = get("sample", {
    "system": "./systems/graphs/EI",
    "decoding": ("systems.sample.Raw", {"duration": 5000}),
    "samples": 1000,
    "output_directory": "./my_dataset",
})
sampler.launch()
sampler.merge()
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `system` | `./systems/graphs/EI` | Path to the system |
| `model` | `None` | Model class (None = system default) |
| `io` | `None` | IO device (None = the system's own array) |
| `selection` | `None` | Subselection of the system to simulate |
| `params` | `None` | Parameter overrides (None = the system's stored parameters) |
| `samples` | `100` | Number of samples to generate, or a `(train, test)` split |
| `noise` | `True` | Enable background noise |
| `inputs` | `None` | Design deciding what each sample is run with (None = the sample index) |
| `encoding` | `systems.sample.WithoutInput` | Encoding class (dotted path) |
| `decoding` | `("systems.sample.Raw", {"duration": 31000})` | Decoding class, which also sets the duration of each sample |
| `output_directory` | `None` | Where to save individual samples (None = the run's own `samples/`) |
| `nprocs_per_worker` | `1` | MPI ranks per simulation worker |
| `ranks` | `-1` | MPI ranks to request when launching (-1 reads `MPI_RANKS`) |
| `nodes` | `None` | Nodes to request when launching |

### Custom encoding

By default, samples are generated with no external stimulus (spontaneous activity). To use custom stimulation patterns, specify an [Encoding](/guide/concepts/encoding) class by its dotted import path:

```sh
livn systems sample encoding=my_module.MyEncoding --launch
```

### Custom decoding

The default decoding (`Raw`) records spikes, voltages, and membrane currents for 31 s. To customize, use:

```sh
livn systems sample decoding='["systems.sample.Raw", {"duration": 5000, "voltages": false}]' --launch
livn systems sample decoding='["my_module.MyDecoding", {"duration": 5000}]' --launch
```

## Running at scale

Dataset generation is parallelized via MPI using livn's `DistributedEnv`. To run with MPI, prepend the `interface.remotes.mpi` execution module:

```sh
livn systems interface.remotes.mpi sample \
    system=./systems/graphs/EI \
    output_directory=./my_dataset \
    **resources='{"--n": 32}' \
    --launch
```

On Slurm clusters, use the `slurm` (or `tacc` for TACC systems) execution module:

```sh
livn systems slurm sample \
    system=./systems/graphs/EI \
    output_directory=./my_dataset \
    **resources='{"--nodes": 2, "--ntasks-per-node": 56, "-p": "normal", "-t": "4:00:00"}' \
    --launch
```

See the [machinable execution docs](https://machinable.org/guide/execution) for details.

Or size the job from the sampler's own configuration:

```sh
livn systems slurm sample \
    system=./systems/graphs/EI \
    nodes=2 ranks=56 \
    **resources='{"-p": "normal", "-t": "4:00:00"}' \
    --launch
```

Note that jobs are resumable in that only missing inputs are generated.

### Work distribution

With `N` MPI ranks and `nprocs_per_worker = P`:
- 1 rank is the controller
- `(N - 1) / P` workers run simulations in parallel
- Each worker handles the full simulation for one sample at a time

For large systems (a full culture, or CA1), use `nprocs_per_worker > 1` so that each simulation is itself parallelized across multiple ranks.

## Merging samples

After generation, merge individual samples into a structured dataset:

```sh
livn systems sample output_directory=./my_dataset --merge
```

Or in Python:

```python
sampler.merge(include_voltage=False)  # omit voltage traces to save space
```

This creates a [Hugging Face Dataset](https://huggingface.co/docs/datasets/) with train/test splits. 

## Publishing

Upload the merged dataset to Hugging Face Hub:

```python
sampler.publish(repo_id="my-org/my-dataset")
```

## Using generated datasets

### Loading

```python
from datasets import load_dataset

dataset = load_dataset("livn-org/livn", name="EI")
sample = dataset["train"][0]

# Access spike data
it = sample["trial_it"][0]  # neuron IDs
t = sample["trial_t"][0]    # spike times
```

### Observing through an IO device

Apply an [IO transformation](/guide/concepts/io) to see the data as it would appear in a real experiment:

```python
from livn.io import MEA, electrode_array_coordinates_for_area
from livn.system import System

system = System("./systems/graphs/EI")
(xmin, ymin, _), (xmax, ymax, _) = system.bounding_box
mea = MEA(electrode_array_coordinates_for_area(200, ((xmin, ymin), (xmax, ymax))))

cit, ct = mea.channel_recording(system.neuron_coordinates, it, t)
print("Channel 0 spikes:", ct[0])
```

### As an RL replay buffer

The generated datasets can bootstrap off-policy RL agents:

```python
# Load dataset as replay buffer
for sample in dataset["train"]:
    state = sample["trial_it"][0]
    # ... process for RL training
```

## Predefined datasets

livn publishes datasets for all standard systems on [Hugging Face](https://huggingface.co/datasets/livn-org/livn). See [Datasets](/systems/datasets) for the full listing.