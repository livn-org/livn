# Datasets

livn provides curated datasets hosted on [Hugging Face](https://huggingface.co/datasets/livn-org/livn) for training and evaluation. Each dataset consists of simulation recordings from a predefined [system](/guide/concepts/system), capturing spike times, voltage traces, and membrane currents under varying stimulation conditions.

## Available datasets

Each data sample consists of a 5-second simulation under varying feature inputs.

| System | Neurons (exc./inh.) | Train samples | Test samples |
|--------|---------------------|---------------|--------------|
| S1 | 10 (8/2) | 50,000 | 1,000 |
| S2 | 100 (80/20) | 50,000 | 1,000 |
| S3 | 1,000 (800/200) | 5,000 | 100 |
| S4 | 10,000 (8,000/2,000) | 500 | 50 |

## Loading a dataset

```python
from datasets import load_dataset

dataset = load_dataset("livn-org/livn", name="S2")
sample = dataset["train"][0]

# Each sample contains:
# - trial_it: spike neuron IDs per trial
# - trial_t: spike times per trial
# - trial_iv: voltage recording neuron IDs per trial
# - trial_vv: voltage traces per trial
```

## Observing data through an IO device

The raw dataset contains neuron-level recordings identified by neuron IDs (GIDs). To observe the data as it would appear in a real experiment, apply an [IO transformation](/guide/concepts/io):

```python
from livn.io import MEA

mea = MEA.from_directory("./systems/graphs/S2")
cit, ct = mea.channel_recording(system.neuron_coordinates, it, t)

# cit and ct are dicts mapping channel_id -> recorded spikes
print("Spikes at channel 0:", ct[0])
```

See the [Using the Dataset](/examples/dataset) example for a complete walkthrough.

## Generating your own datasets

For advanced users who want to generate datasets with custom systems, models, or stimulation protocols, see the [Generating Datasets](/guide/systems/sampling) guide.
