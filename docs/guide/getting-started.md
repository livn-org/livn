# Getting Started

livn is an open-source interactive simulation environment for learning to control in vitro neural networks. It generates synthetic neural data with ground truth at scale, enabling the development and testing of ML models in interactive settings that mimic experimental platforms such as multi-electrode arrays. By providing an extensible platform for developing and benchmarking machine learning models, livn aims to accelerate progress in both ML-driven understanding and engineering of in vitro neural systems and fundamental understanding of computation in biological neural networks.

## Installation

```sh
uv pip install livn
```

or

```sh
git clone https://github.com/livn-org/livn.git
cd livn
uv sync
```

See the [Installation guide](/installation/) for more details, including backend-specific dependencies.

## Running simulations

```python
from livn import make

env = make('S1')

env.record_spikes()
env.record_voltage()

it, t, iv, v, *_ = env.run(100)

print("Initial voltages: ", v[:, 0])
```

## Using the Dataset

```python
from livn.io import MEA
from datasets import load_dataset

system_name = "S1"
dataset = load_dataset("livn-org/livn", name=system_name)
sample = dataset["train"][0]
it = sample["trial_it"][0]
t = sample["trial_t"][0]

# use a multi-electrode array to 'observe' the data
mea = MEA.from_directory("./systems/graphs/" + system_name)
cit, ct = mea.channel_recording(system.neuron_coordinates, it, t)

print("Spikes in channel 0:")
print(ct[0])
```

## Next Steps

- Learn about the available [backends](/guide/backends)
- Explore [examples](/examples/)

