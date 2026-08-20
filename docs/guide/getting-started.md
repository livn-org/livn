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

env = make('EI')

env.record_spikes()
env.record_voltage()

it, t, iv, v, *_ = env.run(100)

print("Initial voltages: ", v[:, 0])
```

## Next Steps

- Learn about the available [backends](/guide/backends)
- Explore [examples](/examples/)

