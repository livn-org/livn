# livn

**A testbed for learning to interact with in vitro neural networks**

[![Documentation](https://img.shields.io/badge/docs-livn--org.github.io-blue)](https://livn-org.github.io/livn/)
[![Paper](https://img.shields.io/badge/paper-bioRxiv-green)](https://www.biorxiv.org/content/early/2025/12/19/2025.12.16.694706)

livn is an open-source interactive simulation environment for learning to control in vitro neural networks. It generates synthetic neural data with ground truth at scale, enabling the development and testing of ML models in interactive settings that mimic experimental platforms such as multi-electrode arrays. By providing an extensible platform for developing and benchmarking machine learning models, livn aims to accelerate progress in both ML-driven understanding and engineering of in vitro neural systems and fundamental understanding of computation in biological neural networks.

```sh
pip install livn
```

**[Read the full documentation →](https://livn-org.github.io/livn/)**

## Features

- **Multiple backends** — Run simulations using Brian2, NEURON, or JAX (diffrax) depending on your needs
- **Multi-electrode arrays** — Observe neural activity through realistic MEA configurations with stimulation and recording
- **Differentiable** — Differentiate through the full simulation environment (including IO) when using the JAX backend
- **RL integration** — Standard [Gymnasium](https://gymnasium.farama.org/) interface for reinforcement learning experiments
- **Curated datasets** — Access pre-generated datasets on [Hugging Face](https://huggingface.co/datasets/livn-org) for training and evaluation
- **Parallel execution** — Scale up with MPI-based parallel simulations for large-scale experiments on CPU and GPU

## Quickstart

### Running a simulation

```python
from livn import make

env = make("EI1")

env.apply_model_defaults()
env.record_spikes()
env.record_voltage()

it, t, iv, v, *_ = env.run(100)

print("Initial voltages: ", v[:, 0])
```

### Backends

livn supports three simulation backends:

```sh
export LIVN_BACKEND=brian2     # default; point-neuron models
export LIVN_BACKEND=diffrax    # JAX-based, differentiable
export LIVN_BACKEND=neuron     # MPI-parallel, multi-compartment
```

See the [documentation](https://livn-org.github.io/livn/guide/backends) for details.

## Predefined systems

| System | Neurons | EXC | INH | MEA channels | Description |
|--------|---------|-----|-----|-------------|-------------|
| **EI1** | 10 | 8 | 2 | 1 | Quick prototyping, unit tests |
| **EI2** | 100 | 80 | 20 | 16 | Development, RL experiments |
| **EI3** | 1,000 | 800 | 200 | 64 | Medium-scale experiments |
| **EI4** | 10,000 | 8,000 | 2,000 | 1,024 | Large-scale experiments |

## Documentation

Full documentation at **[livn-org.github.io/livn](https://livn-org.github.io/livn/)** covering:

- [Getting started](https://livn-org.github.io/livn/guide/getting-started) — installation, first simulation, using datasets
- [Concepts](https://livn-org.github.io/livn/guide/concepts/env) — environments, systems, models, IO, encoding, decoding, stimulus
- [Systems](https://livn-org.github.io/livn/guide/systems/) — generating, tuning, and sampling custom systems
- [Examples](https://livn-org.github.io/livn/examples/dataset) — dataset usage, differentiable simulation, reinforcement learning

### Full installation

Clone this repo and install with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --all-packages --all-groups --all-extras
```

## Citation

If you use livn in your research, please cite:

```bibtex
@article{GressmannLIVN2025,
	author = {Gressmann, Frithjof and Raikov, Ivan Georgiev and Pham, Hau Ngoc and Coats, Evan and Soltesz, Ivan and Rauchwerger, Lawrence},
	title = {livn: A testbed for learning to interact with in vitro neural networks},
	elocation-id = {2025.12.16.694706},
	year = {2025},
	doi = {10.64898/2025.12.16.694706},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/12/19/2025.12.16.694706},
	eprint = {https://www.biorxiv.org/content/early/2025/12/19/2025.12.16.694706.full.pdf},
	journal = {bioRxiv}
}
```

## License

MIT, see [LICENSE.txt](LICENSE.txt).
