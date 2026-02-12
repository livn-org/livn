# Backends

livn supports three simulation backends, each providing a full implementation of the [`Env`](/guide/concepts/env) protocol. Select a backend via the `LIVN_BACKEND` environment variable **before** importing livn:

```sh
export LIVN_BACKEND=brian2     # default
export LIVN_BACKEND=diffrax    # requires livn[diffrax] dependencies
export LIVN_BACKEND=neuron     # requires livn[neuron] dependencies and MPI
```

All backends share the same user-facing API - you write your simulation code once and switch backends by changing the environment variable.

## brian2 (default)

[brian2](https://brian2.readthedocs.io/) is the default backend, suitable for rapid prototyping and small-to-medium systems. It models neurons as point processes using brian2's equation-based description language. No additional dependencies are required beyond the base install.

**Strengths:**
- Fast setup, no external system libraries needed
- Good for systems up to ~1,000 neurons on a single machine
- Includes Izhikevich and Leaky Integrate-and-Fire (LIF) models

```python
# brian2 is the default, just import and go
from livn import make

env = make("EI2")
it, t, iv, v, *_ = env.run(100)
```

## Jax

A [JAX](https://jax.readthedocs.io/)-based backend that enables **differentiable simulations** through [Diffrax](https://docs.kidger.site/diffrax/) and [Equinox](https://docs.kidger.site/equinox/). This allows you to compute exact gradients through the simulation and use gradient-based optimization to learn stimulus parameters, decode neural activity, or train surrogate models end-to-end.

**Strengths:**
- End-to-end differentiable: backpropagate through the full simulation
- GPU-accelerated via JAX
- JIT-compiled for fast repeated evaluation

Install with:

```sh
pip install livn[diffrax]
```

```python
import os
os.environ["LIVN_BACKEND"] = "diffrax"

from livn import make
import jax

env = make("EI2")
# Gradients through the simulation are now available
```

See the [Differentiable Simulation](/examples/differentiable) example for a full walkthrough.

## NEURON

The [NEURON](https://www.neuron.yale.edu/neuron/) backend provides high-fidelity, multi-compartment biophysical simulations with MPI-based parallelism. It integrates with the [MiV-Simulator](https://github.com/GazzolaLab/MiV-Simulator) for large-scale network simulations on HPC infrastructure.

**Strengths:**
- Detailed biophysical neuron models (multi-compartment, ion channels, calcium dynamics)
- MPI-parallel: scales to millions of neurons on supercomputers
- Best choice for generating realistic synthetic data

Requires system-level MPI and HDF5 libraries. See [Installation](/installation/) for setup instructions.

```sh
pip install livn[neuron]
```

```python
import os
os.environ["LIVN_BACKEND"] = "neuron"

from livn import make

env = make("EI2")
```

## Comparison

| Feature | brian2 | Diffrax | NEURON |
|---------|--------|---------|--------|
| Differentiable | No | **Yes** | No |
| GPU support | No | **Yes** | No |
| Multi-compartment models | No | **Yes** | **Yes** |
| MPI parallelism | No | No | **Yes** |
| Setup complexity | Low | Medium | High |
| Ideal scale | ≤1,000 neurons | ≤10,000 neurons | ≤millions |
