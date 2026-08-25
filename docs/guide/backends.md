# Backends

livn supports three simulation backends, each providing a full implementation of the [`Env`](/guide/concepts/env) protocol. Select a backend via the `LIVN_BACKEND` environment variable **before** importing livn:

```sh
export LIVN_BACKEND=brian2     # requires livn[brian2] dependencies
export LIVN_BACKEND=diffrax    # requires livn[diffrax] dependencies
export LIVN_BACKEND=neuron     # requires livn[neuron] dependencies and MPI
```

When no `LIVN_BACKEND` is set, livn uses a neutral default backend that provides the full `Env` interface without running any simulation. This is useful for working with systems, I/O, and datasets without installing a simulation engine. To run actual simulations, set one of the backends above.

All backends share the same user-facing API - you write your simulation code once and switch backends by changing the environment variable.

## brian2

[brian2](https://brian2.readthedocs.io/) is a lightweight backend suitable for rapid prototyping and small-to-medium systems. It models neurons as point processes using brian2's equation-based description language.

**Strengths:**
- Fast setup, no external system libraries needed
- Good for systems up to ~1,000 neurons on a single machine
- Runs the [GLIF](/models/glif) point neuron (levels 1–5, the `hard` mechanism) and [Izhikevich](/models/izhikevich)

```python
import os
os.environ["LIVN_BACKEND"] = "brian2"

from livn import make

env = make("EI")
it, t, iv, v, *_ = env.run(100)
```

### GSL integration

By default, brian2 uses the forward Euler method with a small timestep (0.005 ms) for numerical integration. For improved accuracy and performance, you can enable the [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) adaptive solver by setting the `LIVN_USE_LIBGSL` environment variable:

```sh
export LIVN_USE_LIBGSL=1
```

When enabled, the integrator switches to `gsl_rkf45` (adaptive Runge-Kutta-Fehlberg 4(5)) with a base timestep of 0.025 ms. The adaptive method automatically adjusts step sizes to maintain accuracy, allowing a larger base timestep while preserving numerical stability for stiff biophysical equations.

This requires GSL to be installed on your system:

```sh
# Ubuntu/Debian
sudo apt install libgsl-dev

# macOS
brew install gsl
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

env = make("EI")
# Gradients through the simulation are now available
```

Under this backend the [`Run`](/guide/concepts/env#running-a-simulation) returned by `env.run()` is a registered pytree, so it can cross a `jit`, `vmap` or `grad` boundary as a return value. Its arrays are the leaves while `t0`, `duration` and `dt` are static metadata and must be concrete values rather than tracers.

### Which model axes are available where

[GLIF](/models/glif) runs on both diffrax and brian2, but not every axis crosses:

| | diffrax | brian2 |
|---|---|---|
| levels 1–5 | yes | yes |
| `mechanism="hard"` | yes | yes |
| `mechanism="escape"` | yes | — needs the event-driven solver |
| gradients through spike times | yes | no |
| `num_samples` batching, membrane diffusion | yes | — |

Asking for the escape mechanism on brian2 raises rather than silently falling back.

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

env = make("EI")
```

### CoreNEURON

To use CoreNEURON set:

```sh
export LIVN_CORENEURON=1
```

CoreNEURON can speed up the integration but it comes with the following limitations:

- Voltage and membrane current cannot be recorded. `Vector.record` on a range variable is not carried back across the transfer, so a trace would return as a single sample of the initial state. `record_voltage()` and `record_membrane_current()` therefore raise an error.

- A stimulus is handed over before the solve, not computed during it. One run holds the entire input so delivering it whole may raise a `MemoryError`.

- CoreNEURON is a different implementation of the same equations. Expect around 99% of spikes to be identical and the rest to move by a fraction of a millisecond. Nothing recorded under one solver will reproduce bit-for-bit under the other, so do not mix them within a study, and re-fit rather than carry parameters across.


## Comparison

| Feature | brian2 | Diffrax | NEURON |
|---------|--------|---------|--------|
| Differentiable | No | **Yes** | No |
| GPU support | No | **Yes** | No |
| Multi-compartment models | No | **Yes** | **Yes** |
| Built-in opsins | No | **Yes** (RhO3c) | **Yes** (RhO3c, RhO6c) |
| MPI parallelism | No | No | **Yes** |
| Setup complexity | Low | Medium | High |
| Ideal scale | ≤1,000 neurons | ≤10,000 neurons | ≤millions |
