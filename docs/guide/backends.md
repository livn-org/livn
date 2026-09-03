# Backends

livn supports four simulation backends, each providing a full implementation of the [`Env`](/guide/concepts/env) protocol. Select a backend via the `LIVN_BACKEND` environment variable **before** importing livn:

```sh
export LIVN_BACKEND=native     # ships with livn, no further dependencies
export LIVN_BACKEND=brian2     # requires livn[brian2] dependencies
export LIVN_BACKEND=diffrax    # requires livn[diffrax] dependencies
export LIVN_BACKEND=neuron     # source checkout only, see below
```

When `LIVN_BACKEND` is not set at all, livn uses the `native` backend if its library is available (it is in the wheels published for Linux, macOS and Windows) and otherwise a neutral no-op backend that provides the full `Env` interface without running any simulation. Setting `LIVN_BACKEND=` (empty) selects that no-op backend explicitly, which is useful for working with systems, I/O, and datasets without simulating anything.

All backends share the same user-facing API - you write your simulation code once and switch backends by changing the environment variable.

## native

The default. A self-contained C library (`librcsd`) that implements the cell models of [`ReducedCalciumSomaDendrite`](/models/rcsd) -- the Booth-Rinzel-Kiehn motoneuron and the V1In Renshaw interneuron with their axon chains -- together with the synapses, the STDP rules, the Ornstein-Uhlenbeck background and the RhO3c opsin, using the same numerical scheme as NEURON (fixed-step staggered Crank-Nicolson with cnexp state updates). A run reproduces the NEURON backend step for step, spike for spike, including the seeded noise streams, so results can be mixed across the two.

**Strengths:**
- `pip install livn` is the whole installation: no MPI, no HDF5 library, no compiler
- Streams a stimulus a window at a time, so a policy over a long protocol never has to be materialized, while voltage and membrane current recording stay available
- Per-cell parameters, plasticity, replayable noise and extracellular, current, current-density and optical stimulation

```python
from livn import make

env = make("EI")  # LIVN_BACKEND unset: native is picked up automatically
it, t, iv, v, *_ = env.run(100)
```

It is also the only backend that runs in the browser via the PyEmscripten wheel, so `micropip.install("livn")` under Pyodide 314+ gives a simulating environment with bit-identical results (see [Pyodide](/installation/pyodide)).

It is single-process (no MPI) and does not support the `conductance` stimulus mode. Systems whose cells are not one of the two templates above need the NEURON backend. On a source checkout without a wheel, the library is compiled on first use with the system's C compiler into `~/.cache/livn/native` (`LIVN_CACHE_DIR` overrides the location, `LIVN_NATIVE_LIB` points at a library built by hand).

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

**Installed from a source checkout only.** There is no `livn[neuron]` on PyPI: the backend needs system-level MPI and a parallel HDF5, and `neuroh5` publishes no distributions, which PyPI's ban on git dependencies makes impossible to express. Install the system libraries first ([Installation](/installation/)), then:

```sh
git clone https://github.com/livn-org/livn.git
cd livn
uv sync --group neuron
```

If you have not got that far, `native` is the stand-in and reproduces NEURON's results step for step. Selecting `LIVN_BACKEND=neuron` without the stack installed says as much, with these instructions.

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

| Feature | native | brian2 | Diffrax | NEURON |
|---------|--------|--------|---------|--------|
| Differentiable | No | No | **Yes** | No |
| GPU support | No | No | **Yes** | No |
| Multi-compartment models | **Yes** (the rcsd templates) | No | **Yes** | **Yes** |
| Built-in opsins | **Yes** (RhO3c) | No | **Yes** (RhO3c) | **Yes** (RhO3c, RhO6c) |
| MPI parallelism | No | No | No | **Yes** |
| Setup complexity | None | Low | Medium | High |
| Ideal scale | ≤10,000 neurons | ≤1,000 neurons | ≤10,000 neurons | ≤millions |
