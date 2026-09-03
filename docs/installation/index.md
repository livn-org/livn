# Installation

::: tip
Ensure that an up-to-date version of [uv](https://docs.astral.sh/uv/) is available

If you are planning to use supercomputing infrastruture, check out our installation scripts and instructions in the menu. 
:::



## Quick Install

Provides the core functionality to get started, including the predefined systems and datasets, and simulation through the built-in `native` backend. May be behind the latest development in the main branch.

```sh
uv pip install livn
```

## Advanced Setup

All backends and whistles! Required if you are interested in generating your own systems (instead of using the pre-defined ones), or scaling up via MPI using NEURON.

::: warning The NEURON backend is a source install
There is no `livn[neuron]` on PyPI as the backend needs compatible MPI, parallel HDF5, and a `neuroh5` build environment (see Prerequisites):

```sh
git clone https://github.com/livn-org/livn.git
cd livn
uv sync --group neuron
```

Prefer the [`native` backend](/guide/backends#native) to get started and move to NEURON when you need MPI across many ranks, full-morphology cells, or other advanced features.
:::

### Prerequisites

For advanced use cases, an MPI and HDF5 installation is required. It is typically easiest to install using your system's package manager. However, if you prefer building parallel HDF5 yourself follow [these instructions](./phdf5.md).

**Linux (Debian) 🐧 / Windows (WSL2) 🪟**

```sh
apt install -y cmake mpich libmpich-dev libhdf5-mpich-dev hdf5-tools
```

**macOS 🍎**

```sh
brew install hdf5-mpi
```

#### neuroh5

`neuroh5` is part of the NEURON stack and is installed by `uv sync --group neuron`. However, generating custom 3D systems with realistic morphology also requires the binaries build as follows:

```sh
git clone https://github.com/iraikov/neuroh5.git
cd neuroh5
cmake .
make
export PATH="/path/to/neuroh5/bin:$PATH"
```

### Installation

```sh
git clone https://github.com/livn-org/livn.git
cd livn
uv sync

# customize as needed, e.g. core + system generation ...
uv sync --package systems 
# or just get the whole smash ...
uv sync --all-packages --all-groups --all-extras
```

::: tip
It is important to ensure that the `mpi4py`/`h5py` package links against the correct MPI version. To force a package rebuild using the currently active MPI installation, use:

```sh
uv pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

To detect possibly missing environment variables, you can use:
```
uv run https://raw.githubusercontent.com/GazzolaLab/MiV-Simulator/refs/heads/main/configure_mpi.py
```
:::

### Resources

- [Paper describing the H5 file format](https://www.biorxiv.org/content/10.1101/2021.11.02.466940v1.full)
- [neuroh5](https://github.com/iraikov/neuroh5)
- A VS Code extension for opening H5 files: `h5web.vscode-h5web`



