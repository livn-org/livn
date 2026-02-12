# Installation

::: tip
Ensure that an up-to-date version of [uv](https://docs.astral.sh/uv/) is available

If you are planning to use supercomputing infrastruture, check out our installation scripts and instructions in the menu. 
:::



## Quick Install

Provides the core functionality to get started (predefined systems and datasets, JAX/brian2 simulations). May be behind the latest development in the main branch.

```sh
uv pip install livn
```

## Advanced Setup

All backends and whistles! Required if you are interested in generating your own systems (instead of using the pre-defined ones), or scaling up via MPI using NEURON. 

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

#### neuroh5 (optional)

::: details Install if generating 3D morphological systems

If you want to generate custom 3D systems with realistic morphology (not common), you will need to compile `neuroh5`. This is **not required** if you download livn's default systems or if you like to generate custom 2D systems.

```sh
git clone https://github.com/iraikov/neuroh5.git
cd neuroh5
cmake .
make

# add the neuroh5 binaries to your PATH
export PATH="/path/to/neuroh5/bin:$PATH"
```

:::


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
It is important to ensure that the `mpi4py` package links against the correct MPI version. To force a package rebuild using the currently active MPI installation, use:

```sh
uv pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```
:::

**Tests**

You may need the following additional dependencies to run all tests:

```sh
uv pip install pyarrow multiprocess xxhash
```

### Resources

- [Paper describing the H5 file format](https://www.biorxiv.org/content/10.1101/2021.11.02.466940v1.full)
- [h5py](https://docs.h5py.org/en/stable/) and [neuroh5](https://github.com/iraikov/neuroh5)
- A VS Code extension for opening H5 files: `h5web.vscode-h5web`



