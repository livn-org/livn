# Installation on DeltaAI

[DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/prog-env.html#gpudirect-for-mpi-cuda) provides a preconfigured MPI / parallel-HDF5 environment via the Cray Programming Environment.

::: tip
Ensure that you are using the latest version of uv via `uv self update`. To ensure that packages are linking against the latest packages, run `uv cache clean`.
:::

## Modules

DeltaAI provides the required dependencies through the following modules:

```sh
module purge
module load default
module load PrgEnv-gnu cray-mpich cray-hdf5-parallel craype-accel-nvidia90 cudatoolkit
module load craype-arm-grace   # sets CRAY_CPU_TARGET for the GH200 ARM nodes
```

`mpi4py`, `h5py`, `neuroh5`, and `NEURON` are then built from source against Cray's MPI and parallel HDF5 via the `cc` / `CC` compiler wrappers. miv-simulator's `configure_mpi.py` auto-detects the Cray PE layout and exports the build-environment variables those packages need; activate a venv first so it can derive the matching Python include path.

```sh
cd /work/hdd/<partition>/$USER/livn
uv venv
source .venv/bin/activate
eval "$(uv run --no-project \
    https://raw.githubusercontent.com/GazzolaLab/MiV-Simulator/main/configure_mpi.py)"
uv sync --all-packages --all-groups --all-extras
```

For more details, see the [standard instructions](../installation/).

::: warning Runtime environment
NEURON loads its MPI library by name at runtime, so it has to be pointed at Cray's MPI shared library. Additionally, libhugetlbfs's ELF segment remapping crashes during `dlopen` of MPI-linked extensions on aarch64 Cray PE, so disable it.

```sh
export MPI_LIB_NRN_PATH="$CRAY_MPICH_DIR/lib/libmpich.so"
export LD_LIBRARY_PATH="$CRAY_MPICH_DIR/lib:${LD_LIBRARY_PATH:-}"
export HUGETLB_ELFMAP=no
```
:::

## Node-local install for fast startup

Lustre / `/work` may perform poorly under thousands of `stat()` calls from `import`s by each rank. Thus, it's recommended to install the venv to node-local `/tmp` (or even `/dev/shm`) on the compute node. You can stage the same venv to many nodes by sharing a single tarball.

### One-time: build a relocatable venv on `/tmp`

Run this on a compute node (so `/tmp` is the node-local fast disk). `UV_LINK_MODE=copy` produces a venv with no hardlinks back to the uv cache, so it can be tarred and moved freely. `UV_PYTHON_INSTALL_DIR` places uv's managed CPython next to the venv so it can be shipped with it.

```sh
export UV_PYTHON_INSTALL_DIR=/tmp/python
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-managed  # ignore /usr/bin/python3.12; install uv's CPython into /tmp/python

cd /work/hdd/<partition>/$USER/livn
uv venv /tmp/venv                # picks the requires-python from pyproject.toml
source /tmp/venv/bin/activate
eval "$(uv run --no-project \
    https://raw.githubusercontent.com/GazzolaLab/MiV-Simulator/main/configure_mpi.py)"

# Build into the active venv at /tmp/venv (project still on /work)
#  --compile-bytecode pre-builds .pyc files
uv sync --all-packages --all-groups --all-extras --compile-bytecode --active

# Pack the venv together with its interpreter for later use
cd /tmp
tar -cf "$SCRATCH/shared/venv.tar" venv python
```

### Per-job: stage to every allocated node

Inside your job script (or in an interactive `salloc`), unpack the tarball once per node before running. Use `srun --ntasks-per-node=1` so the unpack happens on each node, not each rank:

```sh
srun --ntasks-per-node=1 bash -c '
  cd /tmp
  tar -xf "$SCRATCH/shared/venv.tar"
'

# Run from the node-local copy. Data files stay on /work
LIVN_BACKEND=neuron srun --mpi=cray_shasta --export=ALL -n N \
    /tmp/venv/bin/python examples/parallel_simulation.py
```

## Running

```sh
LIVN_BACKEND=neuron srun --mpi=cray_shasta --export=ALL -n N \
    .venv/bin/python examples/parallel_simulation.py
```

Use the path to whichever venv you want, e.g. `.venv/bin/python` for the standard install, or `/tmp/venv/bin/python` on a node-local copy. Make sure the runtime exports from the warning above are set in the shell launching `srun`.
