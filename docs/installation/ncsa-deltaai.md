# Installation on DeltaAI

[DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/prog-env.html#gpudirect-for-mpi-cuda) provides a preconfigured MPI/PHDF5 environment via the following modules:

::: tip 
Ensure that you are using the latest version of uv via `uv self update`. To ensure that packages are linking against the latest packages, run `uv cache clean`.
:::

```sh

module purge
module load default

module load cce/20.0.0 cray-mpich/9.0.1 cray-hdf5-parallel/1.14.3.7 cray-python/3.11.7 cmake/3.30.2

export UV_PYTHON=/opt/cray/pe/python/3.11.7/bin/python

export HDF5_DIR='/opt/cray/pe/hdf5-parallel/1.14.3.7/cray/20.0'
export HDF5_MPI='ON'
```

With this environment, use `uv sync --all-packages --all-groups --all-extras` as normal to create a venv (e.g. on `/tmp` of the node, or `/work/hdd/<partition>/$USER/livn`). For more details, see the [standard instructions](../installation/).
