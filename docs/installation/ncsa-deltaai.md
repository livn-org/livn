# Installation on DeltaAI

[DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/prog-env.html#gpudirect-for-mpi-cuda) provides a preconfigured MPI/PHDF5 environment via the following modules:

```sh
module purge
module load default

module load PrgEnv-gnu cray-mpich/9.0.1 cray-hdf5-parallel/1.14.3.7 cmake/3.30.2 cray-python/3.11.7

export HDF5_ROOT="$HDF5_DIR"
export HDF5_USE_SHLIB=yes
export HDF5_HL_LIBRARIES="$HDF5_DIR/lib/libhdf5_hl.so"

export MPICC="cc -shared"
export UV_PYTHON=/opt/cray/pe/python/3.11.7/bin/python
```

With this environment, use `uv` as normal to create a venv (e.g. on `/work/hdd/<partition>/$USER/livn`). For more details, see the [standard instructions](../installation/).
