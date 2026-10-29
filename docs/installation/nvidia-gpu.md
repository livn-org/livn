# NVIDIA GPUs (Linux)

The [`native` backend](/guide/backends#native) can run its whole time step on an NVIDIA GPU. It is the same C library, compiled with OpenACC by NVIDIA's `nvc`, so every system, stimulus and recording that runs on the CPU build runs on the GPU build too. It is currently opt-in and Linux-only.

::: tip When it pays off
The GPU build wins by more the larger the network. On one GH200 against the 72 Grace cores of the same node (spikes recorded, wall seconds per simulated second):

| cells | 72-core CPU | GPU | speed-up |
|---|---|---|---|
| 1,000 | 16.7 | 4.9 | 3.4x |
| 11,000 | 93.4 | 14.6 | 6.4x |
| 100,000 | 775 | 111.5 | 7.0x |

Below a few thousand cells a step is dominated by kernel launches, and on a desktop GPU with managed memory the CPU build with [threads](/guide/backends#threads) can be as fast.
:::

## Requirements

- Linux on x86_64 or aarch64, with an NVIDIA GPU of the Pascal generation or newer and its driver
- the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk), which provides `nvc`
- a source checkout of livn so the GPU library is compiled from the C sources on first use (the wheel ships only the CPU library)

## Setup

Put `nvc` on the path and select a memory mode with `LIVN_NATIVE_GPU`:

```sh
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/<version>/compilers/bin:$PATH
export CC=nvc
export LIVN_NATIVE_GPU=managed   # a discrete GPU
# export LIVN_NATIVE_GPU=unified # Grace Hopper, where CPU and GPU share memory
```

- `managed` makes the heap the library allocates CUDA managed memory, which pages between host and device. It works on any supported GPU.
- `unified` relies on memory the GPU can address coherently, as on a Grace Hopper superchip, and moves nothing.

With these set, the first `import` that loads the backend compiles the library into the cache (`~/.cache/livn/native`, `LIVN_CACHE_DIR` overrides it). The GPU build is cached apart from the CPU build, so unsetting `LIVN_NATIVE_GPU` returns to the CPU library without a rebuild. Nothing else changes: `make`, `Env.run` and the recording calls are the same.

Check what was loaded:

```python
from livn.backend.native import _lib

lib = _lib.load()
lib.rcsd_version()   # b'0.1.0+openacc' for the GPU build
_lib.loaded_path()
```

A GPU build that fails to compile raises its compiler error rather than falling back to the CPU library, and a library that does not match `LIVN_NATIVE_GPU` (a CPU build when a GPU one is asked for, or the reverse) is refused when it is loaded.

## What differs from the CPU build

- **Results are close, not bit-identical.** The events that reach one synapse in a step are summed in whatever order the GPU threads arrive, so the last bits of a conductance can differ from the CPU build's. Spike trains agree exactly at first, and once one spike moves by a step the trajectories separate while rates agree. The CPU build is the one that reproduces NEURON to the bit, so keep it for reference comparisons and tests.
- **One process, one GPU.** livn does not assign devices to MPI ranks so every process uses the first GPU unless the OpenACC runtime is told otherwise, for example with `ACC_DEVICE_NUM` set per process.
- **Threads do not apply.** `LIVN_NATIVE_THREADS` and `set_num_threads` have no effect on the GPU build since the GPU splits the work itself.

## Profiling

Nsight Systems releases older than the installed driver record no kernels. The OpenACC runtime's own timer works with any driver, once the library is preloaded so that the profiling runtime can see the OpenACC runtime linked into it:

```sh
LD_PRELOAD=$(python -c "from livn.backend.native import _lib; _lib.load(); print(_lib.loaded_path())") \
NV_ACC_TIME=1 python my_run.py
```

It prints each kernel's launches and device time by source line when the process exits.
