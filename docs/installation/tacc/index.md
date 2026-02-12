# Installation on TACC

## 0. Setup

Install [uv](https://docs.astral.sh/uv/) and add the following to `~/.bashrc`:

```sh
  ##################################################################
  # **** PLACE MODULE COMMANDS HERE and ONLY HERE.              ****
  ##################################################################

  # { system-specific modules here }
  # VISTA:
  module load gcc/13.2.0 openmpi/5.0.5 phdf5/1.14.4 python3_mpi/3.11.8 cuda/12.5
  # FRONTERA:
  module load xalt/2.10.34 impi/21.9.0 intel/23.1.0 cmake/3.31.5

  ###################################################################
  # **** PLACE Environment Variables including PATH here.        ****
  ###################################################################

  export UV_TOOL_DIR=$SCRATCH/uv/tools
  export UV_TOOL_BIN_DIR=$SCRATCH/uv/bin
  export UV_PYTHON_INSTALL_DIR=$SCRATCH/uv/python
  export UV_CACHE_DIR=$SCRATCH/uv/cache

  #####################################################################
  # **** Place any else below.                                     ****
  #####################################################################

  alias python="$SCRATCH/venv/bin/python"
```

## 1. Clone the repo

```sh
git clone https://github.com/livn-org/livn.git
```

## 2. Launch job to create venvs

```sh
# Frontera
sbatch -p development docs/installation/tacc/sync_venv.sh $(realpath .) frontera

# Vista
sbatch -p gg docs/installation/tacc/sync_venv.sh $(realpath .) vista
```

## 3. Use the created venvs

This will create 2 venvs for global and local use:

```sh
# global venv
source $SCRATCH/venv/bin/activate

# local venv for nodes
$SCRATCH/venvs/venv.tar  # extract on worker when running job to use /tmp/.venv
```
