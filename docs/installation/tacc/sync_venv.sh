#!/usr/bin/env bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --job-name=sync-venv

if [ $# -lt 2 ]; then
    echo "Usage: sbatch sync_venv.sh <absolute_project_directory> <system>"
    echo "  <system> must be either 'vista' or 'frontera'"
    echo "Example: sbatch sync_venv.sh \$(realpath .) vista"
    echo "Or: sbatch sync_venv.sh /absolute/path/to/project frontera"
    exit 1
fi

PROJECT_DIRECTORY="$1"
SYSTEM="$2"

# Validate system argument
if [ "$SYSTEM" != "vista" ] && [ "$SYSTEM" != "frontera" ]; then
    echo "Error: System must be either 'vista' or 'frontera', got: '$SYSTEM'"
    exit 1
fi

if [ ! -d "$PROJECT_DIRECTORY" ]; then
    echo "Error: Directory '$PROJECT_DIRECTORY' does not exist"
    exit 1
fi

if [ ! -f "$PROJECT_DIRECTORY/pyproject.toml" ]; then
    echo "Error: No pyproject.toml found in '$PROJECT_DIRECTORY'"
    exit 1
fi

echo "Using project directory: $PROJECT_DIRECTORY"
echo "Using system: $SYSTEM"


if [ "$SYSTEM" == "vista" ]; then
    module load gcc/13.2.0 openmpi/5.0.5 phdf5/1.14.4 python3_mpi/3.11.8 cuda/12.5
elif [ "$SYSTEM" == "frontera" ]; then
    module load xalt/2.10.34 impi/21.9.0 intel/23.1.0 cmake/3.31.5
fi

#
# WORKER VENV
#

# retrieve source files
cd /tmp

cp $PROJECT_DIRECTORY/.python-version .
cp $PROJECT_DIRECTORY/README.md .
cp $PROJECT_DIRECTORY/pyproject.toml .
cp $PROJECT_DIRECTORY/uv.lock .

# Automatically find and copy workspace packages (directories with pyproject.toml)
for dir in $PROJECT_DIRECTORY/*/; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        if [ -f "$dir/pyproject.toml" ]; then
            echo "Found workspace package: $dir_name"
            mkdir -p "$dir_name"
            cp "$dir/pyproject.toml" "./$dir_name/"
            # Copy README if it exists
            if [ -f "$dir/README.md" ]; then
                cp "$dir/README.md" "./$dir_name/"
            fi
        fi
    fi
done

# env
if [ "$SYSTEM" == "vista" ]; then
    export UV_PYTHON=/opt/apps/gcc14/cuda12/openmpi5/python3_mpi/3.11.8/bin/python3.11
    export NOSWITCHERROR=1
    export NVCOMPILER_NOSWITCHERROR=1
    export CC=mpicc
    export CXX=mpicxx
elif [ "$SYSTEM" == "frontera" ]; then
    tar -xf $SCRATCH/venvs/venv.tar

    export UV_NO_BINARY_PACKAGE="mpi4py"

    export MPI_C_COMPILER=/opt/intel/oneapi/mpi/2021.9.0/bin/mpicc
    export MPI_CXX_COMPILER=/opt/intel/oneapi/mpi/2021.9.0/bin/mpicxx
    export MPI_Fortran_COMPILER=/opt/intel/oneapi/mpi/2021.9.0/bin/mpifc

    export CC=/opt/intel/oneapi/mpi/2021.9.0/bin/mpicc
    export CXX=/opt/intel/oneapi/mpi/2021.9.0/bin/mpicxx

    export CFLAGS="-fveclib=none"

    export PATH=/tmp/phdf5/install/HDF_Group/HDF5/1.14.5/bin:$PATH
    export LD_LIBRARY_PATH=/tmp/phdf5/install/HDF_Group/HDF5/1.14.5/lib:$LD_LIBRARY_PATH
    export HDF5_MPI="ON"
    export HDF5_DIR="/tmp/phdf5/install/HDF_Group/HDF5/1.14.5/"

    export UV_PYTHON_INSTALL_DIR=/tmp/python
fi

rm -rf .venv

# sync project dependencies
uv sync --all-packages --all-groups --all-extras

# optional dependencies
uv add rich globus-sdk uncertainties seaborn matplotlib

mkdir -p $SCRATCH/venvs
if [ "$SYSTEM" == "vista" ]; then
    tar -cf $SCRATCH/venvs/venv.tar .venv
elif [ "$SYSTEM" == "frontera" ]; then
    tar -cf $SCRATCH/venvs/venv.tar .venv python phdf5
fi


#
# GLOBAL VENV
#

if [ "$SYSTEM" == "frontera" ]; then
    export PATH=$SCRATCH/venvs/phdf5/install/HDF_Group/HDF5/1.14.5/bin:$PATH
    export LD_LIBRARY_PATH=$SCRATCH/venvs/phdf5/install/HDF_Group/HDF5/1.14.5/lib:$LD_LIBRARY_PATH
    export UV_PYTHON_INSTALL_DIR=$SCRATCH/uv/python
fi

if [ ! -d "$SCRATCH/venv" ]; then
    uv venv $SCRATCH/venv
fi

. $SCRATCH/venv/bin/activate

uv sync --all-packages --all-groups --all-extras --active