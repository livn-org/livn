# Installation on DeltaAI

'''
module purge
module load default
module load PrgEnv-gnu
module load cray-mpich/8.1.33
module load cray-hdf5-parallel/1.14.3.7
module load cmake/3.30.2
'''

'''
echo "HDF5 Directory is: $HDF5_DIR"
export HDF5_ROOT="$HDF5_DIR"
export HDF5_USE_SHLIB=yes
export HDF5_HL_LIBRARIES="$HDF5_DIR/lib/libhdf5_hl.so"
'''

Then follow the [standard instructions](../installation/)
