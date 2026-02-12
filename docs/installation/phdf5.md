# Building parallel HDF5

::: tip 
This is only required for advanced use cases. [Learn more](../installation/)
:::

If your system or package manager does not provide an suitable MPI-enabled version of HDF5 (phdf5), you may build it from source as follows:

```sh
INSTALL_DIRECTORY=/tmp/phdf5

mkdir -p $INSTALL_DIRECTORY/install
cd $INSTALL_DIRECTORY


wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.5/hdf5-1.14.5.tar.gz
tar -xzf hdf5-1.14.5.tar.gz

wget https://github.com/HDFGroup/hdf5_plugins/releases/download/snapshot-1.14.5/hdf5_plugins-1.14.tar.gz
wget https://github.com/madler/zlib/releases/download/v1.3/zlib-1.3.tar.gz
wget https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.1.6.tar.gz
wget https://github.com/MathisRosenhauer/libaec/releases/download/v1.1.3/libaec-1.1.3.tar.gz

cp hdf5-1.14.5/config/cmake/scripts/* .

# edit HDF5options.cmake and set `DBUILD_TESTING = OFF`
# edit HDF5config.cmake and set `LOCAL_SKIP_TEST = True`

ctest -S HDF5config.cmake,BUILD_GENERATOR=Unix,MPI=1 -C Release -VV -O hdf5.log

cd $INSTALL_DIRECTORY/install
../HDF5-1.14.5-Linux.sh
```

Once build, you can update your environment variables, e.g.

```sh
export PATH=$PHDF5_INSTALL_DIRECTORY/install/HDF5-1.14.5-Linux/HDF_Group/HDF5/1.14.5/bin:$PHDF5_INSTALL_DIRECTORY/install/HDF5-1.14.5-Linux/HDF_Group/HDF5/1.14.5/bin:$PATH
export LD_LIBRARY_PATH=$PHDF5_INSTALL_DIRECTORY/install/HDF5-1.14.5-Linux/HDF_Group/HDF5/1.14.5/lib:$PHDF5_INSTALL_DIRECTORY/install/HDF5-1.14.5-Linux/HDF_Group/HDF5/1.14.5/lib:$LD_LIBRARY_PATH

export HDF5_MPI="ON"
export HDF5_DIR="$PHDF5_INSTALL_DIRECTORY/install/HDF5-1.14.5-Linux/HDF_Group/HDF5/1.14.5/"
```
