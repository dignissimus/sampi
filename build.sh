#!/bin/bash
export CC="${CC:-mpicc}"
export CXX="${CXX:-mpicxx}"
echo "Building with compiler CC=$CC"
echo "Building with compiler CXX=$CXX"
rm -rf build
mkdir build
cd build
cmake ..
make
