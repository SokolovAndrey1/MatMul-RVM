#!/bin/bash

# Project dir
CURRENT_DIR=$PWD

# Project build settings
ENABLE_TEST="ON" # ON or OFF
BUILD_TYPE="Release" # Release or Debug
BUILD_FOLDER="_build"

# Path to C and C++ compilers
C_COMPILER_PATH="$CURRENT_DIR/tools/gcc/bin/riscv64-unknown-linux-gnu-gcc"
CXX_COMPILER_PATH="$CURRENT_DIR/tools/gcc/bin/riscv64-unknown-linux-gnu-g++"

# Clear build folder
# Comment if not needed
rm -rf $BUILD_FOLDER

# Confuigure project
cmake CMakeLists.txt \
 -DENABLE_TEST=$ENABLE_TEST \
 -B"$BUILD_FOLDER" \
 -DCMAKE_C_COMPILER=$C_COMPILER_PATH \
 -DCMAKE_CXX_COMPILER=$CXX_COMPILER_PATH \
 -DBUILD_TYPE=$BUILD_TYPE \
 -DBUILD_STATIC=ON

# Build project
cmake --build _build
