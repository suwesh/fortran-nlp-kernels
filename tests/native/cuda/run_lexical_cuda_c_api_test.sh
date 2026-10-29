#!/usr/bin/env bash

set -euo pipefail

project_root="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"

cd "$project_root"

build_directory="build/cuda"
test_executable="$build_directory/test_lexical_cuda_c_api"

mkdir -p "$build_directory"

rm -f \
    "$build_directory/fnlp_lexical_cuda_core.mod" \
    "$build_directory/fnlp_lexical_cuda_c_api.mod" \
    "$build_directory/lexical_cuda_core.o" \
    "$build_directory/lexical_cuda_c_api.o" \
    "$build_directory/test_lexical_cuda_c_api.o" \
    "$test_executable"

# Compile the CUDA Fortran lexical core.
nvfortran \
    -c \
    -cuda \
    -gpu=cc86,cuda13.3 \
    -O3 \
    src/native/cuda/lexical/lexical_cuda_core.cuf \
    -module "$build_directory" \
    -o "$build_directory/lexical_cuda_core.o"

# Compile the CUDA Fortran bind(C) adapter.
nvfortran \
    -c \
    -cuda \
    -gpu=cc86,cuda13.3 \
    -O3 \
    src/native/cuda/lexical/lexical_cuda_c_api.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/lexical_cuda_c_api.o"

# Compile the native C caller against the public CUDA C header.
gcc \
    -c \
    -std=c11 \
    -O2 \
    -Wall \
    -Wextra \
    -Werror \
    -I src/native/include \
    tests/native/cuda/test_lexical_cuda_c_api.c \
    -o "$build_directory/test_lexical_cuda_c_api.o"

# Use nvfortran as the final linker so the CUDA Fortran runtime,
# CUDA runtime, and Fortran runtime libraries are included.
nvfortran \
    -cuda \
    -gpu=cc86,cuda13.3 \
    -Mnomain \
    "$build_directory/lexical_cuda_core.o" \
    "$build_directory/lexical_cuda_c_api.o" \
    "$build_directory/test_lexical_cuda_c_api.o" \
    -o "$test_executable"

"$test_executable"