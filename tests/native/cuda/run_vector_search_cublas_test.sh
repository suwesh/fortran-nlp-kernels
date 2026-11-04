#!/usr/bin/env bash

set -euo pipefail

project_root="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"

cd "$project_root"

NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}"
NVFORTRAN="${NVHPC_ROOT}/compilers/bin/nvfortran"
CUDA_ARCH="${CUDA_ARCH:-cc86}"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-cuda13.3}"

if [[ ! -x "$NVFORTRAN" ]]; then
    echo "nvfortran was not found at: $NVFORTRAN" >&2
    exit 1
fi

build_directory="build/cuda/vector_search_cublas"
test_executable="$build_directory/test_vector_search_cublas"

mkdir -p "$build_directory"

rm -f \
    "$build_directory/fnlp_topk_core.mod" \
    "$build_directory/fnlp_vector_search_core.mod" \
    "$build_directory/fnlp_vector_search_cublas.mod" \
    "$build_directory/topk_core.o" \
    "$build_directory/vector_search_core.o" \
    "$build_directory/vector_search_cublas.o" \
    "$build_directory/test_vector_search_cublas.o" \
    "$test_executable"

# Compile the deterministic CPU top-k module.
"$NVFORTRAN" \
    -c \
    -O3 \
    src/native/cpu/vector_search/topk_core.f90 \
    -module "$build_directory" \
    -o "$build_directory/topk_core.o"

# Compile the CPU serial reference implementation.
"$NVFORTRAN" \
    -c \
    -O3 \
    src/native/cpu/vector_search/vector_search_core.f90 \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/vector_search_core.o"

# Compile the CUDA Fortran cuBLAS implementation.
"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -cudalib=cublas \
    -O3 \
    src/native/cuda/vector_search/vector_search_cublas.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/vector_search_cublas.o"

# Compile the native cuBLAS correctness tests.
"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -cudalib=cublas \
    -O3 \
    tests/native/cuda/test_vector_search_cublas.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/test_vector_search_cublas.o"

# Link with the CUDA Fortran runtime and cuBLAS explicitly.
"$NVFORTRAN" \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -cudalib=cublas \
    "$build_directory/topk_core.o" \
    "$build_directory/vector_search_core.o" \
    "$build_directory/vector_search_cublas.o" \
    "$build_directory/test_vector_search_cublas.o" \
    -o "$test_executable"

"$test_executable"
