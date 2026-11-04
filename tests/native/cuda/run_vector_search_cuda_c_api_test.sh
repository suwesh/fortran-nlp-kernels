#!/usr/bin/env bash

set -euo pipefail

project_root="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"

cd "$project_root"

NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}"
NVFORTRAN="${NVHPC_ROOT}/compilers/bin/nvfortran"
CC="${CC:-gcc}"
CUDA_ARCH="${CUDA_ARCH:-cc86}"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-cuda13.3}"

if [[ ! -x "$NVFORTRAN" ]]; then
    echo "nvfortran was not found at: $NVFORTRAN" >&2
    exit 1
fi

if ! command -v "$CC" >/dev/null 2>&1; then
    echo "C compiler was not found: $CC" >&2
    exit 1
fi

build_directory="build/cuda/vector_search_c_api"
test_executable="$build_directory/test_vector_search_cuda_c_api"

mkdir -p "$build_directory"

rm -f \
    "$build_directory"/*.mod \
    "$build_directory"/*.o \
    "$test_executable"

# Deterministic host-side top-k selection.
"$NVFORTRAN" \
    -c \
    -O3 \
    src/native/cpu/vector_search/topk_core.f90 \
    -module "$build_directory" \
    -o "$build_directory/topk_core.o"

# Handwritten CUDA Fortran scoring backend.
"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -O3 \
    src/native/cuda/vector_search/vector_search_cuda_core.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/vector_search_cuda_core.o"

# cuBLAS SGEMV/SGEMM backend.
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

# bind(C) wrappers for all handwritten CUDA and cuBLAS searches.
"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -cudalib=cublas \
    -O3 \
    src/native/cuda/vector_search/vector_search_cuda_c_api.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/vector_search_cuda_c_api.o"

# Compile the C caller against the public CUDA vector-search header.
"$CC" \
    -std=c11 \
    -O3 \
    -Wall \
    -Wextra \
    -Werror \
    -I src/native/include \
    -c tests/native/cuda/test_vector_search_cuda_c_api.c \
    -o "$build_directory/test_vector_search_cuda_c_api.o"

# Use nvfortran as the final linker so CUDA Fortran runtime libraries
# are included. Link cuBLAS explicitly.
"$NVFORTRAN" \
    -Mnomain \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -cudalib=cublas \
    "$build_directory/topk_core.o" \
    "$build_directory/vector_search_cuda_core.o" \
    "$build_directory/vector_search_cublas.o" \
    "$build_directory/vector_search_cuda_c_api.o" \
    "$build_directory/test_vector_search_cuda_c_api.o" \
    -o "$test_executable"

"$test_executable"
