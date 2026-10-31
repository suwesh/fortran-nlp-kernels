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

build_directory="build/cuda/vector_search"
test_executable="$build_directory/test_vector_search_cuda_core"

mkdir -p "$build_directory"

rm -f \
    "$build_directory/fnlp_topk_core.mod" \
    "$build_directory/fnlp_vector_search_core.mod" \
    "$build_directory/fnlp_vector_search_cuda_core.mod" \
    "$build_directory/topk_core.o" \
    "$build_directory/vector_search_core.o" \
    "$build_directory/vector_search_cuda_core.o" \
    "$build_directory/test_vector_search_cuda_core.o" \
    "$test_executable"

# Compile the existing CPU top-k reference module.
"$NVFORTRAN" \
    -c \
    -O3 \
    src/native/cpu/vector_search/topk_core.f90 \
    -module "$build_directory" \
    -o "$build_directory/topk_core.o"

# Compile the existing CPU vector-scoring reference module.
"$NVFORTRAN" \
    -c \
    -O3 \
    src/native/cpu/vector_search/vector_search_core.f90 \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/vector_search_core.o"

# Compile the handwritten CUDA Fortran scoring module.
"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -O3 \
    src/native/cuda/vector_search/vector_search_cuda_core.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/vector_search_cuda_core.o"

# Compile the native CUDA vector-search test program.
"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -O3 \
    tests/native/cuda/test_vector_search_cuda_core.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/test_vector_search_cuda_core.o"

# Link with nvfortran so CUDA Fortran runtime libraries are included.
"$NVFORTRAN" \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    "$build_directory/topk_core.o" \
    "$build_directory/vector_search_core.o" \
    "$build_directory/vector_search_cuda_core.o" \
    "$build_directory/test_vector_search_cuda_core.o" \
    -o "$test_executable"

"$test_executable"
