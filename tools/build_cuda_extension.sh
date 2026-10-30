#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
invocation_directory="$PWD"

# NVIDIA HPC SDK configuration.
NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}"
NVFORTRAN="${NVHPC_ROOT}/compilers/bin/nvfortran"

if [[ ! -x "$NVFORTRAN" ]]; then
    echo "nvfortran was not found at: $NVFORTRAN" >&2
    exit 1
fi

# Ensure the active Python environment contains nanobind.
if ! python -c "import nanobind" >/dev/null 2>&1; then
    echo "nanobind is unavailable in the active python environment" >&2
    echo "Active Python: $(command -v python)" >&2
    exit 1
fi

NANOBIND_ROOT="$(
    python -c \
        'import nanobind, pathlib; print(pathlib.Path(nanobind.__file__).parent)'
)"
NANOBIND_INCLUDE="$(
    python -m nanobind --include_dir
)"

NANOBIND_SOURCE="${NANOBIND_ROOT}/src/nb_combined.cpp"
ROBIN_MAP_INCLUDE="${NANOBIND_ROOT}/ext/robin_map/include"

PYTHON_INCLUDE="$(
    python -c \
        'import sysconfig; print(sysconfig.get_paths()["include"])'    
)"

EXTENSION_SUFFIX="$(
    python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'
)"

CUDA_ARCH="${CUDA_ARCH:-cc86}"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-cuda13.3}"

default_build_directory="build/cuda/python"
default_extension_path="src/fortran_nlp/_cuda${EXTENSION_SUFFIX}"

if [[ $# -ge 2 ]]; then
    output_directory="$1"
    output_filename="$2"
    private_directory="${3:-build/cuda/python}"

    if [[ "$output_directory" != /* ]]; then
        output_directory="$invocation_directory/$output_directory"
    fi

    if [[ "$private_directory" != /* ]]; then
        private_directory="$invocation_directory/$private_directory"
    fi

    extension_path="$output_directory/$output_filename"
    build_directory="$private_directory"
else
    extension_path="$project_root/$default_extension_path"
    build_directory="$project_root/$default_build_directory"
fi

mkdir -p "$build_directory"
mkdir -p "$(dirname "$extension_path")"

cd "$project_root"

rm -f \
    "$build_directory/fnlp_lexical_cuda_core.mod" \
    "$build_directory/fnlp_lexical_cuda_c_api.mod" \
    "$build_directory/lexical_cuda_core.o" \
    "$build_directory/lexical_cuda_c_api.o" \
    "$build_directory/cuda_bindings.o" \
    "$build_directory/nb_combined.o" \
    "$extension_path"

echo "Building CUDA Fortran lexical core..."

"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -O3 \
    -fPIC \
    src/native/cuda/lexical/lexical_cuda_core.cuf \
    -module "$build_directory" \
    -o "$build_directory/lexical_cuda_core.o"

echo "Building CUDA FOrtran C ABI..."

"$NVFORTRAN" \
    -c \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -O3 \
    -fPIC \
    src/native/cuda/lexical/lexical_cuda_c_api.cuf \
    -I "$build_directory" \
    -module "$build_directory" \
    -o "$build_directory/lexical_cuda_c_api.o"

echo "Building nanobind CUDA adapter with g++..."

g++ \
    -c \
    -std=c++17 \
    -O3 \
    -fPIC \
    -Wall \
    -Wextra \
    -Werror \
    -I "$NANOBIND_INCLUDE" \
    -I "$PYTHON_INCLUDE" \
    -I src/native/include \
    src/native/bindings/cuda_bindings.cpp \
    -o "$build_directory/cuda_bindings.o"

echo "Building nanobind runtime with g++..."

g++ \
    -c \
    -std=c++17 \
    -O3 \
    -fPIC \
    -fvisibility=hidden \
    -I "$NANOBIND_INCLUDE" \
    -I "$ROBIN_MAP_INCLUDE" \
    -I "$PYTHON_INCLUDE" \
    "$NANOBIND_SOURCE" \
    -o "$build_directory/nb_combined.o"

echo "Linking Python CUDA extension..."

"$NVFORTRAN" \
    -shared \
    -cuda \
    -gpu="${CUDA_ARCH},${CUDA_TOOLKIT}" \
    -Mnomain \
    "$build_directory/lexical_cuda_core.o" \
    "$build_directory/lexical_cuda_c_api.o" \
    "$build_directory/cuda_bindings.o" \
    "$build_directory/nb_combined.o" \
    -lstdc++ \
    -o "$extension_path"

echo "Built CUDA extension:"
echo "  $extension_path"

test -f "$extension_path"

echo "CUDA Python extension build: PASS"
echo "  $extension_path"