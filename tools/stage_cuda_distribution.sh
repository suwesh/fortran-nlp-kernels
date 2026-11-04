#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage_directory="${1:-$project_root/build/cuda_distribution}"

if [[ "$stage_directory" != /* ]]; then
    stage_directory="$project_root/$stage_directory"
fi

case "$stage_directory" in
    "$project_root/build/"*)
        ;;
    *)
        echo "Refusing to stage distribution outside of build directory: $stage_directory" >&2
        echo "Pass a path under $project_root/build/." >&2
        exit 1
        ;;
esac

required_files=(
    "packaging/cuda/pyproject.toml"
    "packaging/cuda/meson.build"
    "packaging/cuda/meson_options.txt"
    "packaging/cuda/README.md"
    "LICENSE"
    "src/native/cpu/vector_search/topk_core.f90"
    "src/native/cuda/lexical/lexical_cuda_c_api.cuf"
    "src/native/cuda/lexical/lexical_cuda_core.cuf"
    "src/native/cuda/vector_search/vector_search_cublas.cuf"
    "src/native/cuda/vector_search/vector_search_cuda_c_api.cuf"
    "src/native/cuda/vector_search/vector_search_cuda_core.cuf"
    "src/native/include/fortran_nlp_cuda_lexical.h"
    "src/native/include/fortran_nlp_cuda_vector_search.h"
    "src/native/bindings/cuda_bindings.cpp"
    "tools/build_cuda_extension.sh"
)

for relative_path in "${required_files[@]}"; do
    if [[ ! -f "$project_root/$relative_path" ]]; then
        echo "Required CUDA companion file not found: $relative_path" >&2
        exit 1
    fi
done

rm -rf "$stage_directory"

mkdir -p \
    "$stage_directory/src/native/cpu/vector_search" \
    "$stage_directory/src/native/cuda/lexical" \
    "$stage_directory/src/native/cuda/vector_search" \
    "$stage_directory/src/native/include" \
    "$stage_directory/src/native/bindings" \
    "$stage_directory/tools"

cp "$project_root/packaging/cuda/pyproject.toml" \
    "$stage_directory/pyproject.toml"

cp "$project_root/packaging/cuda/meson.build" \
    "$stage_directory/meson.build"

cp "$project_root/packaging/cuda/meson_options.txt" \
    "$stage_directory/meson_options.txt"

cp "$project_root/packaging/cuda/README.md" \
    "$stage_directory/README.md"

cp "$project_root/LICENSE" \
    "$stage_directory/LICENSE"

cp "$project_root/src/native/cpu/vector_search/topk_core.f90" \
    "$stage_directory/src/native/cpu/vector_search/topk_core.f90"

cp "$project_root/src/native/cuda/lexical/lexical_cuda_c_api.cuf" \
    "$stage_directory/src/native/cuda/lexical/lexical_cuda_c_api.cuf"

cp "$project_root/src/native/cuda/lexical/lexical_cuda_core.cuf" \
    "$stage_directory/src/native/cuda/lexical/lexical_cuda_core.cuf"

cp "$project_root/src/native/cuda/vector_search/vector_search_cublas.cuf" \
    "$stage_directory/src/native/cuda/vector_search/vector_search_cublas.cuf"

cp "$project_root/src/native/cuda/vector_search/vector_search_cuda_c_api.cuf" \
    "$stage_directory/src/native/cuda/vector_search/vector_search_cuda_c_api.cuf"

cp "$project_root/src/native/cuda/vector_search/vector_search_cuda_core.cuf" \
    "$stage_directory/src/native/cuda/vector_search/vector_search_cuda_core.cuf"

cp "$project_root/src/native/include/fortran_nlp_cuda_lexical.h" \
    "$stage_directory/src/native/include/fortran_nlp_cuda_lexical.h"

cp "$project_root/src/native/include/fortran_nlp_cuda_vector_search.h" \
    "$stage_directory/src/native/include/fortran_nlp_cuda_vector_search.h"

cp "$project_root/src/native/bindings/cuda_bindings.cpp" \
    "$stage_directory/src/native/bindings/cuda_bindings.cpp"

cp "$project_root/tools/build_cuda_extension.sh" \
    "$stage_directory/tools/build_cuda_extension.sh"

chmod +x "$stage_directory/tools/build_cuda_extension.sh"

expected_staged_files=(
    "pyproject.toml"
    "meson.build"
    "meson_options.txt"
    "README.md"
    "LICENSE"
    "src/native/cpu/vector_search/topk_core.f90"
    "src/native/cuda/lexical/lexical_cuda_c_api.cuf"
    "src/native/cuda/lexical/lexical_cuda_core.cuf"
    "src/native/cuda/vector_search/vector_search_cublas.cuf"
    "src/native/cuda/vector_search/vector_search_cuda_c_api.cuf"
    "src/native/cuda/vector_search/vector_search_cuda_core.cuf"
    "src/native/include/fortran_nlp_cuda_lexical.h"
    "src/native/include/fortran_nlp_cuda_vector_search.h"
    "src/native/bindings/cuda_bindings.cpp"
    "tools/build_cuda_extension.sh"
)

for relative_path in "${expected_staged_files[@]}"; do
    if [[ ! -f "$stage_directory/$relative_path" ]]; then
        echo "Expected CUDA companion staged file not found: $relative_path" >&2
        exit 1
    fi
done

staged_file_count="$(find "$stage_directory" -type f | wc -l)"
expected_file_count="${#expected_staged_files[@]}"

if [[ "$staged_file_count" -ne "$expected_file_count" ]]; then
    echo "CUDA companion staging contains unexpected number of staged files: $staged_file_count (expected $expected_file_count)" >&2
    find "$stage_directory" -type f -printf '  %P\n' | sort  >&2
    exit 1
fi

echo "CUDA companion distribution staged successfully to: $stage_directory"
echo "Staged files:"
find "$stage_directory" -type f -printf '  %P\n' | sort
