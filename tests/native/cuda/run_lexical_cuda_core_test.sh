#!/usr/bin/env bash

set -euo pipefail

project_root="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"

cd "$project_root"

build_directory="build/cuda"
test_executable="$build_directory/test_lexical_cuda_core"

mkdir -p "$build_directory"

rm -f \
    "$build_directory/fnlp_lexical_core.mod" \
    "$build_directory/fnlp_lexical_cuda_core.mod" \
    "$test_executable"

nvfortran \
    -cuda \
    -gpu=cc86,cuda13.3 \
    -O3 \
    src/native/cpu/lexical/lexical_core.f90 \
    src/native/cuda/lexical/lexical_cuda_core.cuf \
    tests/native/cuda/test_lexical_cuda_core.cuf \
    -module "$build_directory" \
    -o "$test_executable"

"$test_executable"