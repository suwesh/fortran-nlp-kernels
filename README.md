# fortran-nlp-kernels

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![Fortran](https://img.shields.io/badge/Fortran-2018-purple.svg)](https://fortran-lang.org/)
[![CUDA Fortran](https://img.shields.io/badge/CUDA%20Fortran-76B900?logo=nvidia&logoColor=white)](https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](#platform-support)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-green.svg)](#release-status)

**Exact lexical and dense-vector similarity search for Python with compiled Fortran, OpenMP, OpenBLAS, CUDA Fortran, and cuBLAS backends.**

`fortran-nlp-kernels` provides exact word-shingle Jaccard search and exact flat vector search for inner product, squared L2 distance, and cosine similarity. It combines a Python API with native CPU and NVIDIA GPU implementations, deterministic top-k selection, and a benchmark-informed CPU-only `backend="auto"` resolver.

## Release status

Version `0.1.1` adds explicit CUDA Fortran and cuBLAS vector-search backends for single-query and batched search. CUDA lexical Jaccard search remains available through the handwritten CUDA Fortran backend.

Version `0.1.0` introduced benchmark-informed CPU backend selection, optional machine-specific OpenMP and OpenBLAS thread calibration, versioned calibration profiles, runtime memory guardrails, and temporary thread control.

The package is intended for exact similarity search, systems experimentation, reproducible performance engineering, and integration into Python applications.

## Platform support

### CPU distribution

- Linux x86-64
- CPython 3.10-3.13
- `manylinux_2_28` wheels
- Serial Fortran, OpenMP Fortran, and OpenBLAS backends

### CUDA companion distribution

- Linux x86-64
- CPython 3.12 for the initial `0.1.1` CUDA wheel
- NVIDIA GPU with a compatible driver
- CUDA 13.3 runtime and cuBLAS available on the host
- Any additional NVIDIA HPC SDK runtime libraries reported by the published wheel
- External-runtime packaging model: NVIDIA runtime libraries are not bundled into the wheel

The CUDA companion is distributed separately as `fortran-nlp-kernels-cuda` and installs `_cuda` beside the base package. The base CPU package remains usable without NVIDIA software.

## Features

- Exact word-shingle Jaccard search, including batched search
- Exact inner-product, squared-L2, and cosine vector search
- Serial and OpenMP Fortran CPU kernels
- OpenBLAS SGEMV and SGEMM vector backends
- Handwritten CUDA Fortran lexical and vector kernels
- cuBLAS SGEMV and SGEMM vector backends
- Deterministic CPU top-k with stable ties and zero-based indices
- CPU workload-aware `backend="auto"` dispatch
- Optional machine-specific OpenMP and OpenBLAS thread calibration
- Runtime memory guardrails and temporary thread control
- Native Fortran, C ABI, nanobind, NumPy-reference, and public API tests

## Installation: PyPI

### CPU package

Install the stable CPU distribution:

```bash
python -m pip install fortran-nlp-kernels==0.1.1
```

The prebuilt CPU wheels do not require a Fortran compiler or OpenBLAS development package.

### CUDA companion

Install the CUDA companion and its matching base package:

```bash
python -m pip install fortran-nlp-kernels-cuda==0.1.2
```

The companion declares an exact dependency on:

```text
fortran-nlp-kernels==0.1.1
```

The CUDA wheel uses external NVIDIA runtime libraries. Before importing `fortran_nlp._cuda`, ensure that the supported NVIDIA driver, CUDA runtime, cuBLAS, and required NVIDIA HPC SDK runtime libraries are discoverable by the Linux dynamic loader.

Basic diagnostics:

```bash
nvidia-smi
```

After installation:

```bash
python - <<'PY'
from fortran_nlp import _cuda
print(_cuda.__file__)
PY
```

To inspect unresolved or externally loaded libraries:

```bash
ldd "$(python -c 'from fortran_nlp import _cuda; print(_cuda.__file__)')"
```

The release notes for the CUDA companion specify the exact build toolchain and runtime versions. The initial `0.1.1` CUDA wheel targets CUDA 13.3 and NVIDIA HPC SDK 26.9.

### Editable CPU build

```bash
git clone https://github.com/suwesh/fortran-nlp-kernels.git
cd fortran-nlp-kernels
python -m pip install -e .
```

A CPU source build requires a supported Python environment, GFortran, a C++17 compiler, OpenMP, and OpenBLAS development files.

On Ubuntu:

```bash
sudo apt update
sudo apt install gfortran g++ libopenblas-dev libgomp1
```

### Editable CUDA-enabled build

```bash
python -m pip install -e . \
  --config-settings=setup-args="-Dfnlp_cuda=true" \
  --config-settings=setup-args="-Dfnlp_cuda_arch=cc86" \
  --config-settings=setup-args="-Dfnlp_cuda_toolkit=cuda13.3"
```

A CUDA source build additionally requires NVIDIA HPC SDK with `nvfortran`, a compatible CUDA toolkit, and an architecture value appropriate for the target GPU. The example uses `cc86`, which matches the development RTX 3060 system.

## Quick start

### Exact lexical Jaccard search

```python
from fortran_nlp.lexical import JaccardIndex

index = JaccardIndex(
    [
        "reset the account password",
        "update contact information",
        "create a new insurance application",
    ],
    backend="auto",
)

results = index.search(
    "how to reset the account password",
    k=2,
    minimum_score=0.01,
)

for result in results:
    print(result)
```

`JaccardIndex.search_batch()` compares multiple candidates against the same indexed corpus:

```python
batch_results = index.search_batch(
    [
        "how to reset the account password",
        "contact information update",
    ],
    k=2,
    minimum_score=0.01,
)
```

Example result:

```text
JaccardResult(
    document_id=0,
    score=0.5,
    intersection_count=2,
    union_count=4
)
```

Text preprocessing performs Unicode normalization, case folding, fixed-size word shingling, deterministic 64-bit hashing, sorting, and deduplication before native execution.

## Exact vector search

The Python-facing database layout is:

```text
(vector_count, dimension)
```

All public vector-search inputs are converted to contiguous `float32` arrays. The package internally stores the database in a Fortran-compatible `(dimension, vector_count)` layout.

```python
import numpy as np

from fortran_nlp.vector_search import (
    FlatIndexCosine,
    FlatIndexIP,
    FlatIndexL2,
)

database = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-1.0, 2.0, 1.0],
    ],
    dtype=np.float32,
)

query = np.array([1.0, 2.0, 3.0], dtype=np.float32)

ip_index = FlatIndexIP(database, backend="auto")
scores, ids = ip_index.search(query, k=2)

l2_index = FlatIndexL2(database, backend="cuda_fortran")
distances, ids = l2_index.search(query, k=2)

cosine_index = FlatIndexCosine(database, backend="cuda_cublas")
scores, ids = cosine_index.search(query, k=2)
```

`FlatIndexL2` returns squared Euclidean distance and does not calculate the square root. `FlatIndexCosine` normalizes the database and queries before inner-product search; zero vectors are invalid.

## Batched vector search

With `backend="auto"`, batched inner-product and cosine search selects between repeated OpenBLAS `SGEMV` and native OpenBLAS `SGEMM` according to query count, the active policy, and the runtime memory budget. Batched squared-L2 search uses the selected repeated serial, repeated OpenMP, or explicit BLAS execution path.

The public query layout is:

```text
(query_count, dimension)
```

Batched search uses the same index:

```python
queries = np.array(
    [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

scores, ids = ip_index.search_batch(queries, k=2)

print(scores.shape)  # (query_count, k)
print(ids.shape)     # (query_count, k)
```

## Backends

### Lexical search

```text
fortran          -> serial Fortran
fortran_openmp   -> OpenMP Fortran
cuda_fortran     -> handwritten CUDA Fortran
auto             -> CPU workload-aware dispatch
```

### Vector search

```text
fortran          -> serial Fortran
fortran_openmp   -> OpenMP Fortran
fortran_blas     -> OpenBLAS SGEMV/SGEMM
cuda_fortran     -> handwritten CUDA Fortran scoring + CPU top-k
cuda_cublas      -> cuBLAS scoring + CPU top-k
auto             -> CPU workload-aware dispatch
```

CUDA backends are explicit in `0.1.1`. `backend="auto"` remains CPU-only. Automatic CPU/GPU crossover dispatch is deferred until calibrated GPU performance data is available.

## Automatic CPU backend selection

Automatic resolution happens at search time, when the complete workload is known. The original `"auto"` request remains visible through the index's `backend` property.

Policy precedence is:

```text
1. Explicit backend request
2. Compatible machine calibration profile
3. Packaged benchmark-informed fallback policy
```

### Packaged fallback policy

Without a compatible machine profile, `0.1.0` uses conservative workload rules derived from the packaged CPU performance findings.

```text
Lexical single query
small workload             -> serial Fortran
larger workload            -> OpenMP Fortran

Lexical batch
small corpus               -> repeated selected single-query path
scaled batch size 1        -> native batched OpenMP
scaled batch size 2-31     -> repeated OpenMP
scaled batch size >= 32    -> native batched OpenMP

IP and cosine single query -> OpenBLAS SGEMV
IP and cosine batch 1-7    -> repeated SGEMV
IP and cosine batch >= 8   -> OpenBLAS SGEMM

Squared-L2 single query    -> OpenMP Fortran
Squared-L2 batch           -> repeated OpenMP
```

Native batched lexical execution and vector `SGEMM` are selected only when their estimated allocations fit the runtime memory budget. Otherwise, automatic dispatch falls back to repeated execution.

### Explicit backend override

Explicit backends always bypass automatic policy selection:

```python
serial_index = FlatIndexIP(database, backend="fortran")
openmp_index = FlatIndexIP(database, backend="fortran_openmp")
blas_index = FlatIndexIP(database, backend="fortran_blas")
```

Explicit requests also preserve the user's existing OpenMP or OpenBLAS runtime settings.

## Optional CPU calibration

The stable release includes explicit, bounded machine calibration for representative OpenMP and OpenBLAS thread counts.

Calibration is never run during installation, import, index construction, or ordinary search. It does not modify the packaged fallback policy.

Run calibration:

```bash
fortran-nlp-calibrate
```

Equivalent module invocation:

```bash
python -m fortran_nlp.calibration
```

Show the active compatible profile:

```bash
fortran-nlp-calibrate --show
```

Reset the selected user profile:

```bash
fortran-nlp-calibrate --reset
```

Use a custom profile path:

```bash
fortran-nlp-calibrate --output /path/to/cpu_profile.json
```

Or configure profile discovery:

```bash
export FORTRAN_NLP_PROFILE=/path/to/cpu_profile.json
```

The default profile location is:

```text
~/.config/fortran-nlp-kernels/cpu_profile.json
```

Profiles are versioned and checked against the installed package and machine fingerprint. Missing, malformed, stale, or incompatible profiles safely fall back to the packaged policy.

The `0.1.0` calibration command tunes representative thread counts for:

- lexical OpenMP execution;
- squared-L2 OpenMP execution;
- OpenBLAS `SGEMV`;
- OpenBLAS `SGEMM`.

The generated profile records candidate thread counts, selected counts, median timings, warm-ups, repeats, and the crossover-policy source. Machine-specific workload crossover calibration is deferred to a later release.

## Thread control

Automatic execution temporarily applies the selected OpenMP or OpenBLAS thread count around the native call and restores the previous runtime setting afterward, including when an exception occurs.

Explicit backends preserve the user's runtime configuration. Environment variables remain useful for explicit execution and controlled benchmarks:

```bash
# Handwritten serial
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmark.py

# Handwritten OpenMP
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 python benchmark.py

# OpenBLAS
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=6 python benchmark.py
```

The package does not globally overwrite `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS`.

## Architecture

```text
Python public API
        ↓
Validation, normalization, and workload description
        ↓
Explicit backend routing or CPU backend resolver
        ↓
nanobind C++ extension
        ↓
stable C ABI
        ↓
Fortran / OpenMP / OpenBLAS / CUDA Fortran / cuBLAS
        ↓
deterministic CPU top-k
```

The CPU vector path is separated into:

```text
vector_search_core.f90   -> serial and OpenMP IP/L2 kernels
vector_search_blas.f90   -> OpenBLAS SGEMV/SGEMM
vector_search_c_api.f90  -> C ABI
cpu_bindings.cpp         -> Python adapter
topk_core.f90            -> deterministic exact top-k
```

The CUDA vector path follows the same layered structure:

```text
vector_search_cuda_core.cuf   -> handwritten CUDA IP/L2 scoring
vector_search_cublas.cuf      -> cuBLAS IP/L2 scoring
vector_search_cuda_c_api.cuf  -> C ABI
cuda_bindings.cpp             -> Python adapter
topk_core.f90                 -> deterministic CPU top-k
```

For CUDA vector search in `0.1.1`:

```text
GPU scoring
-> copy the complete score array or matrix to host
-> deterministic CPU top-k
-> return values and zero-based indices
```

The cuBLAS squared-L2 path uses:

```text
GPU squared-norm kernels
-> cuBLAS SGEMV/SGEMM inner products
-> GPU L2 norm-expansion kernel
-> host top-k
```

## Correctness and validation

The implementation is validated across:

```text
native Fortran tests
C ABI tests
nanobind extension tests
manual numerical examples
NumPy equivalence tests
CPU/CUDA equivalence tests
handwritten CUDA/cuBLAS equivalence tests
single-query and batched-query tests
cosine normalization tests
public Python API tests
input-validation and edge-case tests
deterministic tie-handling tests
fallback-policy and calibration-profile tests
memory-guardrail and thread-restoration tests
CPU-only and CUDA-enabled build tests
```

Validated `0.1.1` configurations include:

- CUDA-enabled editable build: 428 Python tests passed
- CUDA-focused suites: 223 tests passed
- Public CUDA vector-index suite: 116 tests passed
- Low-level CUDA vector suite: 84 tests passed
- CPU-only editable build: 205 tests passed with `_cuda` absent

Run the complete available suite:

```bash
python -m pytest tests -q
```

Native test programs and runners are available under:

```text
tests/native/
```

## Current limitations

- Linux x86-64 only
- Result IDs are zero-based positional indices
- Exact top-k selection currently uses an `O(Nk)` implementation
- CUDA wheels require external NVIDIA runtime libraries
- CUDA backends must be selected explicitly
- `backend="auto"` is CPU-only
- No GPU-resident database or corpus
- No GPU top-k or fused scoring-selection kernel
- No CPU/GPU crossover policy, multi-GPU execution, or mixed-precision scoring
- No CPU affinity, NUMA-aware dispatch, or hybrid-core-aware placement

## Build and release system

The project uses:

```text
meson-python   -> Python build backend
Meson          -> native build configuration
GFortran       -> CPU Fortran compilation
nanobind       -> compiled Python extensions
OpenMP         -> handwritten multicore kernels
OpenBLAS       -> CPU SGEMV and SGEMM
NVIDIA HPC SDK -> CUDA Fortran compilation
cuBLAS         -> GPU SGEMV and SGEMM
```

Build and validate the CPU source distribution and wheel:

```bash
python -m pip install build twine
rm -rf dist build .mesonpy-*
python -m build
python -m twine check dist/*
```

Official CPU and CUDA distributions are built in GitHub Actions and published to separate PyPI projects through Trusted Publishing.

## Research direction

The library is being developed as a reusable software artifact and performance-engineering study covering:

- exact lexical and dense-vector similarity search;
- Python/native interface overhead;
- serial and multicore CPU scaling;
- handwritten Fortran versus optimized BLAS;
- C++ and CUDA C++ research baselines;
- CUDA Fortran and cuBLAS backends;
- CPU/GPU crossover points;
- kernel-only and end-to-end performance.

Working technical-report title:

> **Fortran Kernels for NLP: Exact Lexical and Vector Similarity Search Across Multicore CPUs and NVIDIA GPUs**

## Roadmap

Follow-up work may include:

- faster exact top-k selection;
- persistent GPU-resident databases and corpora;
- GPU top-k and fused score-selection kernels;
- calibrated CPU/GPU crossover dispatch;
- broader Python, platform, and GPU-architecture support;
- CUDA C++ and FAISS research comparisons;
- multi-GPU and mixed-precision experiments.

## License

This project is licensed under **GPL-3.0-only**. See [LICENSE](LICENSE).

## Author

**Suwesh Prasad Sah**

- Homepage: <https://suwesh.github.io/>
- Repository: <https://github.com/suwesh/fortran-nlp-kernels>
- Issues: <https://github.com/suwesh/fortran-nlp-kernels/issues>
