# fortran-nlp-kernels

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![Fortran](https://img.shields.io/badge/Fortran-2018-purple.svg)](https://fortran-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](#platform-support)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](#release-status)

**Exact lexical and dense-vector similarity search for Python with Fortran backends, CPU autotuning, and workload-aware runtime dispatch.**

`fortran-nlp-kernels` provides a Python API for exact word-shingle Jaccard search and exact dense-vector search while executing the computational kernels in compiled Fortran, OpenMP, and OpenBLAS code. 

The stable `0.1.0` release introduces benchmark-informed CPU autotuning and workload-aware runtime dispatch, optional machine-specific thread calibration, versioned calibration profiles, runtime memory guardrails, and temporary OpenMP/OpenBLAS thread control. The shipped fallback policy is based on CPU performance experiments conducted on the `perf-eval` branch against the `0.1.0b2` implementation.

## Release status

`0.1.0` is the first stable Linux CPU release.

The package is intended for exact similarity search, systems experimentation, reproducible CPU performance engineering, and integration into Python applications. Development continues beyond `0.1.0`; future releases may add improved top-k selection, machine-specific crossover calibration, additional platform support, and GPU backends.

## Included

- Exact word-shingle Jaccard search
- Batched lexical Jaccard search
- Exact inner-product vector search
- Exact squared-L2 vector search
- Exact cosine-similarity search
- Serial Fortran lexical and vector scoring
- OpenMP Fortran lexical and vector scoring
- OpenBLAS `SGEMV` single-query vector search
- OpenBLAS `SGEMM` batched vector search
- Workload-aware `backend="auto"` runtime dispatch
- Packaged benchmark-informed CPU fallback policy
- Optional machine-specific OpenMP and OpenBLAS thread calibration
- Versioned calibration profiles with compatibility checks
- Runtime memory guardrails for native lexical batches and vector `SGEMM`
- Temporary thread-runtime control with guaranteed restoration
- Exact top-k selection
- NumPy correctness references
- Python, native Fortran, C ABI, integration, policy, profile, calibration, and runtime tests
- CPython 3.10–3.13 manylinux wheels

## Current limitations

- Linux x86-64 only
- Result IDs are zero-based positional indexes
- Exact top-k selection currently uses an `O(Nk)` implementation
- Machine calibration tunes representative OpenMP and OpenBLAS thread counts; workload crossover thresholds remain supplied by the packaged benchmark policy
- Full-SMT counts are not selected automatically by the shipped fallback policy
- Hybrid-core-aware placement, CPU affinity management, cache topology, and NUMA-aware dispatch are not included
- CUDA Fortran and GPU backends are not included

## Architecture

```text
Python public API
        ↓
Runtime workload descriptor
        ↓
Backend resolver:-
        +-- Level 1: explicit backend override
        +-- Level 2: compatible machine calibration profile
        +-- Level 3: packaged fallback policy
        ↓
BackendDecision
        ↓
OpenMP / OpenBLAS runtime controller
        ↓
Python validation, preprocessing, and orchestration
        ↓
nanobind C++ extension
        ↓
stable C ABI
        ↓
Fortran / OpenMP / OpenBLAS kernels
```

The native lexical path is separated into focused layers:

```text
lexical_core.f90     → lexical algorithms
lexical_c_api.f90    → Fortran-to-C adapter
.h header            → C declarations
cpu_bindings.cpp     → C-to-Python adapter
```

The vector-search path follows the same structure:

```text
vector_search_core.f90   → handwritten IP and squared-L2 kernels
vector_search_blas.f90   → SGEMV and SGEMM implementations
topk_core.f90            → exact top-k selection
vector_search_c_api.f90  → Fortran-to-C adapter
cpu_bindings.cpp         → C-to-Python adapter
```

The automatic-dispatch layer is implemented through:

```text
auto.py                   -> workload descriptors and backend resolution
profile.py                -> machine fingerprinting and profile compatibility
runtime.py                -> memory discovery and temporary thread control
calibration.py            -> bounded optional CPU thread calibration
policies/cpu_fallback_v1.json
                           -> packaged benchmark-informed fallback policy
```

## Platform support

```text
Operating system: Linux
Architecture: x86-64
Python: CPython 3.10, 3.11, 3.12, and 3.13
Wheel policy: manylinux_2_28
```

The source package declares Python 3.10 or newer.

## System prerequisites

### Installing the prebuilt Linux wheel

The published manylinux wheel bundles the required non-policy native libraries. A Fortran compiler and the OpenBLAS development package are not required when installing a compatible prebuilt wheel.

### Building from source

Install the compiler and development libraries:

```bash
sudo apt update
sudo apt install gfortran g++ libopenblas-dev libgomp1
```

A source build also requires a supported Python environment and `pip`. Build-time Python dependencies are installed through `pyproject.toml`.

## Installation

### PyPI

Install the latest available stable release:

```bash
python -m pip install fortran-nlp-kernels
```

To install this exact release:

```bash
python -m pip install fortran-nlp-kernels==0.1.0
```

### Local wheel

Download the wheel matching the installed CPython version from the GitHub release or CI artifacts, then install it by path: 
```bash 
python -m pip install \ 
    ./fortran_nlp_kernels-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

### Editable source installation

```bash
git clone https://github.com/suwesh/fortran-nlp-kernels.git
cd fortran-nlp-kernels
python -m pip install -e .
```

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

Example result:

```text
JaccardResult(
    document_id=0,
    score=0.5,
    intersection_count=2,
    union_count=4
)
```

Text preprocessing performs Unicode normalization, case folding, fixed-size word shingling, deterministic 64-bit hashing, sorting, and deduplication before invoking the native kernel.

### Batched lexical Jaccard search

`JaccardIndex.search_batch()` compares multiple candidate texts against the same indexed corpus.

```python
from fortran_nlp.lexical import JaccardIndex

batch_results = index.search_batch(
    [
        "how to reset the account password",
        "contact information update",
    ],
    k=2,
    minimum_score=0.01,
)

for candidate_index, candidate_results in enumerate(batch_results):
    print(candidate_index, candidate_results)
```

The return type is:

```python
list[list[JaccardResult]]
```

The outer list contains one entry per candidate. Each inner list contains the ranked corpus matches for that candidate.

With `backend="auto"`, batched lexical search chooses among repeated serial, repeated OpenMP, and native batched OpenMP execution according to the workload, active policy, and runtime memory budget.

Explicit lexical batch execution is also supported:

```text
backend="fortran"         -> repeated serial execution
backend="fortran_openmp"  -> repeated OpenMP execution
```

## Exact vector search

The Python-facing database layout is:

```text
(vector_count, dimension)
```

All public vector-search inputs are converted to contiguous `float32` arrays. The package internally stores the database in a Fortran-compatible `(dimension, vector_count)` layout.

### Inner product

Higher scores are better.

```python
import numpy as np

from fortran_nlp.vector_search import FlatIndexIP

database = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-1.0, 2.0, 1.0],
    ],
    dtype=np.float32,
)

query = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float32,
)

index = FlatIndexIP(
    database,
    backend="auto",
)

scores, ids = index.search(query, k=2)

print(scores)
print(ids)
```

### Squared-L2 distance

Lower distances are better. The package returns squared Euclidean distance and does not calculate the square root.

```python
from fortran_nlp.vector_search import FlatIndexL2

index = FlatIndexL2(
    database,
    backend="auto",
)

distances, ids = index.search(query, k=2)
```

### Cosine similarity

Database vectors and queries are L2-normalized, then searched using the inner-product backend.

```python
from fortran_nlp.vector_search import FlatIndexCosine

index = FlatIndexCosine(
    database,
    backend="auto",
)

scores, ids = index.search(query, k=2)
```

Zero vectors cannot be used with `FlatIndexCosine` because cosine normalization is undefined for a zero norm.

## Batched vector search

With `backend="auto"`, batched inner-product and cosine search selects between repeated OpenBLAS `SGEMV` and native OpenBLAS `SGEMM` according to query count, the active policy, and the runtime memory budget. Batched squared-L2 search uses the selected repeated serial, repeated OpenMP, or explicit BLAS execution path.

The public query layout is:

```text
(query_count, dimension)
```

Example:

```python
queries = np.array(
    [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

index = FlatIndexIP(
    database,
    backend="auto",
)

scores, ids = index.search_batch(
    queries,
    k=2,
)

print(scores.shape)  # (query_count, k)
print(ids.shape)     # (query_count, k)
```

`FlatIndexL2` and `FlatIndexCosine` expose the same `search_batch()` interface.

Explicit batched vector execution is supported through all public backends:

```text
backend="fortran"         -> repeated serial execution
backend="fortran_openmp"  -> repeated OpenMP execution
backend="fortran_blas"    -> native BLAS batch execution
```

## Backends

### Lexical search

```text
fortran          -> handwritten serial Fortran
fortran_openmp   -> handwritten OpenMP Fortran
auto             -> workload-aware runtime dispatch
```

### Vector search

```text
fortran          -> handwritten serial Fortran
fortran_openmp   -> handwritten OpenMP Fortran
fortran_blas     -> OpenBLAS SGEMV/SGEMM through Fortran
auto             -> workload-aware runtime dispatch
```

## Automatic backend selection

Both public index families accept:

```python
backend="auto"
```

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

## Correctness and tests

The implementation is validated across:

```text
native Fortran core tests
C ABI tests
nanobind extension tests
NumPy equivalence tests
serial/OpenMP equivalence tests
BLAS/handwritten equivalence tests
single-query and batched-query tests
public Python API tests
input-validation and edge-case tests
deterministic tie-handling tests
fallback-policy and profile tests
calibration-profile management tests
memory-guardrail tests
OpenMP and OpenBLAS restoration tests
package-metadata tests
wheel smoke tests
```

The stable `0.1.0` release passes **205 Python tests**.

Run the Python test suite:

```bash
python -m pytest tests/ -v
```

Native test programs are available under:

```text
tests/native/
```

### Continuous integration

The repository includes two Linux CI workflows:

```text
CPU Tests
-> builds the package from source
-> runs the complete test suite on CPython 3.10-3.13

Build Linux Wheels
-> builds and tests manylinux wheels with cibuildwheel
-> targets CPython 3.10, 3.11, 3.12, and 3.13
```

The wheel workflow runs manually or when a version tag matching `v*` is pushed. Generated wheels are stored as workflow artifacts. Release attachment and PyPI publication remain explicit release steps.

## Build system

The project uses:

```text
meson-python  → Python build backend
Meson         → native build configuration
GFortran      → Fortran compilation
nanobind      → compiled Python extension
OpenMP        → handwritten multicore kernels
OpenBLAS      → SGEMV and SGEMM vector-search backends
```

Build the wheel and source distribution:

```bash
python -m pip install build
rm -rf dist/ build/ .mesonpy-*
python -m build
```

Expected release artifacts:

```text
fortran_nlp_kernels-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
fortran_nlp_kernels-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
fortran_nlp_kernels-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl
fortran_nlp_kernels-0.1.0-cp313-cp313-manylinux_2_28_x86_64.whl
fortran_nlp_kernels-0.1.0.tar.gz
```

Validate distribution metadata and README rendering:

```bash
python -m pip install twine
python -m twine check dist/*
```

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

Planned follow-up work includes:

- machine-specific workload crossover calibration;
- expanded calibration stability statistics;
- optional document and vector IDs;
- optimized exact top-k selection;
- CPU performance evaluation against NumPy, C++, and FAISS;
- automated PyPI Trusted Publishing;
- broader platform and architecture support;
- CUDA Fortran lexical and vector-search kernels;
- cuBLAS-backed vector search;
- CUDA C++ research baselines;
- FAISS GPU exact-flat comparisons;
- GPU-resident corpora and vector databases.

## License

This project is licensed under **GPL-3.0-only**. See [LICENSE](LICENSE).

## Author

**Suwesh Prasad Sah**

- Homepage: <https://suwesh.github.io/>
- Repository: <https://github.com/suwesh/fortran-nlp-kernels>
- Issues: <https://github.com/suwesh/fortran-nlp-kernels/issues>
