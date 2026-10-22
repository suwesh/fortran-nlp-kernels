# fortran-nlp-kernels

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Fortran](https://img.shields.io/badge/Fortran-2018-purple.svg)](https://fortran-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](#platform-support)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0b1-orange.svg)](#beta-status)

**Python-facing exact lexical and dense-vector similarity search backed by Fortran, OpenMP, and OpenBLAS.**

`fortran-nlp-kernels` provides a clean Python API while executing the computational kernels in compiled native code. The initial CPU beta includes exact word-shingle Jaccard search, exact inner-product search, exact squared-L2 search, cosine similarity, multicore OpenMP execution, and batched BLAS search.

## Beta status

`0.1.0b1` is the first Linux CPU beta.

The current beta is intended for evaluation, reproducible systems work, and early integration. Public APIs may still evolve before the stable `0.1.0` release.

### Included

- Exact word-shingle Jaccard search
- Serial Fortran lexical scoring
- OpenMP Fortran lexical scoring
- Exact inner-product vector search
- Exact squared-L2 vector search
- Exact cosine-similarity search
- Serial Fortran vector scoring
- OpenMP Fortran vector scoring
- OpenBLAS `SGEMV` single-query search
- OpenBLAS `SGEMM` batched-query search
- Exact top-k selection
- NumPy correctness references
- Python, native Fortran, C ABI, and integration tests

### Current limitations

- Linux x86-64 only
- The published beta wheel targets CPython 3.12
- Result IDs are currently zero-based positional indexes
- Batched search currently requires `backend="fortran_blas"`
- Handwritten top-k selection currently uses an `O(Nk)` reference implementation
- CUDA Fortran and GPU backends are not included in this beta
- The Linux wheel currently expects OpenBLAS, GNU Fortran runtime, and GNU OpenMP runtime libraries on the host

## Architecture

```text
Python public API
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

## Platform support

The initial `0.1.0b1` wheel targets CPython 3.12. The current development branch is continuously built and tested for: 
```text 
Operating system: Linux 
Architecture: x86-64 
Python: CPython 3.10, 3.11, 3.12, and 3.13 
Wheel policy: manylinux_2_28
```

The source package declares Python 3.10 or newer, but this beta has currently been built and validated with CPython 3.12 on Ubuntu 24.04.

## System prerequisites

### Installing the prebuilt Linux wheel

On Ubuntu or Debian-based systems, install the runtime libraries:

```bash
sudo apt update
sudo apt install libopenblas0 libgomp1 libgfortran5
```

### Building from source

Install the compiler and development libraries:

```bash
sudo apt update
sudo apt install gfortran g++ libopenblas-dev libgomp1
```

A source build also requires a supported Python environment and `pip`. Build-time Python dependencies are installed through `pyproject.toml`.

## Installation

### PyPI beta

After the beta is published:

```bash
python -m pip install --pre fortran-nlp-kernels
```

To request this exact beta:

```bash
python -m pip install fortran-nlp-kernels==0.1.0b1
```

### Local wheel

```bash
python -m pip install \
  dist/fortran_nlp_kernels-0.1.0b1-cp312-cp312-linux_x86_64.whl
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
        "update seller information",
        "create a new loan application",
    ],
    backend="fortran_openmp",
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

## Exact vector search

The Python-facing database layout is:

```text
(vector_count, dimension)
```

All public vector-search inputs are converted to contiguous `float32` arrays. The package internally stores the database in a Fortran-compatible native layout.

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
    backend="fortran_openmp",
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
    backend="fortran",
)

distances, ids = index.search(query, k=2)
```

### Cosine similarity

Database vectors and queries are L2-normalized, then searched using the inner-product backend.

```python
from fortran_nlp.vector_search import FlatIndexCosine

index = FlatIndexCosine(
    database,
    backend="fortran_blas",
)

scores, ids = index.search(query, k=2)
```

Zero vectors cannot be used with `FlatIndexCosine` because cosine normalization is undefined for a zero norm.

## Batched vector search

Batched search uses OpenBLAS `SGEMM` and currently requires:

```python
backend="fortran_blas"
```

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
    backend="fortran_blas",
)

scores, ids = index.search_batch(
    queries,
    k=2,
)

print(scores.shape)  # (query_count, k)
print(ids.shape)     # (query_count, k)
```

`FlatIndexL2` and `FlatIndexCosine` expose the same `search_batch()` interface.

## Backends

### Lexical search

```text
fortran          → handwritten serial Fortran
fortran_openmp   → handwritten OpenMP Fortran
```

### Vector search

```text
fortran          → handwritten serial Fortran
fortran_openmp   → handwritten OpenMP Fortran
fortran_blas     → OpenBLAS SGEMV/SGEMM through Fortran
```

### Automatic backend selection

Both public index families accept:

```python
backend="auto"
```

The current deterministic policy is:

```text
JaccardIndex
auto → fortran_openmp

FlatIndexIP
FlatIndexL2
FlatIndexCosine
auto → fortran_blas
```

Example:

```python
from fortran_nlp.lexical import JaccardIndex
from fortran_nlp.vector_search import FlatIndexCosine

lexical_index = JaccardIndex(
    corpus_texts,
    backend="auto",
)

vector_index = FlatIndexCosine(
    database_vectors,
    backend="auto",
)
```

The current policy selects stable production defaults. It does not
benchmark the host machine dynamically. Future releases may use measured
workload crossover points to refine automatic selection.

## Thread control

The handwritten OpenMP and OpenBLAS backends use different thread controls.

### Handwritten OpenMP backend

```bash
OMP_NUM_THREADS=4 python your_script.py
```

### OpenBLAS backend

```bash
OPENBLAS_NUM_THREADS=4 python your_script.py
```

For controlled benchmarks, avoid enabling high thread counts in both runtimes simultaneously:

```bash
# Handwritten serial
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmark.py

# Handwritten OpenMP
OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 python benchmark.py

# BLAS
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=12 python benchmark.py
```

The library does not globally overwrite these environment variables.

## Correctness and tests

The implementation is validated across several boundaries:

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
package-metadata tests
wheel smoke tests
```

The current development branch passes **98 Python tests** covering
lexical search, exact vector search, native bindings, automatic backend
selection, memory layouts, input validation, and package metadata.

Run the Python test suite:

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=4 \
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
→ builds the package from source
→ runs the complete test suite on CPython 3.10–3.13

Build Linux Wheels
→ builds and tests manylinux wheels with cibuildwheel
→ targets CPython 3.10, 3.11, 3.12, and 3.13
```

The wheel workflow runs manually or when a version tag matching `v*`
is pushed. Generated wheels are stored as temporary workflow artifacts.
The workflow does not automatically attach wheels to GitHub Releases or
upload them to PyPI.

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

Expected beta artifacts:

```text
dist/fortran_nlp_kernels-0.1.0b1-cp312-cp312-linux_x86_64.whl
dist/fortran_nlp_kernels-0.1.0b1.tar.gz
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

- benchmark-informed automatic backend selection;
- optional document and vector IDs;
- handwritten serial and OpenMP batched-query baselines;
- optimized exact top-k selection;
- CPU performance evaluation against NumPy, C++, and FAISS;
- automated PyPI Trusted Publishing;
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
