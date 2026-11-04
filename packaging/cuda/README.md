# fortran-nlp-kernels-cuda

CUDA Fortran and cuBLAS companion distribution for [`fortran-nlp-kernels`](https://github.com/suwesh/fortran-nlp-kernels).

This package installs the optional native `fortran_nlp._cuda` extension. The matching base package provides the Python API, CPU extension, lexical and vector-search modules, backend dispatch, and calibration utilities.

## Installation

```bash
python -m pip install fortran-nlp-kernels-cuda==0.1.1
```

The companion requires and automatically installs:

```text
fortran-nlp-kernels==0.1.1
```

## Supported environment

The initial `0.1.1` CUDA wheel targets:

- Linux x86-64
- CPython 3.12
- NVIDIA GPU with a compatible driver
- CUDA 13.3 runtime
- cuBLAS
- Any additional NVIDIA HPC SDK runtime libraries reported by the published wheel

NVIDIA runtime libraries are external and are not bundled into the wheel.

## Available backends

```text
cuda_fortran -> handwritten CUDA Fortran scoring
cuda_cublas  -> cuBLAS SGEMV/SGEMM scoring
```

Both backends use the existing deterministic CPU top-k implementation. CUDA backends are explicit in `0.1.1` and are not selected by `backend="auto"`.

## Usage

```python
import numpy as np

from fortran_nlp.vector_search import FlatIndexL2

vectors = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-1.0, 2.0, 1.0],
    ],
    dtype=np.float32,
)

query = np.array([1.0, 2.0, 3.0], dtype=np.float32)

index = FlatIndexL2(
    vectors,
    backend="cuda_cublas",
)

distances, indices = index.search(query, k=2)
```

The CUDA companion supports:

- `JaccardIndex` with `backend="cuda_fortran"`
- `FlatIndexL2` with `cuda_fortran` and `cuda_cublas`
- `FlatIndexIP` with `cuda_fortran` and `cuda_cublas`
- `FlatIndexCosine` with `cuda_fortran` and `cuda_cublas`
- Single-query and batched search

## Diagnostics

Check the NVIDIA driver:

```bash
nvidia-smi
```

Verify the extension import:

```bash
python - <<'PY'
from fortran_nlp import _cuda
print(_cuda.__file__)
PY
```

Inspect dynamic-library resolution:

```bash
ldd "$(python -c 'from fortran_nlp import _cuda; print(_cuda.__file__)')"
```

## Source builds

Building the CUDA extension from source requires NVIDIA HPC SDK with `nvfortran`, a compatible CUDA toolkit, and a C++17 compiler.

```bash
nvfortran --version
```

## License

Licensed under **GPL-3.0-only**. See the main repository for source code and license text.