# fortran-nlp-kernels-cuda

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![Fortran](https://img.shields.io/badge/Fortran-2018-purple.svg)](https://fortran-lang.org/)
[![CUDA Fortran](https://img.shields.io/badge/CUDA%20Fortran-76B900?logo=nvidia&logoColor=white)](https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](#platform-support)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.2-green.svg)](#release-status)

CUDA Fortran and cuBLAS companion distribution for `fortran-nlp-kernels`.
This package installs the optional native `fortran_nlp._cuda` extension. The matching base package provides the Python API, CPU extension, lexical and vector-search modules, backend dispatch, preprocessing, normalization, calibration, and runtime utilities.

## Release compatibility

`fortran-nlp-kernels-cuda` is versioned independently from the base package.
The current compatible release pair is:
```text
fortran-nlp-kernels==0.1.1
fortran-nlp-kernels-cuda==0.1.2
```
CUDA companion `0.1.2` is the corrected external-runtime binary release for base package `0.1.1`. It replaces the yanked CUDA companion `0.1.1` wheel.
The companion declares this dependency:
```text
fortran-nlp-kernels==0.1.1
```
Installing the CUDA companion therefore installs the matching base package automatically.

## Installation from PyPI

Create or activate a CPython 3.12 environment:
```bash
python3.12 -m venv env312
source env312/bin/activate
python -m pip install --upgrade pip
```
### Install the CUDA companion:

```bash
python -m pip install fortran-nlp-kernels-cuda
```

For an explicitly reproducible installation:

```bash
python -m pip install \
fortran-nlp-kernels==0.1.1 \
fortran-nlp-kernels-cuda==0.1.2
```

The installation provides:
```text
fortran_nlp/
├── _cpu.cpython-312-...so
├── _cuda.cpython-312-...so
├── lexical/
├── vector_search/
├── auto.py
├── runtime.py
└── calibration.py
```

### External NVIDIA runtime requirements

The CUDA companion uses an external-runtime packaging model.
The wheel does not bundle CUDA, cuBLAS, or NVIDIA HPC SDK runtime libraries. These libraries must be installed separately and discoverable by the Linux dynamic loader.
The validated `0.1.2` environment is:
- Linux x86-64 with glibc 2.34 or newer
- CPython 3.12
- NVIDIA GPU with compute capability 7.5
- compatible NVIDIA driver
- NVIDIA HPC SDK 26.9
- CUDA Runtime 13.3
- cuBLAS 13
- The release wheel was built and validated on an NVIDIA GeForce MX450 with compute capability 7.5.
- Install NVIDIA HPC SDK 26.9 on Ubuntu
- Add the NVIDIA HPC SDK repository:

```bash
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
| sudo gpg --dearmor \
-o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /" \
| sudo tee /etc/apt/sources.list.d/nvhpc.list
```

Install NVIDIA HPC SDK 26.9:

```bash
sudo apt-get update
sudo apt-get install -y nvhpc-26-9
```

Verify the installation:

```bash
/opt/nvidia/hpc_sdk/Linux_x86_64/26.9/compilers/bin/nvfortran \
--version
```

Configure NVIDIA runtime paths
Set the NVIDIA HPC SDK root:

```bash
export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9
```

Add the compiler tools to `PATH`:

```bash
export PATH="$NVHPC_ROOT/compilers/bin:$PATH"
```

Add the required CUDA, cuBLAS, and NVIDIA HPC SDK runtime directories:

```bash
export LD_LIBRARY_PATH="$NVHPC_ROOT/math_libs/13.3/targets/x86_64-linux/lib:$NVHPC_ROOT/cuda/13.3/targets/x86_64-linux/lib:$NVHPC_ROOT/compilers/lib:${LD_LIBRARY_PATH:-}"
```

To apply this configuration automatically in future shells, add the same lines to `~/.bashrc`:

```bash
cat >> ~/.bashrc <<'EOF'
export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9
export PATH="$NVHPC_ROOT/compilers/bin:$PATH"
export LD_LIBRARY_PATH="$NVHPC_ROOT/math_libs/13.3/targets/x86_64-linux/lib:$NVHPC_ROOT/cuda/13.3/targets/x86_64-linux/lib:$NVHPC_ROOT/compilers/lib:${LD_LIBRARY_PATH:-}"
EOF
source ~/.bashrc
```

Installation verification

Check that the NVIDIA driver sees the GPU:
```bash
nvidia-smi
```

Display the GPU name, compute capability, driver version, and memory:
```bash
nvidia-smi \
--query-gpu=name,compute_cap,driver_version,memory.total \
--format=csv
```

Verify both native extensions:
```bash
python - <<'PY'
from fortran_nlp import _cpu, _cuda
print("_cpu:", _cpu.__file__)
print("_cuda:", _cuda.__file__)
print("CPU and CUDA extension imports: PASS")
PY
```

The low-level CUDA extension is imported from:
```python
from fortran_nlp import _cuda
```
It is not imported from `fortran_nlp.vector_search`.

Inspect dynamic-library resolution
Locate the installed CUDA extension:
```bash
cuda_extension="$(
python -c 'from fortran_nlp import _cuda; print(_cuda.__file__)'
)"
```

Inspect its shared-library dependencies:
```bash
ldd "$cuda_extension"
```

Check specifically for unresolved libraries:
```bash
ldd "$cuda_extension" | grep "not found"
```

Expected result:
```text
no output
```

If `libcublas.so.13`, `libcublasLt.so.13`, `libcudart.so.13`, `libnvomp.so`, or `libnvcpumath.so` is not found, check that `LD_LIBRARY_PATH` contains all three required directories:
```bash
printf '%s
' "$LD_LIBRARY_PATH"
```

## Available backends

Lexical search
```text
cuda_fortran -> handwritten CUDA Fortran Jaccard scoring
```

Vector search
```text
cuda_fortran -> handwritten CUDA Fortran IP and squared-L2 scoring
cuda_cublas -> cuBLAS SGEMV and SGEMM scoring
```
Both vector backends reuse the existing deterministic CPU top-k implementation.
CUDA backends are explicit. They are not selected by `backend="auto"`.

Lexical Jaccard usage
```python
from fortran_nlp.lexical import JaccardIndex
documents = [
"reset the account password",
"update contact information",
"create a new insurance application",
]
index = JaccardIndex(
documents,
backend="cuda_fortran",
)
results = index.search(
"how to reset the account password",
k=2,
minimum_score=0.01,
)
for result in results:
print(result)
```

Batched CUDA lexical search uses the same index:
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

Vector-search usage

The public database layout is:
```text
(vector_count, dimension)
```

The public batched-query layout is:
```text
(query_count, dimension)
```

Inputs are converted to `float32`.
Inner product with CUDA Fortran

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
backend="cuda_fortran",
)
scores, indices = index.search(query, k=2)
print(scores)
print(indices)
```

Squared-L2 with CUDA Fortran
```python
from fortran_nlp.vector_search import FlatIndexL2
index = FlatIndexL2(
database,
backend="cuda_fortran",
)
distances, indices = index.search(query, k=2)
print(distances)
print(indices)
```

`FlatIndexL2` returns squared Euclidean distances.
Inner product with cuBLAS
```python
index = FlatIndexIP(
database,
backend="cuda_cublas",
)
scores, indices = index.search(query, k=2)
```

Cosine similarity with cuBLAS
```python
from fortran_nlp.vector_search import FlatIndexCosine
index = FlatIndexCosine(
database,
backend="cuda_cublas",
)
scores, indices = index.search(query, k=2)
```

`FlatIndexCosine` normalizes database vectors and queries before inner-product search. Zero vectors are invalid because cosine normalization is undefined for a zero norm.

Batched vector search
```python
queries = np.array(
[
[1.0, 2.0, 3.0],
[0.0, 1.0, 0.0],
],
dtype=np.float32,
)
scores, indices = index.search_batch(
queries,
k=2,
)
print(scores.shape)
print(indices.shape)
```

The returned shapes are:
```text
(query_count, k)
```

### CUDA vector-search architecture
CUDA vector search in the current release follows this execution model:
```text
GPU scoring
-> copy the complete score array or matrix to the host
-> deterministic CPU top-k
-> return values and zero-based indices
```

The handwritten backend uses one CUDA thread per database vector for single-query scoring and one CUDA thread per query-database pair for batched scoring.
The cuBLAS backend uses:
```text
SGEMV -> single-query inner product
SGEMM -> batched inner product
GPU squared-norm kernels -> L2 norm expansion
GPU L2-combination kernel -> squared-L2 scores
CPU top-k -> deterministic result selection
```

GPU top-k, GPU-resident indexes, CPU/GPU automatic dispatch, multi-GPU execution, and mixed-precision scoring are not included in this release.
Building from source
A source build requires:
- Linux x86-64
- CPython 3.12
- NVIDIA HPC SDK 26.9
- CUDA 13.3
- compatible NVIDIA driver and GPU
- C++17 compiler
- Meson and meson-python
- nanobind 3.1 or newer
- NumPy 2.0 or newer

### Clone the repository:
```bash
git clone https://github.com/suwesh/fortran-nlp-kernels.git
cd fortran-nlp-kernels
```

Build an editable installation with CUDA enabled:
```bash
python -m pip install -e . \
tings=setup-args="-Dfnlp_cuda=true" \
--config-settings=setup-args="-Dfnlp_cuda_arch=cc75" \
--config-settings=setup-args="-Dfnlp_cuda_toolkit=cuda13.3" \
--config-settings=setup-args="-Dfnlp_nvhpc_root=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9"
```

The `fnlp_cuda_arch` value must match the intended GPU target. The published `0.1.2` wheel targets `cc75`.
Verify the editable build:
```bash
python - <<'PY'
from fortran_nlp import _cpu, _cuda
print("_cpu:", _cpu.__file__)
print("_cuda:", _cuda.__file__)
PY
```

Troubleshooting
`_cuda` is unavailable

Error:
```text
backend='cuda_fortran' requires the optional fortran_nlp._cuda extension
```

Confirm that the companion is installed:
```bash
python -m pip show fortran-nlp-kernels-cuda
```

Then test the direct import:
```bash
python -c "from fortran_nlp import _cuda; print(_cuda.__file__)"
```

`libcublas.so.13` is not found
Ensure the cuBLAS directory is on `LD_LIBRARY_PATH`:
```bash
export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9
export LD_LIBRARY_PATH="$NVHPC_ROOT/math_libs/13.3/targets/x86_64-linux/lib:$NVHPC_ROOT/cuda/13.3/targets/x86_64-linux/lib:$NVHPC_ROOT/compilers/lib:${LD_LIBRARY_PATH:-}"
```

`libcudart.so.13` is not found

Confirm that the CUDA runtime exists:
```bash
find "$NVHPC_ROOT/cuda/13.3" \
-name 'libcudart.so.13' \
-print
```

Ensure this directory is configured:
```text
$NVHPC_ROOT/cuda/13.3/targets/x86_64-linux/lib
```

NVIDIA HPC SDK runtime is not found

Check these libraries:
```bash
find "$NVHPC_ROOT/compilers/lib" \
( -name 'libnvomp.so' \
-o -name 'libnvcpumath.so' ) \
-print
```

Ensure this directory is configured:
```text
$NVHPC_ROOT/compilers/lib
```

## Current limitations
- Linux x86-64 only
- CPython 3.12 only
- glibc 2.34 or newer
- NVIDIA compute capability 7.5 target
- external NVIDIA runtime installation required
- CUDA backends must be selected explicitly
- `backend="auto"` remains CPU-only
- deterministic top-k executes on the CPU
- complete score arrays or matrices are copied back to the host
- no GPU-resident database
- no GPU top-k
- no multi-GPU execution
- no mixed-precision scoring

## License

This project is licensed under **GPL-3.0-only**. See [LICENSE](LICENSE).

## Author

**Suwesh Prasad Sah**

- Homepage: <https://suwesh.github.io/>
- Repository: <https://github.com/suwesh/fortran-nlp-kernels>
- Issues: <https://github.com/suwesh/fortran-nlp-kernels/issues>