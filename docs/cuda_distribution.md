docs/cuda_distribution.md

# CUDA Distribution and Publishing Strategy

## Decision

The project will use two synchronized Python distributions built from the same repository, commit, and release tag:

```text
fortran-nlp-kernels
fortran-nlp-kernels-cuda
```

The base distribution remains lightweight and CPU-capable. The CUDA companion provides the optional _cuda native extension.

Both distributions use the same version:

fortran-nlp-kernels==0.1.1
fortran-nlp-kernels-cuda==0.1.1

User installation
CPU only
pip install fortran-nlp-kernels

CUDA companion directly
pip install fortran-nlp-kernels-cuda


The CUDA distribution depends on the matching base version, so pip also installs the base package when required.

CUDA through the base-package extra
pip install "fortran-nlp-kernels[cuda]"


The [cuda] extra installs the same fortran-nlp-kernels-cuda distribution.

Both CUDA installation commands produce the same final environment:

fortran-nlp-kernels
fortran-nlp-kernels-cuda


If the matching base package is already installed, pip installs only the CUDA companion wheel.

Installed Python layout

The distributions use separate internal package ownership:

site-packages/
├── fortran_nlp/
│   ├── __init__.py
│   ├── lexical/
│   ├── vector_search/
│   ├── auto.py
│   ├── runtime.py
│   └── _cpu<EXT_SUFFIX>
│
└── fortran_nlp_cuda/
    ├── __init__.py
    └── _cuda<EXT_SUFFIX>


The base package discovers the optional CUDA companion internally:

try:
    from fortran_nlp_cuda import _cuda
except ImportError:
    _cuda = None


Users continue using the unified public API:

from fortran_nlp.lexical import JaccardIndex

index = JaccardIndex(
    documents,
    backend="cuda_fortran",
)


Users do not need to import fortran_nlp_cuda directly.

Source and packaging boundaries

All implementation code remains in the central source tree:

src/
├── fortran_nlp/
└── native/
    ├── cpu/
    ├── cuda/
    │   ├── lexical/
    │   └── vector_search/
    ├── bindings/
    │   ├── cpu_bindings.cpp
    │   └── cuda_bindings.cpp
    └── include/


The CUDA packaging project does not duplicate kernels or bindings. It defines only how the CUDA companion wheel is assembled:

packaging/
└── cuda/
    ├── pyproject.toml
    ├── meson.build
    ├── meson.options
    ├── README.md
    └── src/
        └── fortran_nlp_cuda/
            └── __init__.py


The CUDA packaging build references the canonical implementation under:

src/native/cuda/
src/native/bindings/cuda_bindings.cpp
src/native/include/

Base distribution contents

The root packaging project produces:

fortran-nlp-kernels


It contains:

public Python API;
_cpu native extension;
CPU serial and OpenMP backends;
OpenBLAS vector backends;
CPU autotuning and calibration;
workload-aware CPU dispatch;
optional CUDA backend discovery;
[cuda] convenience dependency metadata.

It does not contain:

_cuda extension;
CUDA or NVHPC runtime libraries;
CUDA native objects;
CUDA compiler tooling.
CUDA companion contents

The CUDA packaging project produces:

fortran-nlp-kernels-cuda


It contains:

fortran_nlp_cuda/__init__.py;
_cuda native extension;
required redistributable CUDA or NVHPC runtime libraries, if necessary;
CUDA-specific package metadata.

It does not contain:

_cpu extension;
duplicate public Python APIs;
CPU policies or calibration;
nvfortran;
the complete NVIDIA HPC SDK;
the NVIDIA driver;
source files, object files, or Fortran module files.
Package metadata
Base package

The root pyproject.toml declares:

[project]
name = "fortran-nlp-kernels"
version = "0.1.1"

[project.optional-dependencies]
cuda = [
    "fortran-nlp-kernels-cuda==0.1.1",
]

CUDA companion

packaging/cuda/pyproject.toml declares:

[project]
name = "fortran-nlp-kernels-cuda"
version = "0.1.1"

dependencies = [
    "fortran-nlp-kernels==0.1.1",
]


Exact version matching is initially required to prevent incompatible base and CUDA extensions from being combined.

Compiler-level build flow
Base wheel
GFortran
    -> CPU Fortran kernels and C ABI

g++
    -> CPU nanobind adapter

Meson
    -> _cpu shared extension

wheel repair
    -> eligible OpenBLAS and Fortran runtime dependencies

PyPI
    -> fortran-nlp-kernels

CUDA companion wheel
nvfortran
    -> CUDA Fortran kernels
    -> CUDA Fortran C ABI
    -> cuBLAS interfaces

g++
    -> CUDA nanobind adapter
    -> nanobind runtime

Meson
    -> _cuda shared extension

dependency audit
    -> CUDA and NVHPC runtime requirements

wheel packaging and repair
    -> permitted runtime dependencies

PyPI
    -> fortran-nlp-kernels-cuda


nvfortran is a build-time dependency. Users installing a prebuilt CUDA wheel must not need the compiler.

Runtime requirements

A prebuilt CUDA companion requires:

a compatible NVIDIA GPU;
a compatible NVIDIA driver;
required runtime libraries not bundled in the wheel.

The wheel must not bundle:

nvfortran
full NVIDIA HPC SDK
NVIDIA display driver
libcuda.so.1
build objects
Fortran .mod files


libcuda.so.1 must be provided by the installed NVIDIA driver.

Before publication, inspect the final extension with:

ldd path/to/_cuda<EXT_SUFFIX>
readelf -d path/to/_cuda<EXT_SUFFIX>
du -h path/to/_cuda<EXT_SUFFIX>


The final runtime and redistribution policy will be decided after lexical CUDA, vector CUDA, and cuBLAS are all present in _cuda.

Development build

Development uses one repository and one optional Meson CUDA configuration:

python -m pip install -e . \
  --no-build-isolation \
  -Csetup-args=-Dfnlp_cuda=true \
  -Csetup-args=-Dfnlp_cuda_arch=cc86 \
  -Csetup-args=-Dfnlp_cuda_toolkit=cuda13.3


CPU-only development remains available:

python -m pip install -e . \
  --no-build-isolation \
  -Csetup-args=-Dfnlp_cuda=false


This development arrangement builds _cpu and _cuda together. It does not determine the final wheel boundary.

Git and release model

CUDA does not use a permanent branch.

Development occurs on the temporary:

cuda-fortran


branch.

Before release:

complete lexical CUDA;
complete vector CUDA;
complete cuBLAS backends;
create and validate packaging/cuda/;
test both distributions in clean environments;
merge cuda-fortran into main;
create one release tag;
build both distributions from that tag;
publish the base distribution first;
publish the CUDA companion second;
smoke-test both installation paths;
delete the feature branch.

One tag produces both distributions:

v0.1.1
├── fortran-nlp-kernels==0.1.1
└── fortran-nlp-kernels-cuda==0.1.1

Release build flow
Base distribution

Build with CUDA excluded:

python -m build \
  -Csetup-args=-Dfnlp_cuda=false


Upload to the existing PyPI project:

python -m twine upload \
  dist/fortran_nlp_kernels-0.1.1-*

CUDA companion

Build using the separate CUDA packaging project:

cd packaging/cuda
python -m build


Upload to the new PyPI project:

python -m twine upload \
  dist/fortran_nlp_kernels_cuda-0.1.1-*


The distribution identity comes from the relevant pyproject.toml. The root package name is never edited back and forth during release.

Publication order

Publish:

1. fortran-nlp-kernels==0.1.1
2. fortran-nlp-kernels-cuda==0.1.1


The base package must exist first because the CUDA companion has an exact dependency on it.

Clean-environment release tests

Test these installation paths separately:

pip install fortran-nlp-kernels

pip install fortran-nlp-kernels-cuda

pip install "fortran-nlp-kernels[cuda]"


Validate:

CPU-only import and execution;
CUDA installation from an empty environment;
CUDA installation when the base package already exists;
CUDA unavailable error from the CPU-only environment;
lexical and vector CPU/CUDA equivalence;
installation without nvfortran;
clean uninstall and upgrade behavior.
CUDA architecture policy

Development currently targets:

cc86
cuda13.3


This is appropriate for the RTX 3060 development machine but is too narrow for a general public wheel.

Before release, decide and test:

supported CUDA family;
supported GPU compute capabilities;
fat-binary architecture list;
PTX forward-compatibility strategy;
extension-size impact;
clean runtime behavior on each supported GPU generation.

The first CUDA wheel must clearly document its actual hardware and runtime compatibility.

Future distribution family

The architecture can later expand to:

fortran-nlp-kernels
fortran-nlp-kernels-cuda12x
fortran-nlp-kernels-cuda13x
fortran-nlp-kernels-rocm
fortran-nlp-kernels-oneapi


All accelerator packages remain implementation companions. Users continue using the unified fortran_nlp public API.

Deferred packaging options

The following remain future options:

accelerator-specific custom wheel indexes;
hardware-aware wheel variants;
automatic installer selection by GPU and driver;
independently versioned accelerator ABI compatibility.

The first CUDA release will use conventional synchronized companion packages on PyPI.


This documents the locked approach: shared implementation under `src/`, separate distribution manifests, one repository and tag, synchronized base/CUDA versions, two PyPI projects, and both direct and `[cuda]` installation paths.