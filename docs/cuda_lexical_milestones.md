Yes, your understanding is correct. The initial CUDA Fortran lexical implementation is complete through the native core, C ABI, and low-level _cuda extension. What remains is Python integration, public API wiring, tests, and eventually proper optional packaging.

Current completion: about 65%

For explicit lexical CUDA support only, not GPU autotuning or GPU packaging:

CUDA Fortran implementation       100%
Native correctness validation     100%
C ABI                             100%
nanobind _cuda binding             90%
Low-level Python tests              0%
Public JaccardIndex integration     0%
Public CUDA tests                   0%
Optional package/build design       0%


The low-level binding is 90%, rather than 100%, because _cuda imports successfully but its functions have not yet been called and validated from Python.

What is already complete
1. CUDA Fortran lexical core
src/native/cuda/lexical/lexical_cuda_core.cuf


Implemented:

Private:
- jaccard_sorted_device
- jaccard_corpus_kernel
- jaccard_batch_corpus_kernel

Public:
- jaccard_corpus_cuda
- jaccard_batch_corpus_cuda


Supported:

single candidate against the corpus
multiple packed candidates against the corpus
empty candidate
empty document
batch size one
int64-safe flattened pair indexing
host-to-device transfers
device-to-host transfers
CUDA launch and execution checks
2. Native CUDA Fortran tests
tests/native/cuda/test_lexical_cuda_core.cuf
tests/native/cuda/run_lexical_cuda_core_test.sh


All passing against the CPU implementation.

3. CUDA C ABI
src/native/cuda/lexical/lexical_cuda_c_api.cuf
src/native/include/fortran_nlp_cuda_lexical.h


Exports:

fnlp_jaccard_corpus_cuda()
fnlp_jaccard_batch_corpus_cuda()

4. Native C ABI tests
tests/native/cuda/test_lexical_cuda_c_api.c
tests/native/cuda/run_lexical_cuda_c_api_test.sh


All passing against manually specified expected values.

5. Low-level nanobind extension
src/native/bindings/cuda_bindings.cpp


Built as:

src/fortran_nlp/_cuda.cpython-312-x86_64-linux-gnu.so


Successfully imported with:

import fortran_nlp._cuda


Exposed functions:

_cuda.jaccard_corpus_cuda(...)
_cuda.jaccard_batch_corpus_cuda(...)

What happens next
Milestone 1: Direct Python _cuda tests

Create:

tests/test_cuda_lexical_native.py


This tests the low-level binding directly:

from fortran_nlp import _cuda


Test:

single-candidate outputs
empty candidate
batched outputs
batch size one
NumPy dtypes
output shapes
Fortran-contiguous batch matrices
invalid offsets
wrong dtype behavior
noncontiguous input handling

This completes validation of:

Python
→ nanobind
→ C ABI
→ CUDA Fortran
→ GPU
→ NumPy outputs


After this: about 72% complete.

Milestone 2: Add explicit CUDA backend to JaccardIndex

The integration belongs primarily in:

src/fortran_nlp/lexical/index.py


Currently:

from fortran_nlp import _cpu


Do not replace that with a mandatory _cuda import, because CPU-only installations must continue working.

Use an optional import:

try:
    from fortran_nlp import _cuda
except ImportError:
    _cuda = None


Then add:

SUPPORTED_BACKENDS = {
    "auto",
    "fortran",
    "fortran_openmp",
    "cuda_fortran",
}


For 0.1.0, the existing CPU routes remain unchanged:

fortran         → _cpu
fortran_openmp  → _cpu
auto            → current CPU resolver


The new explicit route becomes:

cuda_fortran    → _cuda

Single search

Conceptually:

if decision.backend == "cuda_fortran":
    intersections, unions, scores = (
        _cuda.jaccard_corpus_cuda(
            candidate_shingles,
            self._corpus_shingles,
            self._document_offsets,
        )
    )

Batch search

Conceptually:

if decision.backend == "cuda_fortran":
    intersections, unions, scores = (
        _cuda.jaccard_batch_corpus_cuda(
            packed_candidates,
            candidate_offsets,
            self._corpus_shingles,
            self._document_offsets,
        )
    )


This public API remains:

index = JaccardIndex(
    corpus_texts,
    backend="cuda_fortran",
)

results = index.search(candidate, k=10)


The user does not call _cuda directly.

After this: about 82% complete.

Yes, explicit override only for now

Correct. Do not change auto.py yet.

Initial behavior:

backend="fortran"         → CPU serial
backend="fortran_openmp"  → CPU OpenMP
backend="cuda_fortran"    → CUDA Fortran
backend="auto"            → existing CPU automatic dispatch


This means GPU use is opt-in:

JaccardIndex(
    corpus,
    backend="cuda_fortran",
)


Automatic CPU/GPU switching comes later, after benchmarks establish reliable crossover points.

Do not add GPU rules to:

src/fortran_nlp/auto.py
src/fortran_nlp/policies/cpu_fallback_v1.json
src/fortran_nlp/calibration.py


Those remain CPU-only for now.

Milestone 3: Public Python tests

Create:

tests/test_cuda_jaccard_index.py


Test through the actual public API:

from fortran_nlp.lexical import JaccardIndex


Coverage:

backend="cuda_fortran" accepted
.backend == "cuda_fortran"
single search equals CPU serial
batch search equals CPU/OpenMP
empty candidate handling
minimum_score
k
tie ordering
batch size one
CUDA unavailable error
_cuda missing from CPU-only installation

After this: about 90% complete.

lexical/__init__.py

No change is required.

Keep:

from .index import JaccardIndex, JaccardResult

__all__ = [
    "JaccardIndex",
    "JaccardResult",
]


CUDA is an implementation backend of JaccardIndex, not a separate public class.

Users continue importing:

from fortran_nlp.lexical import JaccardIndex


They should not need:

from fortran_nlp import _cuda

Top-level fortran_nlp/__init__.py

No user-facing CUDA import should be added there.

Keep:

from importlib.metadata import (
    PackageNotFoundError,
    version,
)

try:
    __version__ = version("fortran-nlp-kernels")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
]


Why?

Because this would be dangerous:

from . import _cuda


It would make:

import fortran_nlp


fail on systems without CUDA.

CUDA support must remain optional.

Milestone 4: CUDA availability handling

Add a small internal helper, either inside index.py initially or later in:

src/fortran_nlp/cuda.py


Behavior:

try:
    from fortran_nlp import _cuda
except ImportError:
    _cuda = None


When the user explicitly requests CUDA but it is unavailable:

raise RuntimeError(
    "backend='cuda_fortran' requires the optional "
    "fortran_nlp._cuda extension and a compatible NVIDIA GPU."
)


Important distinction:

import fortran_nlp
    → always succeeds

backend="fortran"
    → always uses CPU extension

backend="cuda_fortran"
    → checks CUDA extension availability

Milestone 5: Reusable development build

Currently:

tests/native/cuda/build_cuda_extension.sh


creates _cuda directly under:

src/fortran_nlp/


That is fine during development.

Before calling lexical CUDA support complete, improve it to include:

configurable compute capability
configurable CUDA toolkit version
current Python extension suffix
clean rebuild
import check
direct Python CUDA test execution

For example, eventual environment overrides:

CUDA_ARCH="${CUDA_ARCH:-cc86}"
CUDA_VERSION="${CUDA_VERSION:-cuda13.3}"


Then:

-gpu="${CUDA_ARCH},${CUDA_VERSION}"


After this: about 95% complete.

Milestone 6: Packaging decision

This is separate from implementing the backend.

The existing CPU wheel should remain independent of CUDA.

Possible future design:

fortran-nlp-kernels
    → CPU package

fortran-nlp-kernels-cuda
    → optional CUDA companion package


Or:

CPU wheel
    → PyPI

CUDA extension
    → source build or separate release artifact


Do not solve this yet. First complete and test the backend locally.

The local CUDA backend can be considered technically complete without a public CUDA wheel.

Final roadmap
Completed
[100%] CUDA Fortran lexical core
[100%] Native CUDA core correctness tests
[100%] CUDA Fortran C ABI
[100%] Native CUDA C ABI tests
[ 90%] Low-level nanobind _cuda extension

Next
[  0%] Direct Python _cuda tests
[  0%] Explicit backend="cuda_fortran" in JaccardIndex
[  0%] Public Python CUDA tests
[  0%] Optional CUDA availability handling
[  0%] Repeatable development build cleanup

Deferred
GPU integration into backend="auto"
CPU/GPU crossover rules
persistent device-resident corpus
CUDA calibration
GPU memory guardrails
CUDA wheel publishing
lexical GPU performance benchmarking

Immediate next file
tests/test_cuda_lexical_native.py


Then:

src/fortran_nlp/lexical/index.py


So yes, the core CUDA Fortran implementation is done. We are now crossing from low-level GPU implementation into the user-facing Python backend integration.