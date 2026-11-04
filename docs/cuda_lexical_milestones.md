# CUDA Lexical Backend Milestones

## Status

**~100% complete**
**Complete for the explicit lexical CUDA scope**

Current scope: explicit `backend="cuda_fortran"` only. GPU auto-dispatch and performance policy are deferred.

## Completed

- [x] CUDA Fortran lexical core
  - pairwise device Jaccard routine
  - single-candidate corpus kernel
  - batched candidate-document kernel
  - host/device transfers, synchronization, and CUDA error checks
- [x] Native CUDA Fortran correctness tests
- [x] CUDA Fortran C ABI
- [x] CUDA lexical C header
- [x] Native CUDA C ABI tests
- [x] Nanobind `_cuda` extension
- [x] Low-level Python `_cuda` tests: **13 passed**
- [x] Explicit `backend="cuda_fortran"` added to `JaccardIndex`
- [x] Optional Meson CUDA build
  - `_cpu` and `_cuda` generated under `build/cp312/`
  - configurable CUDA architecture and toolkit
- [x] Editable CUDA development build using `0.1.1.dev0`
- [x] Add `tests/test_cuda_jaccard_index.py`
  - backend accepted and retained
  - single search matches CPU
  - batch search matches CPU
  - `k` and `minimum_score`
  - deterministic ties
  - empty candidate
  - batch size one
  - clear error when `_cuda` is unavailable
- [x] Full regression suite: **228 passed**
- [x] Verify CPU-only build with `-Dfnlp_cuda=false`
- [x] Add brief README CUDA build and usage notes
- [x] Final lexical CUDA cleanup and commit

## Remaining

## Backend behavior

```text
fortran         -> CPU serial
fortran_openmp  -> CPU OpenMP
auto            -> existing CPU workload-aware resolver
cuda_fortran    -> explicit CUDA Fortran execution
```

## Deferred

- GPU integration into `backend="auto"`
- CPU/GPU crossover policy
- persistent GPU-resident corpus
- GPU memory guardrails
- RTX 3060 lexical performance evaluation
- CUDA distribution strategy