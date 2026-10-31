# CUDA Vector Search Backend Milestones

## Status

**20% complete**

Target release: `0.1.1`

Current scope: explicit CUDA backends with handwritten CUDA Fortran and cuBLAS scoring. Top-k remains on the CPU using the existing deterministic `fnlp_topk_core`.

## Completed

- [x] Started CUDA vector-search implementation
  - created `vector_search_cuda_core.cuf`
  - created native CUDA core test and runner
- [x] Implemented handwritten single-query CUDA Fortran scoring
  - inner product
  - squared L2
  - one CUDA thread per database vector
  - complete score arrays copied back to the host
- [x] Added public single-query scoring wrappers
  - `ip_database_cuda`
  - `l2_database_cuda`
- [x] Added native CUDA Fortran correctness tests for single-query scoring
  - manually verified scores
  - CPU/CUDA equivalence
  - multiple CUDA blocks
  - one-dimensional vectors
  - zero and negative values
  - empty database

## Remaining

- [ ] Complete CUDA vector-search files
  - `vector_search_cuda_core.cuf`
  - `vector_search_cublas.cuf`
  - `vector_search_cuda_c_api.cuf`
  - `fortran_nlp_cuda_vector_search.h`
- [ ] Implement handwritten CUDA Fortran scoring
  - batched inner product
  - batched squared L2
  - cosine through normalized inner product
- [ ] Integrate existing CPU top-k
  - `topk_largest` for inner product and cosine
  - `topk_smallest` for squared L2
  - deterministic ties and zero-based indices
- [ ] Extend native CUDA Fortran correctness tests
  - batched scoring
  - CPU top-k integration
  - cosine-compatible normalized scoring
- [ ] Implement cuBLAS scoring
  - SGEMV single-query path
  - SGEMM batched-query path
  - squared L2 and cosine support
- [ ] Add CUDA vector-search C ABI
- [ ] Add CUDA vector-search C header
- [ ] Add native C ABI tests
- [ ] Extend the nanobind `_cuda` extension
- [ ] Add low-level Python `_cuda` vector tests
- [ ] Add vector CUDA sources and cuBLAS to the optional Meson build
- [ ] Add explicit public vector backends
  - `cuda_fortran`
  - `cuda_cublas`
- [ ] Add public CUDA vector-index tests
- [ ] Verify CUDA-enabled and CPU-only builds
- [ ] Run the complete regression suite
- [ ] Add README usage notes
- [ ] Final CUDA vector-search cleanup and commit

## Backend behavior

```text
fortran         -> CPU serial
fortran_openmp  -> CPU OpenMP
openblas        -> CPU OpenBLAS
auto            -> existing CPU workload-aware resolver
cuda_fortran    -> handwritten CUDA Fortran scoring + CPU top-k
cuda_cublas     -> cuBLAS scoring + CPU top-k
```

## Execution path for 0.1.1
```text
GPU scoring
-> copy full score array to host
-> existing deterministic CPU top-k
-> return scores and zero-based indices
```

## Deferred
- GPU top-k selection
- fused scoring and top-k
- persistent GPU-resident database
- persistent cuBLAS handles and device buffers
- pinned or zero-copy transfers
- GPU integration into backend="auto"
- CPU/GPU crossover policy
- multi-GPU execution
- mixed-precision scoring
- CUDA performance tuning and benchmarks

