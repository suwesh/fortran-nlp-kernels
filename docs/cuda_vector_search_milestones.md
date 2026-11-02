# CUDA Vector Search Backend Milestones

## Status

**70% complete**

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
- [x] Added native CUDA Fortran correctness tests
  - manually verified scores
  - single-query CPU/CUDA equivalence
  - batched score-matrix CPU/CUDA equivalence
  - batch size one
  - single and batched top-k equivalence
  - deterministic ties and zero-based indices
  - multiple CUDA blocks
  - one-dimensional vectors
  - zero and negative values
  - empty database
- [x] Integrated existing CPU top-k
  - `topk_largest` for inner product and cosine
  - `topk_smallest` for squared L2
  - deterministic ties and zero-based indices
- [x] Implemented handwritten batched CUDA Fortran scoring
  - batched inner product
  - batched squared L2
  - one CUDA thread per query-database pair
  - complete score matrices copied back to the host
  - deterministic CPU top-k per query
- [x] Implemented cuBLAS scoring
  - SGEMV single-query inner product
  - SGEMM batched inner product
  - GPU-native squared L2 through norm expansion
  - custom CUDA squared-norm and L2-combination kernels
  - existing deterministic CPU top-k integration
- [x] Added native cuBLAS correctness tests
  - manually verified scores
  - single-query SGEMV CPU/cuBLAS equivalence
  - batched SGEMM CPU/cuBLAS equivalence
  - single and batched top-k equivalence
  - batch size one
  - deterministic ties and zero-based indices
  - one-dimensional vectors
  - zero and negative values
  - large single-query database
  - large query batch
  - nonnegative L2 rounding clamp
- [x] Complete CUDA vector-search files
  - `vector_search_cuda_core.cuf`
  - `vector_search_cublas.cuf`
  - `vector_search_cuda_c_api.cuf`
  - `fortran_nlp_cuda_vector_search.h`\
- [x] Add CUDA vector-search C ABI
- [x] Add CUDA vector-search C header
- [x] Add native C ABI tests
- [x] Extend the nanobind `_cuda` extension

## Remaining

- [ ] Route normalized cosine inputs through handwritten CUDA inner product
- [ ] Add cosine-compatible normalized scoring tests
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

## cuBLAS implementation order:
Recommended order
Plain Text
1. ip_database_cublas
-> learn and verify SGEMV
 
2. ip_search_cublas
-> SGEMV + CPU top-k
 
3. l2_database_cublas
-> norm kernels + same SGEMV + L2 combination kernel
 
4. l2_search_cublas
-> L2 scoring + CPU top-k

Then batch:

Plain Text
5. ip_database_batch_cublas
-> SGEMM
 
6. ip_search_batch_cublas
 
7. l2_database_batch_cublas
-> norm kernels + SGEMM + L2 combination kernel
 
8. l2_search_batch_cublas