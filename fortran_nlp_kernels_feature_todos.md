# `fortran-nlp-kernels` Feature TODOs

**Updated:** 22 October 2026  
**Current focus:** Begin reproducible CPU performance evaluation using `fortran-nlp-kernels==0.1.0b2` before starting the NVIDIA GPU phase.

## Immediate Next: BLAS-Backed Vector Search

- [x] Implement `src/native/cpu/vector_search/vector_search_blas.f90`
- [x] Add single-query inner-product scoring with `SGEMV`
- [x] Add future batched-query scoring path with `SGEMM`
- [x] Reuse exact serial top-k after BLAS score computation
- [x] Expose BLAS routines through `vector_search_c_api.f90`
- [x] Add C declarations to `fortran_nlp_vector_search.h`
- [x] Expose `backend="fortran_blas"` through nanobind
- [x] Add BLAS dependency and thread controls to Meson
- [x] Validate BLAS results against NumPy and existing Fortran kernels
- [ ] Prevent OpenMP/BLAS thread oversubscription during benchmarks

## Lexical Module: Remaining Product Features

- [ ] Add optional user-provided document IDs to `JaccardIndex`
- [x] Use positional IDs when custom IDs are omitted
- [ ] Validate that `document_ids` and `corpus_texts` have equal lengths
- [ ] Return the supplied ID through `JaccardResult.document_id`
- [ ] Add tests for string IDs, integer IDs, and default positional IDs
- [x] Add empty-corpus, empty-query, short-text, identical-document, and duplicate-document tests
- [x] Add low-level malformed-offset validation tests
- [x] Add README installation and `JaccardIndex` usage examples
- [x] (Batched Lexical Search) Implement and validate packed multi-candidate Jaccard corpus search using OpenMP, including candidate offsets, C ABI and nanobind integration, `JaccardIndex.search_batch()`, one-thread and multithreaded execution, repeated single-candidate equivalence, and native-to-public API tests.

## Vector-Search Module: Remaining Product Features

- [x] `FlatIndexIP` with serial and OpenMP Fortran backends
- [x] `FlatIndexL2` with serial and OpenMP Fortran backends
- [x] `FlatIndexCosine` using normalization plus inner product
- [x] Exact serial top-k selection
- [x] C ABI and nanobind integration
- [x] NumPy correctness references and public API tests
- [ ] Add optional user-provided vector IDs to all flat indexes
- [x] Use positional zero-based IDs when custom IDs are omitted
- [x] Add batched-query APIs
- [x] Add `backend="fortran_blas"`
- [ ] Later optimize top-k from `O(Nk)` to a heap or partial-selection implementation
- [ ] Later evaluate OpenMP-local top-k plus global merge
- [ ] Add handwritten serial batched-query implementations
- [ ] Add handwritten OpenMP batched-query implementations

## CPU Packaging and Release

- [x] Run the complete test suite after BLAS integration
- [x] Build and install-test a Linux x86-64 wheel
- [x] Build and inspect the source distribution
- [ ] Test installation in a clean Linux Python environment
- [x] Verify `libgomp` and BLAS runtime dependencies
- [x] Add CPU-only continuous integration
- [x] Complete README API examples for lexical and vector search
- [x] Publish the first CPU-focused pre-release to PyPI

## CPU Performance Evaluation

### Lexical

- [x] Validate exact serial, OpenMP, and batched OpenMP equivalence
- [x] Evaluate serial and OpenMP physical-core scaling
- [x] Identify serial/OpenMP crossover behavior
- [x] Measure latency, throughput, speedup, parallel efficiency, candidate-size sensitivity, and estimated native memory growth
- [ ] Publish reproducible configurations, raw measurements, processed summaries, benchmark tables, and figures for single-candidate search
- [ ] Evaluate OpenMP batched-candidate scaling
- [ ] Evaluate 6-, 8-, and 12-thread SMT sensitivity
- [ ] Measure native-binding versus public-API latency
- [ ] Derive benchmark-informed automatic backend thresholds

### Vector Search

- [ ] NumPy score-computation baseline
- [ ] FAISS CPU `IndexFlatIP` and `IndexFlatL2` baselines
- [ ] Handwritten Fortran serial/OpenMP benchmarks
- [ ] BLAS-backed Fortran benchmarks
- [ ] Separate scoring time from top-k time
- [ ] Single-query and batched-query scaling
- [ ] Control `OMP_NUM_THREADS` and BLAS thread counts independently

## NVIDIA GPU Phase

### Lexical

- [ ] CUDA Fortran exact Jaccard kernel
- [ ] CUDA C++ equivalent for controlled comparison
- [ ] GPU-resident packed corpus
- [ ] Cold, warm, and resident-corpus measurements

### Vector Search

- [ ] Custom CUDA Fortran inner-product kernel
- [ ] Custom CUDA Fortran squared-L2 kernel
- [ ] cuBLAS-backed Fortran implementation
- [ ] Exact GPU top-k
- [ ] FAISS GPU exact-flat baselines
- [ ] GPU-resident database and batched-query support

## Research Artifact

- [ ] Freeze benchmark methodology and hardware/software metadata
- [ ] Measure kernel-only, binding, and end-to-end latency separately
- [ ] Report throughput, speedup, parallel efficiency, memory use, and crossover points
- [ ] Publish reproducible configurations, raw results, tables, and figures
- [ ] C++ serial equivalent
- [ ] C++ OpenMP equivalent
- [ ] Write the technical report:

> **Fortran Kernels for NLP: Exact Lexical and Vector Similarity Search Across Multicore CPUs and NVIDIA GPUs**

## FORTRESS Integration

- [ ] Replace duplicated FORTRESS native lexical code with `JaccardIndex`
- [ ] Use custom document IDs for SOP/document identity
- [ ] Integrate `FlatIndexCosine` or `FlatIndexIP` for exhaustive semantic analysis
- [ ] Keep FORTRESS as the motivating application and downstream consumer
