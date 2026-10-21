# Changelog

All notable changes to `fortran-nlp-kernels` will be documented in
this file.

The project follows semantic versioning. During the beta period, public
APIs may evolve before the stable `0.1.0` release.

## [Unreleased]

### Added

- Automatic backend selection through `backend="auto"`.
  - `JaccardIndex` resolves `auto` to `fortran_openmp`.
  - `FlatIndexIP`, `FlatIndexL2`, and `FlatIndexCosine` resolve `auto`
    to `fortran_blas`.
- Comprehensive lexical edge-case tests.
- Comprehensive vector-search edge-case tests.
- Validation for empty, malformed, noncontiguous, and dtype-converted
  inputs.
- Deterministic lower-positional-ID tie handling.
- Zero-vector validation for cosine similarity.
- Package version exposure through `fortran_nlp.__version__`.
- Linux CPU GitHub Actions test matrix.
- Automated `manylinux_2_28` wheel builds for CPython 3.10, 3.11,
  3.12, and 3.13 through `cibuildwheel`.

### Changed

- Descending NumPy reference ranking now preserves deterministic
  positional ordering for tied similarity scores.
- Development and release validation now cover 79 Python tests.
- Linux wheels are built and tested across CPython 3.10–3.13.

### Planned

- Benchmark-informed automatic backend selection.
- Optional document and vector IDs.
- Handwritten serial and OpenMP batched-query backends.
- Optimized exact top-k selection.
- Automated PyPI Trusted Publishing.
- CPU performance evaluation and CUDA Fortran backends.

## [0.1.0b1] - 2026-10-21

### Added

- First public Linux CPU beta.
- `JaccardIndex` for exact word-shingle Jaccard similarity search.
- Python lexical preprocessing with:
  - Unicode NFKC normalization;
  - case folding;
  - fixed-size word shingles;
  - deterministic BLAKE2b 64-bit hashes;
  - sorted and unique `int64` representations;
  - flattened corpus storage and document offsets.
- Serial Fortran lexical scoring.
- OpenMP Fortran lexical scoring.
- `FlatIndexIP` for exact inner-product search.
- `FlatIndexL2` for exact squared-Euclidean-distance search.
- `FlatIndexCosine` through L2 normalization and inner-product search.
- Handwritten serial Fortran vector-search kernels.
- Handwritten OpenMP Fortran vector-search kernels.
- OpenBLAS `SGEMV` single-query search.
- OpenBLAS `SGEMM` batched-query search.
- Exact serial top-k selection with zero-based positional IDs.
- Stable C ABI adapters for lexical and vector-search kernels.
- nanobind Python extension.
- Meson and meson-python build system.
- Editable installation through `pip install -e .`.
- NumPy correctness references.
- Native Fortran, C ABI, binding, and public Python API tests.
- GPL-3.0-only license metadata.
- CPython 3.12 `manylinux_2_38_x86_64` wheel.
- Source distribution published to PyPI.

### Known limitations

- Result identifiers are zero-based positional indexes.
- Batched search requires `backend="fortran_blas"`.
- Exact top-k selection currently uses an `O(Nk)` implementation.
- CUDA and GPU backends are not included.
- The initial binary wheel targets Linux x86-64 and CPython 3.12.

[Unreleased]: https://github.com/suwesh/fortran-nlp-kernels/compare/v0.1.0b1...HEAD
[0.1.0b1]: https://github.com/suwesh/fortran-nlp-kernels/releases/tag/v0.1.0b1