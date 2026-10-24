# Changelog

All notable changes to `fortran-nlp-kernels` will be documented in
this file.

The project follows semantic versioning. Development continues beyond the stable `0.1.0` release, with future versions introducing new features, performance improvements, and compatibility updates.

## [Unreleased]

## [0.1.0] - 2026-10-24

### Added

- Benchmark-informed CPU backend selection through `backend="auto"`.
- Runtime workload identification for lexical and vector-search operations.
- Three-level runtime-dispatch precedence:
  1. explicit backend override;
  2. compatible machine calibration profile;
  3. packaged fallback policy.
- Immutable and versioned `cpu-fallback-v1` policy distributed inside
  source and binary packages.
- `BackendDecision` records containing the selected backend, execution
  strategy, policy source, reason, and thread policy.
- Workload-aware lexical dispatch across:
  - serial Fortran;
  - OpenMP Fortran;
  - repeated serial execution;
  - repeated OpenMP execution;
  - native batched OpenMP execution.
- Workload-aware vector dispatch across:
  - serial Fortran;
  - OpenMP Fortran;
  - repeated serial execution;
  - repeated OpenMP execution;
  - single-query OpenBLAS `SGEMV`;
  - repeated OpenBLAS `SGEMV`;
  - native batched OpenBLAS `SGEMM`.
- Benchmark-informed vector policies:
  - inner-product and cosine single queries use OpenBLAS `SGEMV`;
  - small inner-product and cosine batches use repeated `SGEMV`;
  - larger inner-product and cosine batches use `SGEMM`;
  - squared-L2 single queries use OpenMP;
  - squared-L2 batches use repeated OpenMP.
- Runtime memory guardrails for:
  - native batched lexical result matrices;
  - vector `SGEMM` score matrices and top-k outputs.
- Safe fallback to repeated execution when automatic native batching
  exceeds the configured memory budget or available memory cannot be
  determined.
- Machine fingerprinting based on:
  - operating system;
  - machine architecture;
  - processor identifier;
  - available logical CPU count;
  - available physical-core count.
- Linux physical-core discovery through CPU topology information exposed
  by sysfs.
- Versioned machine calibration profiles stored outside the installed
  package.
- Profile compatibility checks covering package version and machine
  fingerprint.
- Atomic calibration-profile persistence.
- OpenMP runtime controls exposed through the native extension.
- OpenBLAS runtime controls exposed through the native extension.
- Temporary OpenMP and OpenBLAS thread-policy context managers with
  guaranteed restoration after normal execution or exceptions.
- Bounded, optional CPU thread calibration across representative:
  - lexical OpenMP workloads;
  - squared-L2 OpenMP workloads;
  - OpenBLAS `SGEMV` workloads;
  - OpenBLAS `SGEMM` workloads.
- Calibration evidence containing candidate thread counts, selected thread
  counts, median timings, warm-up count, repeat count, and crossover-policy
  source.
- Calibration console command: `fortran-nlp-calibrate --show`
- Calibration profile reset: `fortran-nlp-calibrate --reset`
- Custom calibration-profile paths through `--output` and `FORTRAN_NLP_PROFILE`.
- Equivalent module execution through: `python -m fortran_nlp.calibration`
- Runtime, policy, profile, calibration-management, lexical-dispatch, vector-dispatch, memory-guardrail, and thread-restoration tests.

### Changed

- `backend="auto"` is now preserved during index construction and resolved separately for each search operation.
- `JaccardIndex.backend`, `FlatIndexIP.backend`, `FlatIndexL2.backend`, and `FlatIndexCosine.backend` now report the backend requested by the user.
- Explicit backend requests bypass automatic policy selection and preserve user-controlled thread-runtime settings.
- Explicit `backend="fortran"` lexical batches now use repeated serial execution.
- Explicit `backend="fortran_openmp"` lexical batches now use repeated OpenMP execution.
- Explicit serial and OpenMP vector batches are now supported through repeated single-query execution.
- Automatic lexical dispatch now selects between serial, OpenMP, repeated, and native-batch execution according to the workload and active policy.
- Automatic inner-product and cosine dispatch now selects between `SGEMV` and `SGEMM` according to query count and available memory.
- Automatic squared-L2 dispatch now prefers the measured OpenMP paths instead of selecting the BLAS implementation by default.
- Automatic OpenMP and OpenBLAS operations now apply bounded thread policies temporarily and restore the previous runtime configuration afterward.
- Machine calibration tunes thread counts while retaining the shipped, benchmark-informed workload crossover policy.
- Source and binary packages now include the automatic-dispatch modules, runtime controller, profile manager, calibration command, and packaged fallback policy.
- Development and release validation now cover 205 Python tests.
- Package metadata and Meson project version were promoted from 0.1.0b2 to stable 0.1.0.

### Fixed
- Automatic backend selection no longer permanently resolves to one backend during index construction.
- Small lexical workloads no longer incur OpenMP overhead solely because `backend="auto"` was requested.
- Automatic squared-L2 search no longer defaults to the slower BLAS path on workloads represented by the shipped CPU findings.
- Small inner-product and cosine batches no longer require native SGEMM when repeated SGEMV is the selected strategy.
- Large native lexical batches and vector SGEMM operations no longer bypass runtime memory-budget checks.
- Automatic thread selection no longer leaves OpenMP or OpenBLAS runtime state modified after a search completes or raises an exception.
- Missing, malformed, stale, or incompatible machine profiles now fall back safely to the packaged policy.
- Calibration never modifies or overwrites the fallback policy installed inside the wheel.

### Known limitations
- Result identifiers are zero-based positional indexes.
- Exact top-k selection currently uses an `O(Nk)` implementation.
- Machine calibration currently tunes representative OpenMP and OpenBLAS thread counts.
- Workload crossover thresholds are supplied by the packaged benchmark policy and are not recalibrated per machine in 0.1.0.
- Full-SMT thread counts are not selected automatically by the shipped fallback policy.
- Hybrid-core-aware placement, CPU affinity management, cache topology, and NUMA-aware dispatch are not included.
- CUDA and GPU backends are not included.
- Binary wheels currently target Linux x86-64.

[Unreleased]: https://github.com/suwesh/fortran-nlp-kernels/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/suwesh/fortran-nlp-kernels/compare/v0.1.0b2...v0.1.0
[0.1.0b2]: https://github.com/suwesh/fortran-nlp-kernels/compare/v0.1.0b1...v0.1.0b2
[0.1.0b1]: https://github.com/suwesh/fortran-nlp-kernels/releases/tag/v0.1.0b1
