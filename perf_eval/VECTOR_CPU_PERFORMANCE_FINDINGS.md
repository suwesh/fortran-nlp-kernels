# Vector CPU Performance Findings

**Evaluation date:** 24 October 2026  
**Package baseline:** `fortran-nlp-kernels==0.1.0b2`  
**Purpose:** Preserve the evidence needed to design stable, workload-aware, machine-calibrated automatic backend and thread selection for the `0.1.0` CPU release.

## 1. Scope and timing boundary

This study evaluates the released exact vector-search routines for inner product (IP) and squared L2 distance across:

- handwritten serial Fortran;
- handwritten OpenMP Fortran;
- OpenBLAS SGEMV single-query search;
- repeated SGEMV batch search;
- native OpenBLAS SGEMM batch search.

The timed native calls include:

```text
score computation
+ exact serial O(Nk) top-k selection
+ native binding call
+ returned-array creation
```

The study does **not** isolate score-only execution, top-k-only execution, index construction, normalization, or public Python API overhead. These complete-search timings represent the behavior available from the published `0.1.0b2` wheel.

Cosine was not timed as a separate native metric because the real vectors are unit-normalized, making cosine scoring equivalent to inner product. Squared L2 remained a separate benchmark because the released implementation and arithmetic path differ from IP.

---

## 2. Test platform and runtime controls

- **CPU:** AMD Ryzen 5 5500
- **Physical cores:** 6
- **Logical processors:** 12
- **SMT:** 2 hardware threads per physical core
- **Environment:** WSL2 Linux
- **Python:** CPython 3.12
- **Package:** `fortran-nlp-kernels==0.1.0b2`
- **Vector dtype:** `float32`
- **Vector dimension:** 1,024

Physical-core OpenMP runs used:

```text
OMP_DYNAMIC=FALSE
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_WAIT_POLICY=active
OPENBLAS_NUM_THREADS=1
```

OpenMP SMT runs used `OMP_PLACES=threads`.

BLAS runs isolated thread ownership:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=<tested count>
```

Handwritten OpenMP and multithreaded OpenBLAS were never enabled simultaneously. This prevented nested OpenMP/BLAS oversubscription.

---

## 3. Production-derived vector database

The packed database combines embeddings from four Sahayak knowledge sources:

```text
FAQs:             67 vectors
Manuals:         141 vectors
Troubleshooting: 240 vectors
Video:            27 vectors
Total:            475 vectors
```

Validated properties:

```text
Shape:            (475, 1024)
dtype:            float32
Python layout:    C-contiguous rows
Native layout:    Fortran-contiguous (1024, vector_count)
Finite values:    all
Zero norms:       none
Minimum norm:     0.99999988
Median norm:      1.00000000
Mean norm:        1.00000000
Maximum norm:     1.00000024
Maximum |norm-1|: 2.38418579e-07
Base memory:      1.855 MiB
```

The native ABI requires:

```text
single query:  (dimension,), C-contiguous float32
query batch:   (dimension, query_count), Fortran-contiguous float32
database:      (dimension, vector_count), Fortran-contiguous float32
```

All layout conversion and replication occurred outside the timed region.

---

## 4. Correctness validation

Seven real database vectors were used as queries across source ranges, with:

```text
k = 1, 5, 10, 50
OMP_NUM_THREADS = 1 and 6
```

The following relationships passed:

```text
IP NumPy reference
= serial Fortran
= OpenMP Fortran
= OpenBLAS SGEMV

Squared-L2 NumPy reference
= serial Fortran
= OpenMP Fortran
= released BLAS L2 path

Native SGEMM batch
= repeated SGEMV queries
```

Additional checks passed:

- exact top-k ID equivalence across backends;
- returned score/distance consistency with returned IDs;
- descending IP and ascending L2 ordering;
- self-match correctness;
- batch output shape `(k, query_count)`;
- unit-normalized IP/L2 ranking equivalence;
- float32 numerical tolerances appropriate for differing reduction orders.

---

## 5. Experiment completeness

All planned configurations completed successfully.

```text
Single-query physical-core matrix:
  1,728 / 1,728 configurations
  339,120 raw measurements

Single-query SMT matrix:
  432 / 432 configurations
  38,880 raw measurements

Batch physical-core matrix:
  2,184 / 2,184 configurations
  243,360 raw measurements

Batch SMT matrix:
  486 / 486 configurations
  14,040 raw measurements

Total:
  4,830 / 4,830 configurations
  635,400 raw measurements
```

Integrity checks:

```text
Duplicate run rows:                0
Repetition-mismatch configurations: 0
Execution-variant checks:          PASS
Package version:                   0.1.0b2
```

---

## 6. Workload matrix

### Single-query physical-core matrix

```text
Metrics:              IP, squared L2
Queries:              0, 67, 208, 474
Replication factors:  1, 2, 5, 10, 25, 50, 100, 250
Database vectors:     475 to 118,750
k:                    1, 10, 50

Variants:
Fortran serial:       1 thread
Fortran OpenMP:       1, 2, 4, 6 threads
OpenBLAS SGEMV:       1, 2, 4, 6 threads
```

### Single-query SMT matrix

```text
Database vectors:     4,750; 47,500; 118,750
k:                    1, 10, 50
OpenMP threads:       8, 10, 12
OpenBLAS threads:     8, 10, 12
```

Six-thread physical-core baselines were reused from the physical matrix.

### Batch physical-core matrix

```text
Metrics:              IP, squared L2
Replication factors:  1, 10, 100, 250
Database vectors:     475; 4,750; 47,500; 118,750
Query batches:        1, 2, 4, 8, 16, 32, 64
k:                    1, 10, 50

Variants:
Repeated serial:      1 thread
Repeated OpenMP:      1, 2, 4, 6 threads
Repeated SGEMV:       1, 2, 4, 6 OpenBLAS threads
Native SGEMM:         1, 2, 4, 6 OpenBLAS threads
```

### Batch SMT matrix

```text
Database vectors:     4,750; 47,500; 118,750
Query batches:        8, 32, 64
k:                    1, 10, 50
Methods:              repeated OpenMP, repeated SGEMV, native SGEMM
Threads:              8, 10, 12
```

---

## 7. Single-query inner-product findings

### 7.1 Physical-core winners

At `k=10`, the median-across-query winner was always OpenBLAS SGEMV across the tested physical-core matrix.

```text
Vectors    Winner   Threads   Median latency   Median CV   Speedup vs serial
475        SGEMV       4       0.029877 ms      0.2360      11.6537x
950        SGEMV       4       0.055870 ms      0.1739      12.4328x
2,375      SGEMV       2       0.142837 ms      0.1715      12.3947x
4,750      SGEMV       1       0.528166 ms      0.1107       6.9032x
11,875     SGEMV       6       1.566481 ms      0.1012       5.7988x
23,750     SGEMV       1       3.132765 ms      0.0655       5.8161x
47,500     SGEMV       6       6.448138 ms      0.0671       5.6418x
118,750    SGEMV       1      16.119255 ms      0.0406       5.6313x
```

Raw median winners at tiny scales should not be converted directly into packaged thread defaults. Differences between SGEMV thread counts were frequently small while timing variability was high. At 4,750, 23,750, and 118,750 vectors, SGEMV-1 was the physical-matrix winner, demonstrating that additional OpenBLAS workers were often unnecessary.

### 7.2 Throughput transition

At small IP databases, SGEMV achieved approximately 32–35 estimated complete-search GFLOP/s and outperformed OpenMP-6 substantially. As database size increased, SGEMV settled near approximately 14.5–15.5 estimated GFLOP/s, converging with OpenMP-6.

This is measured evidence of a workload transition, but not proof of a specific cache or bandwidth cause. The cause requires hardware counters or score-only profiling.

### 7.3 SMT-qualified winners

Using a provisional `CV <= 10%` stability filter at `k=10`:

```text
4,750 vectors:
  SGEMV-6
  0.528323 ms
  CV 0.0844

47,500 vectors:
  OpenMP-8
  5.405163 ms
  CV 0.0448
  1.2441x faster than OpenMP-6

118,750 vectors:
  OpenMP-8
  13.384949 ms
  CV 0.0199
  1.2877x faster than OpenMP-6
```

Therefore:

```text
Small/medium single-query IP:
  SGEMV is the primary candidate.

Larger single-query IP:
  partial-SMT OpenMP can overtake SGEMV.

OpenMP-8:
  strongest conservative partial-SMT candidate.

OpenMP-10:
  can improve medians, but requires stability filtering.

OpenMP-12:
  frequently regresses and is not a safe default.

SGEMV SMT:
  no consistent gain.
```

### 7.4 Resource-aware tie breaking

When SGEMV-1 and a multithreaded alternative are within a small latency tolerance, SGEMV-1 should be preferred because it leaves CPU capacity available for concurrent application work.

---

## 8. Single-query squared-L2 findings

### 8.1 Physical-core winners

Fortran OpenMP-6 won at every tested database size at `k=10`.

```text
Vectors    Winner    Threads   Median latency   Median CV   Speedup vs serial
475        OpenMP       6       0.065897 ms      0.1595       5.2536x
950        OpenMP       6       0.130714 ms      0.1101       5.3190x
2,375      OpenMP       6       0.324024 ms      0.0675       5.4233x
4,750      OpenMP       6       0.697257 ms      0.1024       5.2331x
11,875     OpenMP       6       1.695183 ms      0.0395       5.3862x
23,750     OpenMP       6       3.423727 ms      0.0347       5.3634x
47,500     OpenMP       6       6.797834 ms      0.0299       5.3544x
118,750    OpenMP       6      16.842257 ms      0.0162       5.4268x
```

The released L2 BLAS path stayed around or below handwritten serial performance. This behavior persisted across vector count, query batch, physical-core threading, and SMT.

### 8.2 SMT-qualified winners

Using the provisional `CV <= 10%` filter:

```text
4,750 vectors:
  OpenMP-8
  0.529007 ms
  CV 0.0705
  1.3258x over OpenMP-6

47,500 vectors:
  OpenMP-10
  5.211034 ms
  CV 0.0705
  1.3140x over OpenMP-6

118,750 vectors:
  OpenMP-10
  13.268289 ms
  CV 0.0456
  1.2638x over OpenMP-6
```

Therefore:

```text
Uncalibrated L2 default:
  handwritten OpenMP using physical cores.

Calibrated L2:
  test OpenMP-8 and OpenMP-10.

OpenMP-12:
  not a general default.

Current L2 SGEMV:
  never an automatic default.
```

### 8.3 L2 BLAS requires source-level investigation

Additional OpenBLAS threads and SGEMM batching did not close the several-fold gap to handwritten OpenMP. Candidate causes requiring code inspection include:

- database-norm handling;
- query-norm handling;
- temporary score arrays;
- distance transformation;
- extra full-database passes;
- interaction between postprocessing and top-k.

Thread calibration cannot repair an implementation path whose underlying work is inefficient.

---

## 9. Batch IP findings

### 9.1 Exact SGEMM crossover against repeated SGEMV

At six OpenBLAS threads and `k=10`, native SGEMM gain was:

```text
Database   Batch 1   Batch 2   Batch 4   Batch 8   Batch 16  Batch 32  Batch 64
475          0.191     0.362     0.799     1.093      1.737      2.034      2.338
4,750        0.234     0.458     1.042     1.648      2.336      4.515      4.092
47,500       0.240     0.470     0.910     1.734      2.694      3.640      4.602
118,750      0.273     0.483     0.997     1.671      2.627      3.619      4.474
```

Interpretation:

```text
Batch 1-2:
  repeated SGEMV clearly wins.

Batch 4:
  crossover region.

Batch 8:
  native SGEMM consistently wins.

Batch 16-64:
  native SGEMM is strongly preferred.
```

This result is stable across database size and physical OpenBLAS thread count. A conservative packaged IP batch threshold can therefore use:

```text
query_batch_size >= 8 → native SGEMM
```

Batch 4 should remain calibration-dependent.

### 9.2 Exact physical-matrix method transitions

At `k=10`:

```text
475 vectors:
  batches 1-4  → repeated SGEMV
  batches 8-64 → native SGEMM

4,750 vectors:
  batches 1-2  → repeated SGEMV
  batches 4-64 → native SGEMM

47,500 vectors:
  batches 1-4  → repeated SGEMV
  batches 8-64 → native SGEMM

118,750 vectors:
  batches 1-2  → repeated SGEMV
  batches 4-64 → native SGEMM
```

Individual best OpenBLAS thread counts varied and should not become hard-coded global constants. Several large SGEMM winners used only one or two OpenBLAS threads.

### 9.3 SGEMM and SMT

Native IP SGEMM latency was nearly flat from 6 through 12 OpenBLAS threads. Six physical workers captured essentially all useful SGEMM performance. The large SGEMM advantage arises from batching and matrix reuse, not SMT.

Recommended behavior:

```text
Uncalibrated native SGEMM:
  physical-core count or a conservative calibrated BLAS count.

Do not enable full SMT automatically.
```

---

## 10. Batch squared-L2 findings

### 10.1 Overall winner

Repeated Fortran OpenMP-6 won every physical-core L2 batch workload at `k=10`:

```text
all four database sizes
× all seven query batches
= 28 / 28 workloads
```

Representative results:

```text
4,750 vectors, batch 64:
  repeated OpenMP-6 = 43.741481 ms

47,500 vectors, batch 64:
  repeated OpenMP-6 = 429.997861 ms

118,750 vectors, batch 64:
  repeated OpenMP-6 = 1,076.864787 ms
```

The L2 batch policy is therefore metric-specific and fundamentally different from IP.

### 10.2 SGEMM crossover within the BLAS family

Native L2 SGEMM versus repeated L2 SGEMV at six OpenBLAS threads:

```text
Batch 1:   0.67-0.75x
Batch 2:   0.85-0.89x
Batch 4:   approximately 1.00x
Batch 8:   1.03-1.07x
Batch 16:  1.05-1.11x
Batch 32:  1.06-1.13x
Batch 64:  1.07-1.16x
```

This is useful only when choosing between BLAS implementations. Both L2 BLAS methods remain substantially slower than handwritten OpenMP.

### 10.3 L2 SMT

Repeated L2 OpenMP often improved at 8 threads and sometimes at 10 threads. Twelve threads frequently regressed or became unstable. L2 SGEMV and SGEMM gained essentially nothing from SMT.

Recommended behavior:

```text
Uncalibrated L2 batch:
  repeated OpenMP on physical cores.

Calibrated L2 batch:
  test OpenMP-8 and OpenMP-10 with stability filtering.

Current L2 BLAS:
  explicit override only.
```

---

## 11. Exact top-k observations

Increasing `k` from 1 to 50 produced modest but measurable latency growth.

For faster IP and L2 methods, the relative increase was often approximately 20-30%. For slower serial methods, the same absolute top-k work represented a smaller fraction of complete-search latency.

The correct conclusion is:

> The shared serial `O(Nk)` top-k stage adds measurable latency, especially after scoring is accelerated, but it does not dominate most tested complete-search workloads or change the principal backend ordering over `k=1-50`.

Top-k optimization is worthwhile future work but does not need to block backend selection or stable `0.1.0`.

A future focused decomposition should measure:

```text
score-only execution
top-k-only execution
complete native search
public Python API search
```

---

## 12. Memory behavior

For single-query search:

\[
M \approx 4ND + 4D + 12k
\]

For batch search with only top-k returned:

\[
M \approx 4ND + 4DB + 12Bk
\]

The database dominates memory.

```text
475 vectors:       ~1.86 MiB
950 vectors:       ~3.71 MiB
2,375 vectors:     ~9.28 MiB
4,750 vectors:    ~18.55 MiB
11,875 vectors:   ~46.39 MiB
23,750 vectors:   ~92.77 MiB
47,500 vectors:  ~185.55 MiB
118,750 vectors: ~463.87 MiB
```

At batch 64, the query matrix is only approximately 0.25 MiB. At `k=50`, batch top-k output is only approximately 37.5 KiB.

Therefore:

- query batch size does not materially increase the current API's output memory;
- SGEMM selection can be performance-driven rather than output-memory-driven;
- memory guardrails remain necessary for index construction, matrix copies, GPU transfer, and future full-score APIs;
- process RSS may exceed the analytical native-input model if both Python row-major and native column-major copies remain resident.

---

## 13. Variability and stability-aware calibration

Median latency alone is insufficient for automatic selection.

Observed examples included:

- OpenMP-10 or OpenMP-12 with excellent medians but high coefficient of variation;
- single-query IP OpenMP-12 regressions;
- batch OpenMP-12 configurations exceeding 100% CV;
- SGEMV/SGEMM paths that were slower but often more stable;
- OpenMP-8 frequently providing the strongest speed/stability compromise.

A machine calibrator should evaluate:

```text
median latency
P95 latency
coefficient of variation
gain over a safe baseline
thread count
```

Suggested decision sequence:

1. Validate numerical correctness.
2. Reject configurations with unacceptable CV or P95 behavior.
3. Require a meaningful median gain over a safer baseline.
4. Treat near-equal candidates as equivalent.
5. Prefer fewer threads among equivalent candidates.
6. Persist the chosen backend, threads, thresholds, and evidence.

Initial candidate acceptance thresholds might be:

```text
median gain >= 5-10%
CV <= 10%
positive P95 gain
```

These thresholds should remain configurable until validated across additional machines.

---

## 14. Plotting corrections for public documentation

Several internal figures label a value as parallel efficiency while using repeated Fortran serial as the numerator for a different backend. This combines implementation/algorithm advantage with thread scaling.

Public `0.1.0` documentation should separate three concepts.

### Backend advantage versus serial

\[
A_b(p)=\frac{T_{\text{Fortran serial}}}{T_{b,p}}
\]

Recommended label:

```text
Complete-search speedup versus Fortran serial
```

### Within-backend thread scaling

\[
S_b(p)=\frac{T_{b,1}}{T_{b,p}}
\]

This should be plotted separately for:

```text
Fortran OpenMP
OpenBLAS SGEMV
OpenBLAS SGEMM
```

### True within-backend parallel efficiency

\[
E_b(p)=\frac{S_b(p)}{p}
\]

Only this value should be called `parallel efficiency` without qualification.

The SGEMM gain-versus-repeated-SGEMV figure should remain prominent because its `1.0` line communicates the batch crossover directly.

---

## 15. Provisional automatic-selection architecture

Backend selection must be metric-aware, workload-aware, topology-aware, and stability-aware.

### 15.1 Single-query IP and normalized cosine

Inputs:

```text
vector_count
dimension
k
machine profile
```

Provisional behavior:

```text
Small/medium database:
  OpenBLAS SGEMV, often with one BLAS thread.

Larger database:
  compare SGEMV-1 with calibrated OpenMP-8/10.

If candidates are nearly tied:
  prefer fewer threads.

OpenMP-12:
  never an uncalibrated default.
```

### 15.2 Batch IP and normalized cosine

Inputs:

```text
vector_count
dimension
query_batch_size
k
machine profile
```

Provisional behavior:

```text
batch <= 2:
  repeat the calibrated single-query backend

batch == 4:
  crossover region; use calibration

batch >= 8:
  native SGEMM

SGEMM threads:
  physical-core baseline; no automatic SMT
```

### 15.3 Single-query squared L2

```text
475-118,750 tested vectors:
  handwritten OpenMP

uncalibrated threads:
  physical-core count

calibrated candidates:
  OpenMP-8 and OpenMP-10

current L2 BLAS:
  not automatic
```

### 15.4 Batch squared L2

```text
all tested workloads:
  repeated handwritten OpenMP

uncalibrated threads:
  physical-core count

calibrated candidates:
  OpenMP-8 and OpenMP-10 with stability filtering

current L2 SGEMV/SGEMM:
  explicit override only
```

---

## 16. Machine-calibration design

Calibration should be explicit, not run automatically during installation or import.

Suggested interface:

```bash
fortran-nlp-calibrate
```

or:

```bash
python -m fortran_nlp.calibrate
```

Resolution order:

```text
explicit user backend/thread choice
→ compatible saved machine profile
→ conservative packaged defaults
```

Suggested profile location:

```text
~/.config/fortran-nlp-kernels/cpu_profile.json
```

Suggested profile metadata:

```text
profile schema version
package version
CPU model
physical cores
logical processors
operating environment
OpenMP runtime
OpenBLAS version
vector dimension(s) calibrated
calibration timestamp
calibration workload specification
```

Suggested policy records should store numeric regions, for example:

```json
{
  "metric": "ip",
  "query_mode": "batch",
  "min_query_batch": 8,
  "backend": "blas_batch",
  "threads": 6,
  "median_latency_ns": 0,
  "p95_latency_ns": 0,
  "coefficient_of_variation": 0.0
}
```

The final implementation must avoid unreliable runtime mutation of `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS` after native runtimes have initialized. Thread control should use supported runtime APIs or a clearly documented process-global policy.

---

## 17. Measured facts, interpretations, and unresolved questions

### Directly measured

1. All 4,830 configurations and 635,400 raw measurements completed without duplicates or repetition mismatches.
2. IP SGEMV won the physical single-query matrix across all tested vector counts.
3. SGEMV-1 frequently won or tied larger single-query IP workloads.
4. IP SGEMV's advantage over serial fell from roughly 12x at small sizes to roughly 5.6x at the largest sizes.
5. L2 OpenMP-6 won every tested physical single-query workload.
6. L2 OpenMP-6 achieved approximately 5.2-5.4x speedup over serial.
7. IP SGEMM crossed repeated SGEMV around batch 4-8.
8. IP SGEMM was consistently preferred at batch 8 and above.
9. At large IP batches, SGEMM reached roughly 2-5x advantage over repeated SGEMV.
10. Repeated OpenMP-6 won all 28 tested physical L2 batch workloads at `k=10`.
11. L2 BLAS paths remained around or below handwritten serial performance.
12. SGEMV and SGEMM received little or no benefit from SMT.
13. OpenMP frequently benefited from partial SMT at 8-10 threads.
14. OpenMP-12 frequently regressed or became unstable.
15. Increasing `k` from 1 to 50 produced measurable but non-dominant latency growth.
16. Database storage dominated modeled vector memory.

### Supported interpretations

1. IP SGEMM benefits from reuse across query columns.
2. Repeated SGEMV revisits the database independently per query.
3. SGEMV's single-query benefit is implementation-driven rather than thread-driven.
4. Partial SMT can help handwritten OpenMP hide selected stalls.
5. Full SMT introduces contention or runtime overhead.
6. L2 BLAS performs additional or inefficient metric-specific work absent from direct handwritten L2.
7. Resource-aware tie breaking can improve coexistence by selecting fewer threads for near-equal latency.
8. Automatic selection must use metric and query-batch size as first-class features.

### Requires source inspection or hardware profiling

1. Cause of the IP SGEMV throughput transition.
2. Whether large-batch IP SGEMM is compute-limited, bandwidth-limited, or mixed.
3. Why SGEMM gains essentially nothing from SMT.
4. Why OpenMP-12 is unstable in selected workloads.
5. Exact L2 BLAS bottleneck.
6. Compiler vectorization of handwritten IP/L2 loops.
7. Score-only versus top-k timing decomposition.
8. Total process RSS when user and native layouts coexist.
9. Behavior on native Linux, other CPU vendors, NUMA systems, and servers.

---

## 18. Work required before stable `0.1.0`

1. Regenerate public plots with correct backend-advantage and within-backend scaling terminology.
2. Write the backend-policy specification before coding the policy.
3. Design and version the calibration-profile schema.
4. Implement explicit calibration with compact representative workloads.
5. Implement backend and thread resolution.
6. Add process-global OpenMP/OpenBLAS thread-control policy and tests.
7. Add stability-aware candidate selection.
8. Add explicit-user-override precedence.
9. Add corrupted/incompatible-profile fallback.
10. Add index-construction and matrix-copy memory guardrails.
11. Inspect the L2 BLAS implementation and decide whether optimization is required before stable release.
12. Add focused score/top-k decomposition when practical.
13. Run a compact regression matrix after policy or native changes.
14. Validate clean manylinux wheels on supported Python versions.
15. Publish selected tables, figures, methodology, and limitations.

---

## 19. Final conclusion

The released vector CPU backends do not have one universal winner.

```text
Single-query IP:
  OpenBLAS SGEMV dominates smaller workloads;
  partial-SMT OpenMP becomes competitive or faster at larger workloads.

Batch IP:
  repeated SGEMV wins tiny batches;
  native SGEMM strongly wins at batch 8 and above.

Single and batch L2:
  handwritten OpenMP is decisively superior to the current BLAS paths.

Threading:
  SGEMV/SGEMM gain little from SMT;
  OpenMP can benefit from 8-10 threads;
  12-thread full SMT is frequently harmful or unstable.
```

The stable `0.1.0` automatic selector should therefore be:

- metric-aware;
- query-mode and batch-size-aware;
- vector-count and dimension-aware;
- backend-specific in thread selection;
- stability-aware rather than median-only;
- resource-aware when candidates are nearly tied;
- calibrated explicitly per machine when possible;
- conservative when no compatible profile exists.

This evidence provides a strong foundation for a serious open-source feature: machine-calibrated exact-search backend selection across serial Fortran, OpenMP Fortran, OpenBLAS SGEMV, and OpenBLAS SGEMM.
