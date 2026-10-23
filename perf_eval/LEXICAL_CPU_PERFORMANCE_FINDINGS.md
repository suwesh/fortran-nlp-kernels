# Lexical Fortran CPU Performance Findings

**Date:** 23 October 2026  
**Package baseline:** `fortran-nlp-kernels==0.1.0b2`  
**Purpose:** Preserve the evidence needed to design benchmark-informed defaults and optional machine-calibrated automatic backend selection for the stable `0.1.0` CPU release.

## 1. Scope

This document summarizes the completed CPU evaluation of the exact word-shingle Jaccard implementation in `fortran-nlp-kernels`.

The evaluated native paths were:

- dedicated serial Fortran corpus scoring;
- OpenMP single-candidate corpus scoring;
- repeated serial scoring for candidate batches;
- repeated OpenMP scoring for candidate batches;
- native batched OpenMP scoring using a collapsed candidate-document loop.

The study did not evaluate approximate retrieval. Every candidate was compared exactly with every corpus document.

The formal measurements used the published `0.1.0b2` manylinux wheel. Historical F2PY measurements from FORTRESS are not mixed into the formal dataset because the binding path, Fortran source structure, compiler environment, OpenMP runtime, and workload differed.

---

## 2. Test platform

- **CPU:** AMD Ryzen 5 5500
- **Physical cores:** 6
- **Logical processors:** 12
- **SMT threads per core:** 2
- **Execution environment:** WSL2 Linux
- **Package:** `fortran-nlp-kernels==0.1.0b2`
- **Python:** CPython 3.12
- **Primary OpenMP physical-core controls:**

```text
OMP_DYNAMIC=FALSE
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_WAIT_POLICY=active
OPENBLAS_NUM_THREADS=1
```

- **SMT controls:** identical except `OMP_PLACES=threads`.

`OMP_PLACES=cores` was used for 1, 2, 4, and 6 worker experiments so workers occupied physical-core places. `OMP_PLACES=threads` was used for 8, 10, and 12 worker SMT experiments so each OpenMP worker could occupy a distinct logical hardware thread.

---

## 3. Production-derived lexical corpus

Four knowledge sources were loaded from the Sahayak knowledge base:

- FAQs;
- reconstructed manuals;
- troubleshooting SOPs;
- video chunks.

Manual chunks were grouped by `doc_id`, ordered by `vector_id`, cleaned of embedding-only structural markers, and reconstructed into complete logical manuals.

### Corpus characteristics

```text
Metadata rows:               475
Logical lexical documents:  372
Corpus shingles:             64,577
Minimum shingles/document:  20
Median shingles/document:   135
Mean shingles/document:     173.59
Maximum shingles/document:  4,015
Zero-shingle documents:     0
```

Documents by source:

```text
FAQs:             67
Manuals:          38
Troubleshooting: 240
Video:            27
```

The large difference between the median and maximum document length created a genuinely variable-length workload rather than an artificially uniform microbenchmark.

### Candidate set

Eight production-derived documents were selected as candidates, with one short and one long candidate from each source:

```text
faqs-short:               20 shingles
faqs-long:               192 shingles
manuals-short:            53 shingles
manuals-long:          4,015 shingles
troubleshooting-short:    53 shingles
troubleshooting-long:    295 shingles
video-short:             117 shingles
video-long:              581 shingles
```

This candidate set intentionally spans a wide range of exact intersection costs.

---

## 4. Correctness requirements

Before formal timing, the following were validated on the real corpus:

- dedicated serial and OpenMP results were exactly equivalent;
- native batched OpenMP columns matched repeated serial calls;
- intersections and unions matched exactly;
- scores matched with zero relative tolerance and `1e-15` absolute tolerance;
- every copied candidate scored `1.0` against its source document;
- candidate-to-origin mappings were valid;
- validation passed with one OpenMP worker and six physical-core workers.

The package also had 98 passing Python tests before performance evaluation.

---

## 5. Single-candidate physical-core experiment

### Design

The corpus was evaluated below and above its natural size.

Fractional, source-stratified corpus scales:

```text
0.05, 0.10, 0.25, 0.50, 0.75 of the base corpus
```

Full-corpus replication scales:

```text
1, 2, 5, 10, 25, 50, 100, 250, 500, 1000
```

The combined range was approximately 19 to 372,000 documents.

Execution variants:

```text
Dedicated serial Fortran
OpenMP-1
OpenMP-2
OpenMP-4
OpenMP-6
```

The experiment contained:

```text
8 candidates
15 workload scales
5 execution variants
600 configurations
171,600 raw timing measurements
```

Completeness checks reported:

```text
600/600 configurations
0 duplicate run rows
0 repetition-count mismatches
Package version: 0.1.0b2
```

### Main findings

1. **Dedicated serial was better than OpenMP-1.**

   OpenMP-1 remained around `0.9x` relative to the dedicated serial routine. Parallel-region entry, scheduling, and barrier work remain present even with one worker.

2. **OpenMP became useful once workload granularity was sufficient.**

   Fractional corpus scales exposed the serial/OpenMP crossover. Larger corpora amortized fixed OpenMP costs.

3. **Physical-core scaling was useful but sublinear.**

   At sufficiently large workloads, approximate steady-state behavior was:

   ```text
   OpenMP-2: about 1.7-1.8x
   OpenMP-4: about 2.8-3.1x
   OpenMP-6: about 3.9-4.1x
   ```

4. **Six-core parallel efficiency stabilized around 65-68%.**

   This is reasonable for variable-length, branch-heavy sorted-set intersection with synchronization and shared-memory contention.

5. **Large-workload throughput reached roughly 0.7 billion corpus shingles per second with OpenMP-6.**

6. **Candidate size materially affected latency.**

   Candidate lengths ranged from 20 to 4,015 shingles. Longer candidates increased traversal work, and equal candidate lengths could still behave differently because matching positions, overlap, branch patterns, and exhaustion points differed.

7. **Latency scaled approximately linearly with corpus size.**

   The scoring path remained exhaustive, so work increased with the number of candidate-document comparisons.

8. **Memory growth was linear and dominated by packed corpus storage.**

   For single-candidate search:

   ```text
   input bytes  = candidate shingles + corpus shingles + offsets
   output bytes = 24 * document_count
   ```

   The three outputs are `int64` intersections, `int64` unions, and `float64` scores.

---

## 6. Single-candidate SMT sensitivity

### Design

OpenMP-6 from the physical-core experiment was reused as the baseline. New measurements used:

```text
Threads: 8, 10, 12
Corpus replication factors: 10, 100, 1000
Candidates: all 8
```

This produced 72 new SMT configurations.

### Main findings

1. **SMT provided additional performance beyond six physical-core workers.**

2. **OpenMP-8 was the most conservative stable SMT choice.**

   OpenMP-8 frequently delivered approximately `1.3x` over OpenMP-6 with relatively controlled timing variability.

3. **OpenMP-10 was often the strongest performance/stability compromise.**

   OpenMP-10 commonly delivered approximately `1.45-1.63x` over OpenMP-6 and frequently produced the best median latency.

4. **OpenMP-12 was inconsistent.**

   Full SMT sometimes produced the largest median speedup, but frequently reduced efficiency and sharply increased timing variability.

5. **Full logical-CPU count must not automatically become the default thread count.**

   Hardware-thread availability does not imply that all logical processors improve a branch-heavy native kernel. SMT siblings compete for execution resources, cache bandwidth, memory bandwidth, and branch-prediction capacity.

### Single-candidate thread-policy implication

```text
6 threads:  reliable physical-core baseline
8 threads:  conservative stable SMT option
10 threads: maximum-performance candidate on this machine
12 threads: explicit or calibrated choice only, not a safe default
```

---

## 7. Batched-candidate physical-core experiment

### Compared methods

#### Repeated serial

```text
for each candidate:
    serial candidate-to-corpus search
```

#### Repeated OpenMP

```text
for each candidate:
    OpenMP candidate-to-corpus search
```

Each candidate launches and completes an independent parallel corpus search.

#### Native batched OpenMP

```text
all candidates x all documents
```

The native batched routine uses a collapsed two-dimensional OpenMP candidate-document iteration space and returns full `(documents, candidates)` matrices.

### Design

Corpus sizes:

```text
372
3,720
37,200
372,000 documents
```

Candidate batch sizes:

```text
1, 2, 4, 8, 16, 32, 64
```

Physical-core execution variants:

```text
Repeated serial:       1 thread
Repeated OpenMP:       1, 2, 4, 6 threads
Native batch OpenMP:   1, 2, 4, 6 threads
```

The experiment contained 252 complete configurations.

### Main findings

1. **Native batching was not universally faster than repeated OpenMP.**

2. **Batch size 1 generally favored native batch execution.**

   A single native call avoided part of the repeated-call machinery while distributing document work effectively.

3. **Medium heterogeneous batches often favored repeated OpenMP.**

   Around batch sizes 2-8, repeated OpenMP frequently outperformed the collapsed native batch path, especially with 4-6 workers.

4. **Native batch recovered at larger batch sizes.**

   Around batch sizes 16-32, both methods converged. At batch sizes 32-64, native batch often became competitive or faster.

5. **The collapsed loop changed load balance and memory behavior.**

   Repeated OpenMP independently divides the corpus for each candidate. Native batching flattens candidates and documents into one iteration space. With variable document lengths and candidate sizes, static collapsed scheduling can distribute unequal computational costs.

6. **Native batching reduced Python/native transitions and repeated parallel-region launches.**

   These savings became more valuable as the candidate batch grew.

7. **Candidate composition affected batch scaling.**

   Batch construction used a deterministic production-derived candidate sequence:

   ```text
   batch 1:  first candidate
   batch 8:  all eight candidates
   batch 16: two complete cycles
   batch 64: eight complete cycles
   ```

   Batch size 8 introduced the complete heterogeneous set, including the 4,015-shingle manual candidate. Therefore, batch size and candidate complexity were partially coupled.

8. **Native batch output memory grew as `O(documents * candidates)`.**

   ```text
   output bytes = 24 * document_count * candidate_count
   ```

   At 372,000 documents and batch size 64, output matrices required approximately 571 MiB. Together with the packed corpus, the native working set exceeded approximately 1 GiB.

### Physical-core batch-policy implication

```text
Small batch:
    native batch can be competitive or faster

Medium heterogeneous batch:
    repeated OpenMP may be faster

Large batch:
    native collapsed-loop OpenMP often becomes preferable

Large corpus x large batch:
    output-memory budget must be checked before native batch selection
```

---

## 8. Batched-candidate SMT sensitivity

### Design

OpenMP-6 results from the physical-core batch study were reused. New measurements used:

```text
Corpus sizes: 3,720; 37,200; 372,000 documents
Batch sizes: 8; 32; 64
Methods: repeated OpenMP; native batch OpenMP
Threads: 8; 10; 12
```

This produced 54 new SMT configurations.

### Main findings

1. **OpenMP-8 generally provided a stable gain beyond OpenMP-6.**

2. **OpenMP-10 often produced the best absolute batch performance.**

   The strongest large native-batch configurations gained roughly `1.5x` over OpenMP-6.

3. **OpenMP-12 frequently reduced efficiency and stability.**

   Repeated OpenMP was particularly vulnerable because every candidate launched a separate parallel region.

4. **One repeated-OpenMP configuration showed a severe 12-thread collapse.**

   At approximately 3,720 documents and batch size 64, repeated OpenMP-12 regressed dramatically, while native batch remained comparatively stable. This demonstrated that full SMT can amplify repeated-region scheduling and synchronization costs.

5. **Native batch tolerated SMT better for large batches.**

   One collapsed parallel region often maintained better speed and stability than many repeated parallel regions.

6. **Ten threads emerged as the maximum-performance candidate, while eight threads remained the conservative stable choice.**

### Batched SMT policy implication

```text
6 threads:  reliable physical-core baseline
8 threads:  conservative stable SMT option
10 threads: frequent maximum-performance choice
12 threads: avoid as an automatic default
```

---

## 9. Important methodological caveats

1. **The corpus was production-derived but replicated for scaling.**

   Replication preserved real document lengths, shingle arrays, candidate relationships, and source composition, but also repeated branch and data patterns. Results should be described as controlled scaling over a production-derived corpus.

2. **Fractional corpora used deterministic source-stratified selection.**

3. **Batch size and candidate composition were partially coupled.**

4. **Measurements were conducted under WSL2.**

   Host scheduling, Windows background activity, virtualized timing behavior, dynamic frequency, and thermal conditions may add variability.

5. **Formal measurements used the published manylinux wheel.**

   Historical F2PY results are a separate implementation context.

6. **Estimated memory was analytically calculated, not measured peak RSS.**

7. **Raw repetitions were preserved.**

   Medians, P25, P75, P95, standard deviation, coefficient of variation, throughput, speedup, and efficiency were derived from raw data rather than timing only one call.

---

## 10. Lessons about OpenMP and hardware topology

1. A logical processor is not equivalent to an additional physical core.
2. `OMP_PLACES=cores` places physical-core workers correctly but permits execution within either SMT sibling belonging to that core-place.
3. `OMP_PLACES=threads` is necessary to expose distinct logical hardware threads in SMT studies.
4. `OMP_PROC_BIND=close` and `OMP_DYNAMIC=FALSE` are required for controlled comparisons.
5. More OpenMP threads can improve median latency while reducing parallel efficiency.
6. Full SMT can improve some workloads and catastrophically destabilize others.
7. Repeated short parallel regions are more sensitive to full-SMT overhead than one large collapsed parallel region.
8. Native batching trades fewer transitions and larger parallel work units for larger simultaneous output memory and potentially worse static load balance.

---

## 11. Implications for `backend="auto"`

Automatic selection should distinguish four separate decisions:

1. dedicated serial versus OpenMP for one candidate;
2. OpenMP thread count;
3. repeated single-candidate execution versus native batch execution;
4. whether the native batch output allocation is acceptable.

A single threshold on document count is insufficient.

### Candidate policy inputs

```text
document_count
total_corpus_shingles
candidate_count
candidate_shingle_count
candidate min/median/mean/max shingles
physical_core_count
logical_processor_count
available_memory
estimated batch output bytes
saved machine calibration profile
explicit environment or user override
```

### Estimated output allocation

```text
single candidate:
    24 * document_count bytes

native batch:
    24 * document_count * candidate_count bytes
```

### Provisional decision shape

```text
if user selected an explicit backend:
    honor the explicit backend

elif a compatible saved machine profile exists:
    use calibrated thresholds and calibrated thread counts

else:
    use conservative packaged benchmark-informed defaults
```

### Provisional single-candidate behavior

```text
very small corpus/work:
    dedicated serial

larger corpus/work:
    OpenMP

thread policy:
    physical cores as safe baseline
    partial SMT only when calibration shows a stable benefit
    never default blindly to all logical processors
```

### Provisional batch behavior

```text
small candidate count:
    compare repeated execution with native-batch overhead

medium heterogeneous batch:
    repeated OpenMP may be preferable

large candidate batch:
    native collapsed-loop OpenMP may be preferable

before native batch:
    reject or avoid the path if estimated output memory exceeds policy limit
```

These are provisional rules. Exact thresholds should be produced from a compact calibration routine rather than hard-coded from one machine.

---

## 12. Machine-calibration design for `0.1.0`

Calibration should be explicit and persistent. It should not run automatically during package installation or import.

Suggested interface:

```text
fortran-nlp-calibrate
```

Suggested behavior:

1. detect package version, CPU model, physical cores, logical processors, operating system, and relevant runtime versions;
2. run short representative single-candidate serial/OpenMP probes;
3. test physical-core counts and selected SMT counts;
4. run representative repeated and native batch probes;
5. estimate safe native-batch memory limits;
6. store crossover thresholds and preferred thread counts;
7. save the profile with a schema version and package compatibility metadata.

Suggested location:

```text
~/.config/fortran-nlp-kernels/cpu_profile.json
```

Suggested profile fields:

```text
profile_schema_version
package_version
cpu_model
physical_cores
logical_processors
operating_system
single_serial_openmp_crossover
preferred_single_openmp_threads
preferred_batch_openmp_threads
batch_method_thresholds
maximum_native_batch_output_bytes
calibration_timestamp
calibration_commands_or_parameters
```

Resolution order:

```text
explicit user backend/thread settings
→ compatible saved machine-calibration profile
→ conservative packaged benchmark-informed defaults
```

The profile should be invalidated or ignored when incompatible with the installed package version or detected hardware.

---

## 13. Recommendations for the stable CPU release

Before `0.1.0`:

- complete equivalent vector-search instrumentation;
- define OpenMP/OpenBLAS oversubscription policy;
- implement explicit calibration command/API;
- implement profile loading and validation;
- implement conservative uncalibrated defaults;
- add tests for policy resolution and overrides;
- document thread placement and environment variables;
- publish compact performance tables and selected figures;
- preserve raw measurements and processed artifacts;
- test the stable wheel in a clean Linux environment.

Features such as custom document IDs, custom vector IDs, SIMD research, and other product-polish work do not need to block this performance-focused stable release unless the intended API contract requires them.

---

## 14. Stored artifacts

Formal lexical artifacts include:

```text
perf_eval/results/raw/
    lexical_real_single_raw.csv
    lexical_real_single_smt_raw.csv
    lexical_real_batch_raw.csv
    lexical_real_batch_smt_raw.csv

perf_eval/results/processed/
    configuration summaries
    aggregated summaries
    completeness reports

perf_eval/results/tables/
    compact Markdown tables
    compact CSV tables
    detailed CSV tables

perf_eval/results/figures/
    single-candidate latency, throughput, speedup, efficiency,
    candidate sensitivity, and memory
    single-candidate SMT latency, speedup, efficiency, gain, variability
    batch latency, throughput, native gain, scaling, and memory
    batch SMT latency, speedup, efficiency, gain, and variability
```

Private corpus text, packed corpus arrays, and local absolute-path configuration remain outside version control.

---

## 15. Final lexical CPU conclusion

The exact lexical Fortran core demonstrates useful multicore scaling, but optimal execution depends on workload granularity, candidate complexity, batch size, memory requirements, physical-core count, and SMT behavior.

The strongest general lessons are:

- dedicated serial execution remains important for small work;
- OpenMP-1 is not a replacement for the dedicated serial routine;
- physical-core OpenMP provides stable acceleration;
- partial SMT can improve performance beyond physical-core execution;
- full SMT is not reliably safe;
- repeated OpenMP and native collapsed-loop batching each win in different regions;
- native batching becomes attractive at larger candidate batches but increases simultaneous output memory;
- automatic backend selection must be workload-aware, memory-aware, topology-aware, and preferably informed by a saved machine calibration profile.

These findings should guide the lexical portion of `backend="auto"` in the stable `0.1.0` CPU release. Exact policy thresholds should be finalized only after equivalent vector-search instrumentation so lexical and vector selection share one coherent calibration and configuration architecture.
