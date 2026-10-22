Yes bro, we can go straight into the intensive CPU experiment and use replication factor as one primary scaling axis, similar to the previous FORTRESS benchmark.

But we should make the experiment paper-grade instead of copying the old script unchanged.

What the first experiment measures

Start with kernel-facing single-candidate corpus scoring:

Python → nanobind → C ABI → Fortran kernel → output arrays


Using:

_cpu.jaccard_corpus_serial(...)
_cpu.jaccard_corpus_openmp(...)


This includes binding and output allocation, but excludes:

text normalization
shingle generation
corpus packing
Python ranking
JaccardResult construction

We will measure the public JaccardIndex.search() separately afterward.

Replication-factor experiment

Use a deterministic base corpus, then repeat the packed document collection:

replication_factors = [
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]


Record both:

replication factor
actual document count
actual total corpus shingles


Replication is useful because it preserves document-length distribution and exact workload composition while scaling corpus size.

However, replicated documents may produce unusually favorable cache and branch behavior. Therefore the paper should eventually include both:

replicated-corpus scaling
independently generated corpus scaling


Replication answers scalability on a controlled repeated workload. The generator answers scalability on unique data.

Thread counts

Use separate processes for:

thread_counts = [
    1,
    2,
    4,
    6,
    8,
    12,
]


Adjust the final maximum after checking:

lscpu


Every thread configuration must be launched as a fresh process:

OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 \
python perf_eval/lexical/benchmark_single.py ...


Do not alter OMP_NUM_THREADS inside one Python process.

Workload dimensions

For the intensive single-query experiment:

Candidate shingles:
16, 64, 256, 1024

Base document shingles:
16, 64, 256, 1024

Target overlap:
0.00, 0.25, 0.50, 0.75, 1.00

Length modes:
fixed, variable

Replication factors:
1, 2, 5, 10, 25, 50, 100, 250, 500, 1000

OpenMP threads:
1, 2, 4, 6, 8, 12


This full Cartesian product is extremely large, but that is okay if the runner supports filtering and resumable CSV output. We should not hold all results until the end.

Do not use 500 repetitions for every workload

Your earlier script used:

WARMUP_RUNS = 10
MEASURED_RUNS = 500


That is reasonable for tiny kernels, but wasteful for a corpus with hundreds of thousands or millions of documents.

Use adaptive repetitions:

Corpus documents ≤ 1,000       → 500 repetitions
≤ 10,000                       → 200 repetitions
≤ 100,000                      → 50 repetitions
> 100,000                      → 20 repetitions


Keep:

WARMUP_RUNS = 10


for each backend and configuration.

Every individual repetition must be written to raw CSV. Summaries can be calculated later.

Improve the old measurement structure

Your old script always measured:

all serial runs
then
all OpenMP runs


That can introduce thermal or frequency drift.

Instead, alternate their order:

for run_index in range(measured_runs):
    if run_index % 2 == 0:
        serial first
        OpenMP second
    else:
        OpenMP first
        serial second


Or execute each backend in an independent fresh process. Independent processes are cleaner for the final paper.

Recommended final structure
one process
→ one backend
→ one thread configuration
→ one workload point
→ warm up
→ measure
→ append raw CSV rows
→ terminate process


This avoids:

retained OpenMP runtime state;
cross-backend cache interaction;
thermal-order bias;
accidental thread-setting reuse.
Update lexical_cpu_matrix.json

Use this:

{
  "schema_version": 1,
  "package_version": "0.1.0b2",
  "random_seed": 20261022,

  "base_document_count": 100,

  "replication_factors": [
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000
  ],

  "shingles_per_document": [
    16,
    64,
    256,
    1024
  ],

  "candidate_shingle_counts": [
    16,
    64,
    256,
    1024
  ],

  "target_overlaps": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],

  "thread_counts": [
    1,
    2,
    4,
    6,
    8,
    12
  ],

  "candidate_batch_sizes": [
    1,
    4,
    16,
    64,
    256
  ],

  "length_modes": [
    "fixed",
    "variable"
  ],

  "variable_length_minimum_ratio": 0.25,
  "warmup_iterations": 10,

  "measurement_iterations": {
    "up_to_1000_documents": 500,
    "up_to_10000_documents": 200,
    "up_to_100000_documents": 50,
    "above_100000_documents": 20
  }
}

Results schema

Each raw row should contain:

experiment_id
timestamp
backend
thread_count
seed
length_mode
target_overlap
candidate_shingles
base_document_count
replication_factor
document_count
total_corpus_shingles
mean_document_shingles
minimum_document_shingles
maximum_document_shingles
warmup_runs
measured_run_index
elapsed_ns
documents_per_second
shingles_per_second
package_version
python_version
numpy_version
host_name
git_commit


Do not print enormous arrays such as:

print(document_lengths)
print(document_offsets)
print(scores)


for performance runs. That can create huge terminal output and accidental overhead around the experiment.

Print summaries only:

document count
total shingles
min/mean/max document length
candidate shingles
target and actual overlap

Experimental stages

Run the study in this order.

Experiment L1: Serial corpus scaling
backend = serial
threads = 1
replication factor scaling


Answers:

How does serial Fortran latency and throughput scale?

Experiment L2: OpenMP thread scaling
backend = OpenMP
threads = 1, 2, 4, 6, 8, 12
replication factor scaling


Answers:

At what corpus size does OpenMP become worthwhile?

Experiment L3: Overlap sensitivity
overlap = 0.00, 0.25, 0.50, 0.75, 1.00


Answers:

Does branch behavior change with set overlap?

Experiment L4: Document-length sensitivity
16, 64, 256, 1024 shingles per document


Answers:

How does pairwise intersection cost scale?

Experiment L5: Fixed versus variable lengths

Answers:

Does OpenMP load imbalance appear with heterogeneous documents?

Experiment L6: Batched lexical search
candidate batches = 1, 4, 16, 64, 256


Answers:

Does parallelizing candidate-document pairs improve throughput?

Experiment L7: Language comparison
Python set
C++ serial
C++ OpenMP
Fortran serial
Fortran OpenMP

Metrics

Calculate:

speedup(p)=T1Tp\text{speedup}(p)=\frac{T_1}{T_p} parallel efficiency(p)=speedup(p)p\text{parallel efficiency}(p)= \frac{\text{speedup}(p)}{p} documents/s=NdocumentsT\text{documents/s}= \frac{N_{\text{documents}}}{T} shingles/s=Ncorpus shinglesT\text{shingles/s}= \frac{N_{\text{corpus shingles}}}{T}

For each configuration report:

minimum
median
mean
standard deviation
p25
p75
p95
coefficient of variation
speedup
parallel efficiency
documents/s
shingles/s

Next concrete file

Now write:

perf_eval/lexical/validate_equivalence.py


Before burning CPU across the full matrix, it should verify:

Python set reference
==
Fortran serial
==
Fortran OpenMP
==
batched OpenMP columns


across:

all five overlap levels
fixed and variable lengths
small and large shingle counts
multiple random seeds


Once that passes, we write benchmark_single.py to run the full replication-factor matrix and stream every measurement into perf_eval/results/raw/lexical_single_raw.csv.


===============================


Fresh mind is exactly when methodology should be clarified. Do not start by polishing the Introduction, because that risks deciding the story before the evidence exists.

The best move is:

Research questions
→ experimental methodology
→ benchmark matrix
→ pilot validation
→ formal measurements
→ results interpretation
→ Introduction and Abstract


You can draft a short paper skeleton now, but today’s main task should be understanding-first CPU evaluation.

Recommended approach today
1. Freeze the paper questions
Lexical search
How does exact sorted-shingle Jaccard scoring scale with corpus size?
At what workload does OpenMP outperform serial Fortran?
How does performance scale across six physical cores?
What happens when SMT increases execution from 6 to 8 or 12 threads?
How do shingle count, overlap, and document-length variability affect performance?
How does batched candidate scoring compare with repeated single-candidate searches?
How do Fortran implementations compare with equivalent Python and C++ implementations?
Vector search
How do serial Fortran, OpenMP Fortran, and OpenBLAS compare?
How do those implementations compare with NumPy and FAISS exact-flat indexes?
When does SGEMV outperform handwritten single-query scoring?
How much does SGEMM improve batched-query throughput?
How much time belongs to scoring, top-k selection, bindings, and public API orchestration?
At what database sizes, dimensions, and batch sizes do crossover points occur?

These questions should control what you measure. No random benchmark explosion.

Paper skeleton to create now

Write headings and bullet-point intent only:

1. Introduction
2. Background and Motivation
3. System Design
   3.1 Python-facing API
   3.2 Fortran computational kernels
   3.3 C ABI and nanobind integration
   3.4 OpenMP and OpenBLAS execution
4. Experimental Methodology
   4.1 Research questions
   4.2 Hardware and software environment
   4.3 Lexical workloads
   4.4 Vector-search workloads
   4.5 Baselines
   4.6 Timing boundaries
   4.7 Thread and affinity controls
   4.8 Statistical treatment
5. CPU Evaluation
   5.1 Lexical serial scaling
   5.2 Lexical OpenMP scaling and crossover
   5.3 Batched lexical search
   5.4 Python and C++ lexical comparisons
   5.5 Vector single-query evaluation
   5.6 Vector batched-query evaluation
   5.7 Top-k and interface overhead
6. NVIDIA GPU Evaluation
7. Application-Level Discussion
8. Limitations and Threats to Validity
9. Related Work
10. Conclusion


Do not write the final Introduction yet. Under each section, leave notes about the evidence that section will require.

CPU benchmark table templates

You will likely need several tables because one giant table will become unreadable.

Table 1: Experimental environment
| Category | Configuration |
|---|---|
| CPU | AMD Ryzen 5 5500 |
| Physical cores | 6 |
| Logical processors | 12 |
| Threads per core | 2 |
| Memory | TBD |
| Operating system | TBD |
| Linux kernel | TBD |
| Python | TBD |
| NumPy | TBD |
| FAISS | TBD |
| fortran-nlp-kernels | 0.1.0b2 |
| GFortran | TBD |
| G++ | TBD |
| OpenMP runtime | GNU libgomp |
| OpenBLAS | TBD |
| Compiler optimization | `-O3` |
| OpenMP binding | `OMP_PROC_BIND=close` |
| OpenMP placement | `OMP_PLACES=cores` |
| OpenMP dynamic teams | `OMP_DYNAMIC=FALSE` |
| OpenBLAS threads | 1 during handwritten-kernel tests |

Table 2: Lexical serial/OpenMP scaling
| Documents | Corpus shingles | Candidate shingles | Overlap | Length mode | Backend | Threads | Median latency (ms) | P95 latency (ms) | CV (%) | Documents/s | Speedup | Efficiency (%) |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 64,000 | 64 | 0.50 | Fixed | Fortran serial | 1 | TBD | TBD | TBD | TBD | 1.00 | 100.0 |
| 1,000 | 64,000 | 64 | 0.50 | Fixed | Fortran OpenMP | 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 1,000 | 64,000 | 64 | 0.50 | Fixed | Fortran OpenMP | 4 | TBD | TBD | TBD | TBD | TBD | TBD |
| 1,000 | 64,000 | 64 | 0.50 | Fixed | Fortran OpenMP | 6 | TBD | TBD | TBD | TBD | TBD | TBD |


Use the dedicated serial backend as the speedup baseline:

Sp=TserialTOpenMP,pS_p = \frac{T_{\text{serial}}}{T_{\text{OpenMP},p}} Ep=Spp×100E_p = \frac{S_p}{p}\times 100
Table 3: Serial/OpenMP crossover
| Documents | Corpus shingles | Serial median (ms) | OpenMP-2 (ms) | OpenMP-4 (ms) | OpenMP-6 (ms) | Best backend | Best thread count | Best speedup |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1,000 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 10,000 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 100,000 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1,000,000 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |


This table directly supports a benchmark-informed backend="auto" policy later.

Table 4: SMT sensitivity

Keep this separate from physical-core scaling.

| Documents | Threads | Physical/SMT interpretation | Median latency (ms) | P25 (ms) | P75 (ms) | P95 (ms) | CV (%) | Speedup vs serial |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 | 6 | One worker per physical core | TBD | TBD | TBD | TBD | TBD | TBD |
| 10,000 | 8 | Partial SMT | TBD | TBD | TBD | TBD | TBD | TBD |
| 10,000 | 12 | Full SMT | TBD | TBD | TBD | TBD | TBD | TBD |


This prevents the 12-thread instability from contaminating the main scaling interpretation.

Table 5: Lexical language comparison
| Documents | Document shingles | Candidate shingles | Implementation | Parallelism | Median latency (ms) | P95 (ms) | Documents/s | Speedup vs Python | Speedup vs C++ serial |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | Python set reference | Serial | TBD | TBD | TBD | 1.00 | N/A |
| TBD | TBD | TBD | C++ sorted intersection | Serial | TBD | TBD | TBD | TBD | 1.00 |
| TBD | TBD | TBD | C++ sorted intersection | OpenMP-6 | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | Fortran sorted intersection | Serial | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | Fortran sorted intersection | OpenMP-6 | TBD | TBD | TBD | TBD | TBD |


The C++ implementations belong only in perf_eval/, not the package API.

Table 6: Batched lexical search
| Candidate batch | Documents | Total comparisons | Method | Threads | Median batch latency (ms) | Candidates/s | Comparisons/s | Speedup vs repeated serial | Speedup vs repeated OpenMP |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | TBD | TBD | Batched OpenMP | 6 | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | Batched OpenMP | 6 | TBD | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | Batched OpenMP | 6 | TBD | TBD | TBD | TBD | TBD |
| 64 | TBD | TBD | Batched OpenMP | 6 | TBD | TBD | TBD | TBD | TBD |
| 256 | TBD | TBD | Batched OpenMP | 6 | TBD | TBD | TBD | TBD | TBD |

Table 7: Vector single-query search
| Metric | Database vectors | Dimension | k | Backend | Threads | Median latency (ms) | P95 (ms) | Queries/s | Recall/equivalence |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| IP | TBD | TBD | TBD | NumPy | 1 | TBD | TBD | TBD | Exact |
| IP | TBD | TBD | TBD | FAISS IndexFlatIP | TBD | TBD | TBD | TBD | Exact |
| IP | TBD | TBD | TBD | Fortran serial | 1 | TBD | TBD | TBD | Exact |
| IP | TBD | TBD | TBD | Fortran OpenMP | 6 | TBD | TBD | TBD | Exact |
| IP | TBD | TBD | TBD | OpenBLAS SGEMV | TBD | TBD | TBD | TBD | Exact |


Repeat for squared L2 and cosine, or split those metrics into separate tables if the results become crowded.

Table 8: Vector batched-query search
| Database vectors | Dimension | Query batch | k | Backend | BLAS threads | Median batch latency (ms) | Queries/s | Vectors scored/s | Speedup vs repeated SGEMV | Speedup vs FAISS |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| TBD | TBD | 1 | TBD | OpenBLAS SGEMM | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | 4 | TBD | OpenBLAS SGEMM | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | 16 | TBD | OpenBLAS SGEMM | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | 64 | TBD | OpenBLAS SGEMM | TBD | TBD | TBD | TBD | TBD | TBD |

Table 9: Timing decomposition

This may become the most important vector-search table.

| Workload | Backend | Score computation (ms) | Top-k (ms) | Binding/allocation (ms) | Public API total (ms) | Scoring share (%) | Top-k share (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| IP single query | Fortran serial | TBD | TBD | TBD | TBD | TBD | TBD |
| IP single query | Fortran OpenMP | TBD | TBD | TBD | TBD | TBD | TBD |
| IP single query | OpenBLAS SGEMV | TBD | TBD | TBD | TBD | TBD | TBD |
| IP batch | OpenBLAS SGEMM | TBD | TBD | TBD | TBD | TBD | TBD |


This determines whether optimizing top-k is actually necessary.

What to do today

Do not write polished prose yet. Create a methodology document:

perf_eval/METHODOLOGY.md


Put these sections in it:

# CPU Performance Evaluation Methodology

## Research questions

## Hardware and software environment

## Implementations and baselines

## Workload construction

## Timing boundaries

## Thread controls and affinity

## Warm-up and repetitions

## Correctness requirements

## Raw-data schema

## Statistical summaries

## Planned tables and figures

## Threats to validity


Then preserve the measurements already generated as pilot data:

mv \
  perf_eval/results/raw/lexical_single_raw.csv \
  perf_eval/results/raw/lexical_single_pilot.csv


Add a clear note in the methodology:

Pilot measurements are used only to validate the benchmark harness and
identify appropriate workload ranges. Pilot results are excluded from
the final reported dataset.

Recommendation

Today’s sequence should be:

Create METHODOLOGY.md.
Freeze the CPU table templates.
Record the exact environment.
Refine the benchmark runner to capture all environment variables.
Run one formal experiment at a time.
Interpret each experiment before launching the next.
Draft the Introduction only after the CPU and GPU story becomes clear.

That gives you a sound paper instead of a collection of benchmark numbers.


