These graphs look coherent and already tell a strong performance story. The main caveat is that the very smallest corpus fractions are noisy, which is expected for microsecond-scale calls.

1. OpenMP speedup

The clearest result:

OpenMP-1 stays around 0.9×, confirming runtime overhead versus the dedicated serial routine.
OpenMP-2 stabilizes around 1.7–1.8×.
OpenMP-4 stabilizes around 2.8–3.1×.
OpenMP-6 reaches roughly 3.9–4.1× for sufficiently large workloads.
OpenMP becomes beneficial almost immediately, but reliable scaling appears after roughly hundreds to thousands of documents.

At six physical cores, approximately 4.0× speedup means:

E6=4.06≈66.7%E_6 = \frac{4.0}{6} \approx 66.7\%

That is a solid result for an irregular, branch-heavy, variable-length sorted-set intersection workload.

The shaded candidate-to-candidate variation becomes relatively narrow at larger corpus sizes. That suggests performance becomes more predictable after overhead is amortized.

2. Throughput

The sustained large-workload throughput appears approximately:

Serial: 1.6–1.7 × 10^8 corpus shingles/s
OpenMP-1: 1.5 × 10^8 shingles/s
OpenMP-2: 2.8–3.0 × 10^8 shingles/s
OpenMP-4: 4.8–5.3 × 10^8 shingles/s
OpenMP-6: 6.9–7.4 × 10^8 shingles/s

This agrees with the speedup graph.

The sharp spikes at the smallest corpus sizes should not be emphasized. At those scales:

calls are extremely short;
timer and wrapper overhead have greater influence;
a small latency change creates a large throughput swing;
cache residency is unusually favorable.

The stable right-hand region is the more defensible steady-state result.

One methodological note: “corpus shingles/s” measures corpus traversal, but candidates range from 20 to 4,015 shingles. It is therefore a useful normalized throughput metric, but not a complete count of all pairwise comparison operations. Keep candidate-size sensitivity as a complementary figure.

3. Memory growth

The memory result is almost perfectly linear, exactly as expected.

At the largest workload, around 372,000 documents:

packed input is roughly 500 MiB;
output arrays are roughly 8.5 MiB;
total estimated native memory stays under approximately 600 MiB.

The important observation is that packed corpus storage dominates memory. The three output arrays remain comparatively small:

Moutput=24NdocumentsM_{\text{output}} = 24N_{\text{documents}}

For 372,000 documents:

24×372,000=8,928,000 bytes≈8.51 MiB24 \times 372{,}000 = 8{,}928{,}000\text{ bytes} \approx 8.51\text{ MiB}

So single-candidate search scales comfortably in memory.

Batched search will be different because output memory becomes:

24NdocumentsNcandidates24N_{\text{documents}}N_{\text{candidates}}

That will be worth showing separately during the batch experiment.

Main conclusions so far

A defensible summary is:

On a production-derived, variable-length corpus, the dedicated serial Fortran backend provides the best one-thread performance, while OpenMP becomes advantageous as workload granularity increases. Across the six physical cores of the AMD Ryzen 5 5500, OpenMP reaches approximately 4× median speedup and sustains roughly 700 million corpus shingles per second at large workload sizes. Scaling is sublinear, consistent with parallel-region overhead, variable per-document work, synchronization, and shared-memory resource contention. Estimated native memory grows linearly and is dominated by packed corpus storage rather than output arrays.

Two small graph improvements
Speedup graph

Rename:

Serial crossover


to:

Break-even: speedup = 1


That is more precise.

Throughput graph

Format the y-axis in millions:

Corpus shingles processed per second (millions)


instead of displaying 1e8 above the axis. It will be easier to read in the README.

Overall verdict

The results are not suspicious. The three figures reinforce one another:

speedup increases with threads
→ throughput increases proportionally
→ memory grows linearly with workload


The headline result is currently:

OpenMP-6 steady-state speedup: approximately 4×
OpenMP-6 steady-state throughput: approximately 0.7 billion corpus shingles/s
OpenMP-6 physical-core efficiency: approximately 65–68%

===

Yep, these complete the native single-candidate, physical-core lexical benchmark. The six figures now form a consistent story.

1. Latency versus corpus size

The latency figure shows nearly linear growth on the log-log axes:

more documents
→ proportionally more candidate-document comparisons
→ proportionally greater latency


The ordering is stable after the smallest workloads:

OpenMP-6 fastest
OpenMP-4
OpenMP-2
Fortran serial
OpenMP-1 slowest


The important result is that OpenMP-1 remains slightly slower than dedicated serial. This captures the overhead of entering the OpenMP parallel region even when only one worker executes.

The shaded candidate-dependent variation expands at larger workloads because the candidates range from 20 to 4,015 shingles.

2. Physical-core parallel efficiency

This figure explains the difference between small and large workloads.

At 372 documents
OpenMP-2: ~85%
OpenMP-4: ~63%
OpenMP-6: ~51%


There is too little work to keep all six workers productive after accounting for OpenMP overhead and variable document lengths.

At 372,000 documents
OpenMP-2: ~86%
OpenMP-4: ~76%
OpenMP-6: ~67%


Large workloads amortize parallel-region and synchronization overhead much better.

This supports a clear conclusion:

Increasing the thread count is useful only when the corpus provides sufficient candidate-document work. Larger corpora achieve higher parallel efficiency because fixed OpenMP costs become a smaller portion of total execution time.

The one-thread values around 89–91% are also sensible:

OpenMP-1 efficiency
=
OpenMP-1 speedup / 1


This directly measures OpenMP runtime overhead against the dedicated serial routine.

3. Candidate-size sensitivity

This is an excellent figure because candidate length clearly matters.

Across all four corpus sizes:

20 shingles     → lowest latency
53–295 shingles → gradually increasing latency
581 shingles    → further increase
4,015 shingles  → substantially higher latency


The same ordering appears for both serial and OpenMP-6, showing that the behavior comes from the exact sorted-set intersection workload, not merely the execution backend.

The two candidates with the same shingle count can have different latency. That is not necessarily an error. Equal lengths can still produce different traversal behavior because of:

where matching values occur;
how quickly one sorted set becomes exhausted;
different intersection sizes;
different branch patterns against the corpus documents.

This figure confirms that document count alone does not completely describe lexical workload granularity. Candidate shingle count also changes computational cost.

Combined findings from all six figures

The six figures collectively establish:

Latency grows approximately linearly with corpus size.
OpenMP-1 is slower than the dedicated serial backend.
OpenMP-2, OpenMP-4, and OpenMP-6 provide progressively greater throughput.
OpenMP-6 stabilizes near approximately 4× speedup at large corpus sizes.
Six-core efficiency reaches roughly 65–68% for sufficiently large workloads.
The steady-state OpenMP-6 throughput is approximately 0.7 billion corpus shingles per second.
Longer candidates increase scoring cost materially.
Candidate-dependent variability becomes relatively bounded at scale.
Packed corpus storage dominates memory usage.
Memory consumption grows linearly with corpus size.
What this experiment completes

Completed:

Native single-candidate search
Dedicated serial Fortran
OpenMP-1/2/4/6 physical-core scaling
Corpus-size crossover
Candidate-size sensitivity
Latency and throughput scaling
Parallel speedup and efficiency
Estimated native memory growth


Still separate, optional follow-up experiments:

OpenMP-6/8/12 SMT sensitivity
Batched-candidate Jaccard evaluation
Public JaccardIndex.search() end-to-end overhead
F2PY versus nanobind/C ABI diagnostic


So this is a complete and publishable physical-core result section, even without those follow-ups.

Next immediate checks

Inspect the completeness report:

cat \
  perf_eval/results/processed/lexical_real_single_completeness_report.json


It should confirm:

600/600 configurations
8 candidates
15 workload scales
5 execution variants
0 duplicate rows
0 repetition mismatches
== yes confirmed.

Then inspect the generated table:

cat \
  perf_eval/results/tables/lexical_cpu_readme_table.md


After verifying those, the next step is not another large experiment. It is to write a concise CPU Performance README section using:

one compact table;
the speedup figure;
the latency figure;
a methodology note;
a link to the complete figures and processed results.