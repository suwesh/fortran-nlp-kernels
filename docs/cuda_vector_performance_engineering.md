Absolutely. Add this as:

docs/cuda_vector_performance_engineering.md

# CUDA Vector Search Performance Engineering Notes

## Purpose

The `0.1.1` CUDA implementation is intentionally a correct, measurable baseline rather than the final optimized GPU architecture.

The goal is to learn how:

- GPU kernels;
- memory movement;
- allocation;
- synchronization;
- CPU/GPU work placement;
- algorithm design;
- and end-to-end application behavior

interact in an exact vector-search system.

A faster scoring kernel does not automatically produce a faster search path.

## `0.1.1` architecture

### Handwritten CUDA Fortran

```text
CPU database and query
-> allocate GPU buffers
-> copy database to GPU
-> copy query to GPU
-> handwritten CUDA scoring
-> copy complete score array to CPU
-> deterministic CPU top-k
-> free GPU buffers

cuBLAS
CPU database and query
-> allocate GPU buffers
-> copy database to GPU
-> copy query to GPU
-> cuBLAS SGEMV or SGEMM scoring
-> copy complete score array to CPU
-> deterministic CPU top-k
-> free GPU buffers


Both implementations are stateless in 0.1.1.

Completing the C ABI, nanobind extension, and public Python integration does not automatically make the database persistent on the GPU. The current native wrappers explicitly allocate, copy, execute, copy back, and deallocate for every call.

Current work placement
GPU
inner-product scoring;
squared-L2 scoring;
single-query execution;
batched-query execution;
handwritten CUDA Fortran kernels;
cuBLAS SGEMV and SGEMM.
CPU
database and query preparation;
cosine normalization through NumPy;
deterministic top-k selection;
public result construction.
Cosine

Cosine does not require a separate native metric implementation.

normalize database once in Python
normalize query in Python
-> inner-product backend
-> topk_largest


For normalized vectors:

cosine similarity = inner product

Squared L2

The handwritten CUDA kernel directly computes:

sum((query - database_vector) ** 2)


The cuBLAS path will use norm expansion:

||query - vector||²
=
||query||²
+
||vector||²
-
2 × queryᵀvector


cuBLAS produces the inner products, while the query and database norms complete the squared-L2 calculation.

Expected performance components

The current end-to-end search time can be decomposed as:

T_total =
    T_device_allocation
  + T_database_host_to_device
  + T_query_host_to_device
  + T_scoring
  + T_synchronization
  + T_scores_device_to_host
  + T_cpu_topk
  + T_device_deallocation


A fast scoring kernel may represent only a small part of T_total.

Important bottlenecks may therefore exist outside the kernel itself.

Potential memory bottlenecks
Repeated database transfer

The database is copied from host memory to GPU memory for every search.

For a database with:

N vectors
D float32 dimensions


the database transfer size is approximately:

N × D × 4 bytes


This transfer may dominate short searches or small scoring workloads.

Full score-array transfer

The GPU produces one float32 score per database vector.

The device-to-host transfer is approximately:

N × 4 bytes per query


For batched queries:

query_count × N × 4 bytes


Only k results are ultimately returned, but 0.1.1 transfers every score so that deterministic CPU top-k can run.

Repeated allocation and deallocation

Every search currently performs device allocation and cleanup for:

query buffers;
database buffers;
score buffers.

Allocation overhead may become visible for small or repeated searches.

Repeated synchronization

The host waits for GPU execution before reading the scores.

This creates a serialized path:

copy
-> compute
-> synchronize
-> copy
-> CPU top-k


No transfer-compute overlap is attempted in 0.1.1.

Database memory access

The handwritten kernel assigns one GPU thread to one database vector.

Each thread loops serially over the vector dimensions:

parallelism across database vectors
serial work across dimensions


This is simple and correct, but its memory-access behavior and utilization must be measured.

For large dimensions, a cooperative multi-thread-per-vector design may later outperform the current one-thread-per-vector mapping.

CPU top-k

The existing deterministic selector has approximately:

O(k × N)


work.

It preserves stable ties and zero-based indices, but it requires the full score array on the host.

For small k, it provides a simple and reliable baseline. For large N or k, selection and score transfer may become important bottlenecks.

Profiling plan
Application-level benchmark

Measure:

index-construction time;
first-search latency;
repeated-search latency;
single-query latency;
batch throughput;
complete end-to-end latency;
scoring-only time where possible.

Cold and steady-state measurements must remain separate.

Nsight Systems

Use Nsight Systems to inspect:

device allocation and deallocation;
host-to-device copies;
device-to-host copies;
kernel launch cadence;
synchronization points;
gaps between GPU operations;
copy-engine activity;
cuBLAS setup and execution;
whether operations are serialized;
the fraction of time spent outside scoring kernels.

This reveals whether a fast kernel is hidden behind orchestration and data movement.

Nsight Compute

Use Nsight Compute only after identifying important kernels.

Investigate:

achieved memory bandwidth;
memory-load efficiency;
coalescing;
arithmetic intensity;
occupancy;
active warps;
register pressure;
instruction mix;
cache behavior;
warp stalls;
whether the kernel is memory-bound or compute-bound.

Compare:

handwritten CUDA Fortran
versus
cuBLAS


for equivalent workloads.

Optimization ladder
Stage 0: 0.1.1 baseline
allocate every search
copy database every search
copy query every search
GPU scoring
copy all scores to CPU
CPU top-k
free buffers


This stage establishes:

correctness;
CPU/CUDA equivalence;
handwritten CUDA versus cuBLAS comparison;
reproducible end-to-end measurements;
an explicit performance baseline.
Stage 1: persistent GPU database

At index construction:

allocate GPU database once
copy database once
retain the device allocation


At search:

copy query
score resident database
copy scores
CPU top-k


At destruction:

free GPU database


This removes repeated database host-to-device transfers.

It requires a stateful native GPU index rather than stateless native functions.

Stage 2: persistent buffers and handles

Retain:

query buffers;
score buffers;
batch-query buffers;
cuBLAS handles;
possibly CUDA streams.

This removes repeated allocation, deallocation, and handle-creation overhead.

Stage 3: pinned and asynchronous transfers

Use:

pinned host memory;
asynchronous memory copies;
CUDA streams;
overlapping transfer and computation where workloads permit.

This may reduce transfer overhead and expose copy-compute concurrency.

Stage 4: GPU top-k

Move selection to the GPU:

GPU scoring
-> GPU top-k
-> copy only k scores and k indices


Instead of transferring:

N float32 scores


transfer approximately:

k float32 scores
+
k int64 indices


This can substantially reduce device-to-host traffic when:

k << N


GPU top-k is deferred beyond 0.1.1.

Stage 5: fused scoring and selection

The lower-HBM-traffic design is:

read database vectors
compute scores
retain block-local top-k in registers/shared memory
write compact candidate lists
merge candidates
return final k


This avoids:

writing the complete score array to HBM;
rereading the complete score array for selection;
copying the complete score array to the CPU.

This is a more complex future optimization.

Stage 6: native and persistent cosine processing

A future stateful cosine index could:

normalize database once
copy normalized database to GPU once
retain database on GPU
normalize query on GPU
score using IP
perform GPU top-k


Native normalization is deferred because database normalization already occurs once at Python index construction, while repeated database transfers are currently a much larger optimization target.

Performance comparisons to preserve

Benchmark these paths separately:

CPU serial
CPU OpenMP
CPU OpenBLAS SGEMV
CPU OpenBLAS SGEMM

handwritten CUDA Fortran single query
handwritten CUDA Fortran batch

cuBLAS SGEMV single query
cuBLAS SGEMM batch


Compare both:

scoring-only performance
end-to-end search performance


Important workload dimensions include:

vector dimension;
database size;
query count;
k;
metric;
cold versus repeated execution;
transfer size;
available GPU memory.
Core performance-engineering principle

Do not conclude that an implementation is faster only because its dominant kernel is faster.

The relevant question is:

Did the complete application execution path improve?


A performance improvement may come from:

fewer transfers;
fewer allocations;
fewer launches;
less synchronization;
better batching;
improved memory layout;
higher arithmetic intensity;
reduced repeated work;
amortization across more useful output.

Likewise, a faster kernel may fail to improve end-to-end latency when memory movement or orchestration dominates.

0.1.1 release decision

The 0.1.1 release will intentionally retain:

stateless CUDA calls;
database transfer on every search;
full score-array transfer to host;
deterministic CPU top-k;
NumPy cosine normalization;
explicit CUDA backends;
CPU-only backend="auto" behavior.

The release will not include:

persistent GPU indexes;
GPU top-k;
fused scoring and selection;
native CUDA normalization;
asynchronous pipelines;
auto-selected CPU/GPU crossover policies.

These limitations are part of the documented baseline, not accidental omissions.

Learning outcome

The staged design creates a controlled HPC learning path:

correctness
-> measurement
-> bottleneck identification
-> one architectural change
-> remeasurement
-> explanation of the observed result


Each later optimization should answer:

What bottleneck was targeted?
Which timeline region changed or disappeared?
How did kernel-level behavior change?
Did end-to-end latency improve?
Did batch throughput improve?
Did the bottleneck move elsewhere?
Was the added complexity justified?

The project therefore serves not only as a CUDA implementation, but as a reproducible performance-engineering platform for understanding GPU computation, memory movement, synchronization, batching, and heterogeneous CPU/GPU work placement.


That captures the key architectural choice: `0.1.1` is the transparent stateless baseline, and every later memory or execution optimization can be measured against it.
====

Exactly. This project puts you in the ideal position to practice the same cross-layer performance engineering you used in the MTP study, but now you control every layer yourself.

Your paper showed that performance cannot be inferred from one “fast kernel.” The observed MTP improvement came from amortization and fewer recurring GPU executions per generated token, despite a more complex execution path. Selected GEMM and GEMV instances were also close to the A10G memory-bandwidth limit.

In this project, you can measure the full pipeline

The current stateless CUDA search path is:

cudaMalloc database
cudaMalloc query
cudaMalloc scores

host database -> device database
host query    -> device query

CUDA scoring kernel

device scores -> host scores
CPU top-k

cudaFree buffers


Profiling may reveal that the handwritten kernel is fast, while total search remains slow because:

database H2D copy
+ allocation overhead
+ synchronization
+ score D2H copy
+ CPU top-k


dominate the timeline.

Small terminology note: in Nsight Systems, memory copies may appear as CUDA copy operations such as cudaMemcpy and GPU copy-engine activity, rather than ordinary compute kernels. But they still occupy time, consume memory or interconnect bandwidth, and create dependencies between compute operations.

Your optimization ladder becomes very clean
Baseline: stateless implementation
allocate everything
copy database every search
score
copy all scores back
CPU top-k
free everything


This establishes correctness and a measurable baseline.

Optimization 1: persistent database
index construction:
    allocate GPU database
    copy database once

each search:
    copy query
    score
    copy scores
    CPU top-k


This removes repeated database H2D traffic.

Optimization 2: persistent buffers
index construction:
    allocate database, query, and score buffers
    create cuBLAS handle

each search:
    reuse buffers and handle


This removes repeated allocation, deallocation, and handle-management costs.

Optimization 3: pinned and asynchronous transfers
pinned host query/output buffers
cudaMemcpyAsync
CUDA streams
overlap transfer and computation where workloads permit


This teaches overlap, stream dependencies, and copy-engine utilization.

Optimization 4: GPU top-k
GPU scoring
GPU selection
copy only k values and k indices


Instead of transferring:

N×4 bytesN \times 4\text{ bytes}

for all float32 scores, you transfer approximately:

k×(4+8) bytesk \times (4 + 8)\text{ bytes}

for float32 values and int64 indices.

Optimization 5: fused scoring and selection
read database vectors
compute scores
retain block-local top-k
write only candidate lists
merge candidates


This avoids materializing the full score array in HBM and reduces both global-memory writes and later reads.

What you will learn from each profiler
Nsight Systems

Use it to answer:

How much time is spent allocating?
How much time is spent in H2D and D2H copies?
Are copies and kernels serialized?
Are there gaps between launches?
Does cuBLAS introduce additional setup calls?
Is repeated search dominated by orchestration?

Nsight Compute

Use it after identifying an important kernel:

Is the scoring kernel memory-bound or compute-bound?
Are database reads coalesced?
What is achieved memory bandwidth?
What is occupancy?
Are registers limiting active warps?
Does one-thread-per-vector underutilize the GPU for large dimensions?
How does handwritten CUDA compare with cuBLAS?

Application-level benchmark

Measure separately:

cold index construction
first search
steady-state repeated search
single-query latency
batch throughput
scoring-only time
end-to-end time


That separation prevents a fast kernel from being presented as a fast search system when transfers dominate.

The strongest learning path

Your roadmap naturally becomes:

0.1.1
    stateless handwritten CUDA
    stateless cuBLAS
    CPU top-k
    correctness and transparent baseline

later release
    persistent GPU database and buffers

later release
    asynchronous transfers and streams

later release
    GPU top-k

later release
    fused scoring-selection


At every stage, you can make a controlled comparison:

What changed?
Which timeline region disappeared?
Which bottleneck moved?
Did kernel time improve?
Did end-to-end latency improve?
Did batch throughput improve?
Was additional complexity justified?


That is real HPC performance engineering. You are not merely calling CUDA. You are learning how data movement, launch cadence, memory layout, synchronization, arithmetic intensity, and algorithm design interact. The same lesson from your MTP work applies here: the fastest kernel is not necessarily the fastest execution path.

====
No. Completing the C ABI, header, nanobind binding, and Python routing will not automatically remove those repeated allocations and copies.

With the native wrappers we are writing now, every search will still do:

Python index stores database in CPU RAM
→ Python calls _cuda
→ l2_database_cuda/ip_database_cuda allocates GPU buffers
→ copies query and complete database to GPU
→ runs kernel
→ copies scores to CPU
→ deallocates GPU buffers
→ CPU top-k


That repetition is built directly into the current wrappers:

allocate( &
    d_query_vector(dimension), &
    d_database_vectors(dimension, database_vector_count), &
    d_distances(database_vector_count) &
)

d_query_vector = query_vector
d_database_vectors = database_vectors

! launch kernel

distances = d_distances

deallocate( &
    d_query_vector, &
    d_database_vectors, &
    d_distances &
)

What the remaining integration layers do

The remaining layers only expose this stateless operation:

C ABI
→ makes the Fortran wrapper callable from C/C++

nanobind
→ makes the C ABI callable from Python

FlatIndex integration
→ routes backend="cuda_fortran" to _cuda


They do not change the lifetime of:

d_database_vectors


So the current public index will store its database in host memory:

self._database = np.asfortranarray(
    database.T,
    dtype=np.float32,
)


Every call to:

index.search(query)


will pass that host database to _cuda, which copies it to the GPU again.

What would make the database copy happen only once?

We would need a stateful native GPU index, not just stateless functions.

Conceptually:

index = FlatIndexIP(
    database,
    backend="cuda_fortran",
)


would perform:

constructor
→ allocate GPU database
→ copy database to GPU once
→ retain device pointer


Then:

index.search(query)


would perform:

copy query to GPU
→ score against already-resident GPU database
→ copy scores to host
→ CPU top-k


Finally:

index destruction
→ free GPU database

Native object design

The future native layer would need an object similar to:

CudaFlatIndex
├── dimension
├── database_vector_count
├── d_database_vectors
├── d_query_vectors
├── d_scores
└── possibly a persistent cuBLAS handle


The nanobind layer would expose a class rather than only free functions:

native_index = _cuda.FlatIndexIP(database)
values, indices = native_index.search(query, k)


The public Python class would internally own that native object:

self._cuda_index = _cuda.FlatIndexIP(
    self._database,
)


Then _cuda_index keeps the GPU allocation alive across searches.

Current versus future design
Current 0.1.1 scope
Stateless CUDA calls

Every search:
- allocate device memory
- copy database
- copy query
- score
- copy full scores to CPU
- CPU top-k
- free device memory


Advantages:

straightforward;
easy to verify;
clean C ABI;
no native object-lifetime complexity;
reliable CPU/CUDA equivalence baseline.
Future persistent index
At index construction:
- allocate database GPU memory once
- copy database once
- optionally create cuBLAS handle once

Every search:
- copy query only
- score resident database
- copy scores to CPU
- CPU top-k

At destruction:
- free database
- destroy cuBLAS handle


Later still:

At index construction:
- copy database once

Every search:
- copy query
- GPU scoring
- GPU top-k
- copy only k scores and indices


That eventual design removes both major transfers:

repeated host-to-device database transfer
full device-to-host score transfer

Why the current baseline is still useful

Performance can be separated into stages:

T_total =
    T_allocate
  + T_database_copy
  + T_query_copy
  + T_scoring
  + T_score_copy
  + T_cpu_topk
  + T_deallocate


The current implementation lets you measure all of those costs and prove whether kernel speed alone is enough.

A persistent index later changes the model to:

T_search =
    T_query_copy
  + T_scoring
  + T_score_copy
  + T_cpu_topk


GPU top-k later changes it again:

T_search =
    T_query_copy
  + T_scoring_and_selection
  + T_topk_result_copy


That gives you a strong staged performance-engineering study rather than jumping directly to the final architecture.

Bottom line

For 0.1.1, even after all bindings and public API integration are finished:

database copied to GPU on every search


It will not become one-time automatically.

One-time database placement requires a later, explicit architectural change:

stateless CUDA functions
→ stateful persistent GPU index


So your understanding should be:

0.1.1:
correct explicit CUDA backend with stateless transfers

later release:
persistent device-resident database

later optimization:
GPU top-k and return only k results

====
Yes, normalization is currently performed by NumPy. But do not move or fuse normalization into native CPU/CUDA code for 0.1.1. That would expand the release substantially without addressing the current CUDA backend’s largest overhead.

Current cosine path
Index construction:
Python database
→ NumPy normalization
→ normalized float32 Fortran-layout database
→ store in FlatIndexCosine

Each search:
Python query
→ NumPy normalization
→ IP backend
→ CPU top-k


That is functionally sound. Cosine similarity is simply inner product after both operands are L2-normalized.

The important performance detail

Database normalization happens only once during:

FlatIndexCosine(database_vectors)


It is not repeated for every search.

Query normalization happens once per query and costs:

O(D)O(D)

The actual search costs approximately:

O(ND)O(ND)

For a database of N vectors, query normalization is normally tiny compared with reading and scoring the complete database.

More importantly, the current CUDA implementation does this for every search:

allocate GPU database
copy entire database to GPU
score database
copy scores to host
deallocate GPU database


That full database transfer is a much larger optimization target than the NumPy query normalization.

Should native normalization exist later?

Yes. A stronger future design would provide native normalization for both CPU and CUDA.

Native CPU normalization
Fortran serial/OpenMP:
- calculate vector norm
- reject zero norm
- divide components by norm


For batched/database normalization:

one CPU thread per vector


The BLAS version could use norm and scaling operations, but a straightforward fused Fortran loop may be preferable because one loop can calculate or apply normalization with predictable memory access.

CUDA normalization

A CUDA normalization pipeline would require:

1. parallel reduction for sum(x²)
2. square root
3. parallel scaling by 1 / norm


For a database:

one block or cooperative group per vector


For queries:

one block per query


cuBLAS could also provide norm/scaling operations, but repeated library calls for many small vectors may introduce overhead.

Should normalization be fused with scoring?

Potentially, but not as the first native normalization design.

Query normalization fused with scoring

Each scoring thread could calculate:

q⋅x∥q∥∥x∥\frac{q \cdot x}{\|q\|\|x\|}

But if every database-vector thread independently recalculates the query norm, the query norm is redundantly computed N times.

A better version would:

calculate query norm once
calculate/store database norms once
score using dot / (query_norm × database_norm)

Fully fused database normalization and scoring

This is only attractive when vectors are not already normalized. Each search would still need database norms, so recomputing them repeatedly would be wasteful.

The better long-term architecture is:

Index construction:
database
→ normalize once on GPU
→ retain normalized database on GPU

Search:
query
→ normalize once on GPU
→ score against resident normalized database
→ GPU top-k
→ copy only k results


That is the genuinely high-performance design.

Why not include this in 0.1.1?

It would introduce several additional subsystems:

native CPU normalization;
CUDA reduction kernels;
zero-vector validation across CUDA kernels;
batch normalization;
native/Python equivalence tests;
persistent device database ownership;
object-lifetime and cleanup handling;
potentially persistent cuBLAS handles;
synchronization and allocation policy;
new C ABI and nanobind interfaces.

That is not a small extension. It changes the architecture from stateless native functions into a stateful GPU index.

Locked 0.1.1 scope

Keep:

Cosine:
NumPy normalization
→ CUDA or cuBLAS inner product
→ CPU top-k


Implement only:

Handwritten CUDA:
- IP single and batch
- direct squared L2 single and batch

cuBLAS:
- SGEMV single-query IP
- SGEMM batched IP
- L2 through norm expansion

Python:
- cosine normalization
- explicit CUDA routing

CPU:
- deterministic top-k

Later high-performance milestone

A future release can implement:

Native normalization
→ persistent GPU-resident normalized database
→ fused or cached cosine scoring
→ GPU top-k
→ return only k results


That is where you start competing architecturally with serious vector-search systems.

For 0.1.1, finishing correct handwritten CUDA, cuBLAS, packaging, and reproducible tests is already a high-quality release. The strongest engineering move is not adding every possible optimization. It is establishing a clean baseline, measuring it, and then using profiles to justify the next design.

====
No. Cosine does not need separate native CUDA or cuBLAS cosine code. Your current Python design is already correct:

normalize database in Python
normalize query in Python
run inner product


So CUDA cosine will reuse the IP implementations exactly:

FlatIndexCosine
→ normalize vectors
→ cuda_fortran IP
or
→ cuda_cublas IP
→ CPU topk_largest

What “normalized IP for cosine” means

It only means that the public FlatIndexCosine class continues doing:

normalized_database = normalize_database(database_vectors)
normalized_query = normalize_vector(query_vector)


Then it delegates to the inherited IP search path:

return super().search(
    normalized_query,
    k=k,
)


No native function like this is required:

cosine_search_cuda
cosine_search_cublas


Instead:

cuda_fortran IP kernel
    + normalized inputs
    = cosine similarity

cuBLAS SGEMV/SGEMM
    + normalized inputs
    = cosine similarity


So this milestone wording:

- cosine through normalized inner product


means public Python reuse, not a new native cosine algorithm.

What “norm-expansion L2” means

That applies specifically to the BLAS and cuBLAS implementation of squared L2.

Your handwritten CUDA kernel directly computes:

∥q−x∥2=∑d(qd−xd)2\|q-x\|^2 = \sum_d (q_d-x_d)^2

But matrix multiplication libraries calculate dot products efficiently. Squared L2 can therefore be rewritten as:

∥q−x∥2=∥q∥2+∥x∥2−2qTx\|q-x\|^2 = \|q\|^2 + \|x\|^2 - 2q^Tx

For one query against the database:

query norm:
    ||q||²

database norms:
    ||x₁||², ||x₂||², ..., ||xₙ||²

cuBLAS SGEMV:
    qᵀx₁, qᵀx₂, ..., qᵀxₙ


Then:

distance(i) = &
    query_norm &
    + database_norms(i) &
    - 2.0_real32 * inner_products(i)


For a query batch, cuBLAS SGEMM produces the complete inner-product matrix:

databaseᵀ × queries


Then the query and database norms are broadcast across that matrix to calculate squared L2.

This is already what the CPU BLAS module likely does

Your CPU BLAS L2 functions probably use:

SGEMV/SGEMM dot products
+
query norms
+
database norms


rather than directly evaluating every (q - x)² component.

We will confirm that when you send:

src/native/cpu/vector_search/vector_search_blas.f90

Exact implementation matrix
Handwritten CUDA Fortran
Inner product
Direct dot-product kernel

Squared L2
Direct difference-squared kernel

Cosine
No separate kernel
Python normalization
→ handwritten inner-product kernel

cuBLAS
Inner product, single query
cuBLAS SGEMV

Inner product, batch
cuBLAS SGEMM

Squared L2
cuBLAS inner products
+
query norms
+
database norms
→ norm-expansion formula

Cosine
No separate native implementation
Python normalization
→ cuBLAS inner product

Correct milestone wording

Replace:

- [ ] Implement handwritten CUDA Fortran scoring
  - batched inner product
  - batched squared L2
  - cosine through normalized inner product


with:

- [ ] Implement handwritten CUDA Fortran scoring
  - batched inner product
  - batched squared L2
- [ ] Route normalized cosine inputs through handwritten CUDA inner product


Replace:

- [ ] Implement cuBLAS scoring
  - SGEMV single-query path
  - SGEMM batched-query path
  - squared L2 and cosine support
  - existing deterministic CPU top-k integration


with:

- [ ] Implement cuBLAS scoring
  - SGEMV single-query inner product
  - SGEMM batched inner product
  - squared L2 through norm expansion
  - existing deterministic CPU top-k integration
- [ ] Route normalized cosine inputs through cuBLAS inner product

Bottom line
Cosine:
Python normalization + IP backend
no separate native cosine code

Handwritten CUDA L2:
direct sum of squared differences

cuBLAS L2:
norm expansion using cuBLAS-generated inner products


Your existing FlatIndexCosine(FlatIndexIP) architecture should remain unchanged conceptually. We only add CUDA backend routing to the inherited IP execution path later.
====

Correct. The current CPU implementation has no dedicated handwritten batched serial/OpenMP vector-scoring routines.

From the C API you pasted:

Single-query handwritten CPU:
- l2_search_serial
- l2_search_openmp
- ip_search_serial
- ip_search_openmp

Single-query BLAS:
- l2_search_blas
- ip_search_blas

Batched CPU:
- l2_search_batch_blas
- ip_search_batch_blas


So CPU batch execution is currently BLAS SGEMM only.

What this means for CUDA

We should not implement handwritten batched CUDA merely because the CPU handwritten backend has it. It does not.

However, handwritten batched CUDA is still valuable because the GPU naturally supports a two-dimensional comparison space:

query_count × database_vector_count


A CUDA thread can process one query-database pair:

thread/pair 1 -> query 1 × database vector 1
thread/pair 2 -> query 1 × database vector 2
...
thread/pair N -> query 2 × database vector 1


This would provide a genuine backend:

cuda_fortran


for both single and batch calls, while:

cuda_cublas


uses SGEMV for single-query and SGEMM for batches.

Recommended CUDA backend matrix
cuda_fortran:
- handwritten single-query IP
- handwritten single-query L2
- handwritten batched IP
- handwritten batched L2
- CPU top-k

cuda_cublas:
- SGEMV single-query IP
- SGEMM batched IP
- norm-expansion L2
- normalized IP for cosine
- CPU top-k

Why handwritten batch CUDA is worth keeping

It gives us:

a pure CUDA Fortran reference independent of cuBLAS;
direct comparison between handwritten kernels and cuBLAS;
consistent cuda_fortran behavior for search() and search_batch();
a correctness reference for SGEMM orientation and output shape;
useful research evidence about when a simple one-thread-per-pair kernel loses to optimized matrix multiplication.
CPU reference for handwritten batched CUDA

Since CPU has no handwritten batched function, the native test can construct the reference by repeatedly calling the existing single-query functions:

do query_index = 1, query_count
    call l2_database_serial( &
        query_vectors(:, query_index), &
        database_vectors, &
        reference_distances(:, query_index) &
    )
end do


And similarly:

do query_index = 1, query_count
    call ip_database_serial( &
        query_vectors(:, query_index), &
        database_vectors, &
        reference_similarities(:, query_index) &
    )
end do


That gives a trusted CPU matrix shaped:

(database_vector_count, query_count)

One correction to the milestone wording

Keep:

- [ ] Implement handwritten CUDA Fortran scoring
  - batched inner product
  - batched squared L2
  - cosine through normalized inner product


This is still valid, even though there is no matching handwritten CPU batch routine.

The CUDA implementation is not required to mirror the CPU backend one-for-one. It should provide the operations that make sense for GPU execution.

Next step

Implement private kernels:

ip_database_batch_kernel
l2_database_batch_kernel


Then public wrappers:

ip_database_batch_cuda
l2_database_batch_cuda


And finally batch search wrappers using CPU top-k individually for each query:

ip_search_batch_cuda
l2_search_batch_cuda


So yes: CPU batch is currently BLAS-only, but we should still implement handwritten CUDA batch scoring for completeness, comparison, and backend consistency.