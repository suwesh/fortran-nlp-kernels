# Native SGEMM CPU Microarchitecture Study

**Checkpoint date:** 2026-11-26  
**Project:** `fortran-nlp-kernels`  
**Branch:** `cpu-simd-study`  
**CPU:** Intel Core i5-1135G7, Tiger Lake  
**Profiling tool:** Intel VTune Profiler  
**Compilers:** GNU Fortran and Intel ifx  

---

## 1. Objective

Investigate how compiler choice and SIMD targeting affect a native, single-threaded Fortran SGEMM-like kernel on Intel Tiger Lake.

The controlled variables were:

- compiler: GFortran versus Intel ifx;
- vectorization mode: scalar, AVX2, AVX-512 target, and forced 512-bit width;
- emitted arithmetic width: scalar XMM, 256-bit YMM, or 512-bit ZMM;
- VTune microarchitecture metrics;
- dynamic instruction count, clockticks, CPI, Core Bound, Memory Bound, and Vector Capacity Usage.

The study deliberately excluded:

- OpenMP;
- OpenBLAS, oneMKL, AOCL, BLIS, or another external BLAS implementation;
- Python timing;
- the package's top-k path;
- multi-core scaling.

The purpose was to keep the chain direct:

```text
Fortran source
    -> compiler optimization
    -> generated SIMD instructions
    -> CPU execution
    -> VTune evidence
```

---

## 2. Research questions

1. Does a native Fortran SGEMM loop benefit more from wider SIMD than the earlier inner-product reduction kernel?
2. How do GFortran and ifx differ on the same source and CPU?
3. Does targeting AVX-512 guarantee the use of 512-bit ZMM arithmetic?
4. Does 100% Vector Capacity Usage imply the fastest implementation?
5. How do instruction count, clockticks, CPI, Core Bound, and cache behavior explain the observed runtimes?
6. Does the compiler-selected vector width outperform a forced maximum-width implementation?

---

## 3. Source inspection

The benchmark used a native Fortran matrix multiplication rather than an external `sgemm` library call. The central update was equivalent to:

```fortran
C(i, j) = C(i, j) + A(i, k) * B(k, j)
```

The loop ordering exposed a contiguous `i` loop to the compiler:

```fortran
do j = 1, n
    do k = 1, n
        do i = 1, n
            C(i, j) = C(i, j) + A(i, k) * B(k, j)
        end do
    end do
end do
```

This is not a tuned BLAS implementation. It has no explicit packing, cache blocking, threading, or handwritten intrinsics. It exists to test compiler-generated SIMD from straightforward Fortran.

---

## 4. Benchmark configuration

The final profiling workload used:

```text
Matrix dimension n = 512
Iterations          = 500
Precision           = FP32
Threading           = single-threaded
```

Each matrix contains:

```text
512 x 512 x 4 bytes = 1 MiB
```

Nominal matrix working set:

```text
A = 1 MiB
B = 1 MiB
C = 1 MiB
Total = 3 MiB
```

The working set exceeds the private L1 and L2 caches of one core. Therefore, the final experiment measures both SIMD execution and cache-hierarchy behavior. It is not the same cache-resident workload as the earlier 768-element inner-product benchmark.

The result value, such as `C(1,1)`, was printed after execution to preserve an observable output and discourage dead-code elimination. The printed floating-point value was a correctness/anti-elimination sink, not a timing metric.

---

## 5. Build matrix

### GFortran scalar

```bash
gfortran \
  -O3 \
  -fno-tree-vectorize \
  cpu-simd-study/benchmark_native_sgemm.f90 \
  -o bench_nativesgemm_gfortran_scalar
```

### GFortran AVX2

```bash
gfortran \
  -O3 \
  -mavx2 \
  cpu-simd-study/benchmark_native_sgemm.f90 \
  -o bench_nativesgemm_gfortran_avx2
```

### GFortran forced AVX-512 width

`-march=tigerlake` alone still selected YMM-width arithmetic. Full-width ZMM code required a width preference:

```bash
gfortran \
  -O3 \
  -march=tigerlake \
  -mprefer-vector-width=512 \
  cpu-simd-study/benchmark_native_sgemm.f90 \
  -o bench_nativesgemm_gfortran_avx512
```

### ifx scalar

```bash
ifx \
  -O3 \
  -no-vec \
  cpu-simd-study/benchmark_native_sgemm.f90 \
  -o bench_nativesgemm_ifx_scalar
```

### ifx AVX2

```bash
ifx \
  -O3 \
  -xCORE-AVX2 \
  cpu-simd-study/benchmark_native_sgemm.f90 \
  -o bench_nativesgemm_ifx_avx2
```

### ifx AVX-512 target, compiler-selected width

```bash
ifx \
  -O3 \
  -xCORE-AVX512 \
  cpu-simd-study/benchmark_native_sgemm.f90 \
  -o bench_nativesgemm_ifx_avx512
```

This binary used YMM-width arithmetic in the observed hotspot. Register use above `ymm15` indicated use of the expanded AVX-512 register architecture/AVX-512VL, even though arithmetic remained 256-bit.

### ifx forced 512-bit width

```bash
ifx \
  -O3 \
  -xCORE-AVX512 \
  -mprefer-vector-width=512 \
  cpu-simd-study/benchmark_native_sgemm.f90 \
  -o bench_nativesgemm_ifx_avx512_forced
```

`objdump` and VTune confirmed ZMM instructions in this binary.

---

## 6. VTune collection workflow

Two analyses were collected for every configuration:

```bash
vtune -collect hotspots \
  -result-dir <named-hotspots-result> \
  ./<benchmark-binary>
```

```bash
vtune -collect uarch-exploration \
  -result-dir <named-uarch-result> \
  ./<benchmark-binary>
```

Results were organized by compiler and SIMD target, for example:

```text
cpu-simd-study/native_sgemm_profiles/
├── gfortran/
│   ├── scalar/
│   ├── avx2/
│   └── avx512/
└── ifx/
    ├── scalar/
    ├── avx2/
    ├── avx512/
    └── avx512-forced/
```

Hotspots and Microarchitecture Exploration are separate runs. Their elapsed times must not be compared as if they were repeated measurements from one timing harness, because VTune analyses impose different collection overheads. The standalone `/usr/bin/time` measurements remain the cleaner wall-clock baseline.

---

## 7. Assembly verification

### Scalar code

Scalar builds showed instructions such as:

```text
movss
mulss
addss
```

### AVX2 code

AVX2 builds showed:

```text
vbroadcastss
vmovups
vmulps
vaddps
vfmadd231ps
YMM registers
```

### Forced AVX-512 code

Full-width builds showed:

```text
vmovups with ZMM registers
vfmadd213ps / vfmadd231ps with ZMM registers
ZMM16-ZMM21 and other extended registers
```

The forced-ifx assembly contained a long, aggressively unrolled loop body with many broadcasts, loads, stores, and independent ZMM/FMA accumulator chains.

A long static assembly listing does not imply a higher dynamic instruction count. The forced-ZMM ifx build had a larger static loop body but retired fewer instructions dynamically because each iteration performed more work.

---

## 8. GFortran results

### 8.1 Scalar

Representative standalone runtime:

```text
~26.1 s
```

Representative VTune observations:

```text
Retiring:                    ~36.2%
Vector Capacity Usage:        ~6.2%
```

The scalar build used XMM/scalar arithmetic.

### 8.2 AVX2

Representative standalone/VTune runtime:

```text
~8.2 s
```

Representative VTune observations:

```text
Retiring / microarchitecture utilization: ~42%
Vector Capacity Usage:                     50%
Memory Bound:                              ~11.4%
```

This was a substantial SIMD gain over scalar:

```text
26.1 s -> 8.2 s
approximately 3.2x faster
```

The AVX2 hotspot used YMM registers and packed FP32 arithmetic.

### 8.3 Forced AVX-512

Representative runtime:

```text
~6.1 s
```

Observed:

```text
ZMM registers confirmed
Vector Capacity Usage: 100%
```

Approximate speedups:

```text
Scalar -> AVX2:   ~3.2x
AVX2 -> AVX-512:  ~1.3x
Scalar -> AVX-512: ~4.3x
```

The SGEMM workload therefore behaved differently from the earlier reduction kernel. Here, SIMD width produced clear benefits, and GFortran reached full reported vector capacity with the forced-ZMM build.

---

## 9. ifx results

### 9.1 Scalar

Representative runtime:

```text
~33.9 s
```

Representative counters:

```text
Clockticks:           ~92.1 B
Instructions Retired: ~202.2 B
CPI:                  ~0.455
```

For this SGEMM kernel, GFortran scalar was faster than ifx scalar. This reversed the earlier inner-product result, where ifx scalar had outperformed GFortran SIMD variants.

Comparison with GFortran scalar:

```text
GFortran scalar:
  Clockticks:           ~73.5-73.9 B
  Instructions Retired: ~155.9 B
  CPI:                  ~0.47
  Runtime:              ~26.1 s

ifx scalar:
  Clockticks:           ~92.1 B
  Instructions Retired: ~202.2 B
  CPI:                  ~0.455
  Runtime:              ~33.9 s
```

Approximate IPC:

```text
GFortran scalar IPC ≈ 155.9 / 73.9 ≈ 2.11
ifx scalar IPC      ≈ 202.2 / 92.1 ≈ 2.20
```

ifx retired slightly more instructions per cycle, but executed many more instructions overall. The increased instruction count and clockticks outweighed the modest IPC advantage.

The precise source of additional scalar instructions remains an open question. The assembly suggested different unrolling, address-generation, load/store, and update strategies, but no claim should be made without a more controlled instruction-class breakdown.

### 9.2 AVX2

Representative elapsed time:

```text
~3.49 s
```

Representative counters:

```text
Clockticks:           ~9.08 B
Instructions Retired: ~20.58 B
CPI:                  ~0.441
Vector Capacity Usage: ~50%
```

The hotspot resembled a compiler-generated GEMM microkernel:

- repeated YMM FMAs;
- many independent accumulators;
- aggressive unrolling;
- broadcasted scalar elements;
- register-resident partial sums.

Compared with ifx scalar, AVX2 reduced both clockticks and instructions by approximately an order of magnitude.

### 9.3 AVX-512 target, default width selection

Representative elapsed time:

```text
~2.80 s
```

Representative counters:

```text
Clockticks:           ~6.73 B
Instructions Retired: ~16.56 B
CPI:                  ~0.407
Core Bound:           ~15.4%
Vector Capacity Usage: ~50%
```

No ZMM instructions were found in the executable hotspot. The compiler used YMM-width arithmetic plus the expanded AVX-512 register set/AVX-512VL features.

Compared with ifx AVX2:

```text
Clockticks:           9.08 B -> 6.73 B
Instructions Retired: 20.58 B -> 16.56 B
CPI:                  0.441 -> 0.407
```

The AVX-512-targeted build improved code generation even without choosing full-width ZMM arithmetic.

### 9.4 Forced ifx ZMM width

The forced build used:

```bash
-xCORE-AVX512 -mprefer-vector-width=512
```

ZMM instructions were confirmed by both `objdump` and VTune assembly.

#### Microarchitecture Exploration result

```text
Elapsed Time:          2.240 s
Clockticks:            6.3696 B
Instructions Retired:  8.7912 B
CPI:                   0.725
Retiring:              33.0% of pipeline slots
Front-End Bound:        2.0%
Bad Speculation:        1.1%
Back-End Bound:        63.9%
Memory Bound:          21.7%
Core Bound:            42.2%
Vector Capacity Usage: 100.0%
Average CPU Frequency:  3.1 GHz
```

Memory hierarchy details:

```text
L1 Bound:               10.1% of clockticks
DTLB Overhead:           7.1%
Load STLB Hit:           6.2%
Load STLB Miss:          0.9%
Loads Blocked by Store Forwarding: 2.6%
L1 Latency Dependency:  18.7%
Split Loads:             4.0%
4K Aliasing:             8.5%
FB Full:                 1.1%
L2 Bound:                7.0%
L2 Hit Latency:         63.5%
L3 Bound:                0.5%
DRAM Bound:              3.2%
Store Bound:             0.5%
```

Core-side details:

```text
Port Utilization:       40.8% of clockticks
Cycles of 1 Port Utilized: 35.1%
Cycles of 2 Ports Utilized: 17.1%
Cycles of 3+ Ports Utilized: 19.8%
```

#### Hotspots result

```text
Elapsed Time: 5.733 s
CPU Time:     2.319 s
MAIN__:       2.229 s, 96.1% of CPU time
Parallelism:  5.1%
Microarchitecture Usage: 32.7%
```

The difference between Hotspots elapsed time and Microarchitecture Exploration elapsed time is collection overhead and run-to-run variation, not a contradiction in the kernel result.

---

## 10. Default ifx width versus forced ZMM

### Default ifx AVX-512 target

```text
Arithmetic width:       YMM / 256-bit
Vector Capacity Usage:  ~50%
Clockticks:             ~6.73 B
Instructions Retired:   ~16.56 B
CPI:                    ~0.407
Core Bound:             ~15.4%
Elapsed:                ~2.80 s
```

### Forced ifx 512-bit width

```text
Arithmetic width:       ZMM / 512-bit
Vector Capacity Usage:  100%
Clockticks:             ~6.37 B
Instructions Retired:   ~8.79 B
CPI:                    ~0.725
Core Bound:             42.2%
Memory Bound:           21.7%
Elapsed in UE run:       2.240 s
```

Interpretation:

- Forcing ZMM nearly halved the dynamic instruction count.
- Clockticks declined only modestly compared with the instruction-count reduction.
- CPI increased substantially.
- Core Bound and Memory Bound pressure increased.
- Vector Capacity Usage reached 100%.
- Full-width ZMM still improved observed execution time, but the improvement was far smaller than the reduction in dynamic instructions.

The wider kernel performs more work per instruction but creates heavier execution and cache pressure. This is a textbook example of why instruction count alone and Vector Capacity Usage alone cannot predict performance.

---

## 11. GFortran ZMM versus ifx compiler-selected YMM

An especially important result was:

```text
GFortran forced ZMM:
  Vector Capacity Usage: 100%
  Runtime:               ~6.1 s
  CPI:                   ~1.08
  Core Bound:            ~35.9%
  Clockticks:            ~16.6 B
  Instructions Retired:  ~15.4 B

ifx AVX-512 target, YMM arithmetic:
  Vector Capacity Usage: ~50%
  Runtime:               ~2.8 s
  CPI:                   ~0.407
  Core Bound:            ~15.4%
  Clockticks:            ~6.73 B
  Instructions Retired:  ~16.56 B
```

The ifx build was much faster despite lower Vector Capacity Usage and no full-width ZMM arithmetic in the hotspot.

This demonstrates:

```text
Maximum vector width != maximum performance
Maximum vector-capacity usage != maximum performance
```

The better-performing implementation had:

- much lower CPI;
- far fewer clockticks;
- roughly half the Core Bound pressure;
- better scheduling and instruction-level parallelism;
- a more efficient compiler-generated microkernel.

---

## 12. Comparison with the earlier inner-product study

### Inner-product reduction kernel

Key result:

```text
Compiler optimization strategy mattered more than SIMD width.
```

Observed:

- ifx scalar outperformed GFortran SIMD variants;
- ifx AVX2 approximately matched ifx AVX-512;
- GFortran SIMD had low Vector Capacity Usage;
- reduction dependencies limited wider-vector benefits.

### Native SGEMM kernel

Key result:

```text
SIMD width produced clear benefits, but compiler strategy still mattered.
```

Observed:

- GFortran improved coherently from scalar to AVX2 to forced AVX-512;
- GFortran AVX-512 reached 100% Vector Capacity Usage;
- ifx AVX2 and AVX-512 substantially outperformed GFortran;
- ifx's default AVX-512 target preferred YMM arithmetic;
- forced ifx ZMM reduced instruction count and improved elapsed time, but sharply increased CPI, Core Bound, and memory/cache pressure.

Together, the two studies show that kernel structure changes the value of SIMD width:

```text
Reduction-heavy dot product:
  dependency management and compiler strategy dominate

SGEMM:
  arithmetic intensity and independent work allow wider SIMD to pay off
```

---

## 13. Major lessons

### 13.1 Compiler flags permit or prefer behavior; they do not always mandate it

`-xCORE-AVX512` enabled the AVX-512 target but did not compel ifx to use ZMM arithmetic. The compiler selected YMM plus AVX-512VL/expanded registers until `-mprefer-vector-width=512` was added.

### 13.2 AVX-512 can help when the workload exposes enough independent arithmetic

Unlike the reduction kernel, native SGEMM gave the compiler enough FMA work and independent accumulations to benefit from AVX2 and AVX-512.

### 13.3 100% Vector Capacity Usage is not a universal performance score

GFortran ZMM reached 100% but was substantially slower than ifx's default YMM implementation at approximately 50%.

### 13.4 Static code size and dynamic instruction count are different

The forced-ZMM ifx loop body was long and heavily unrolled, but retired fewer dynamic instructions.

### 13.5 Lower CPI can outweigh vector-width differences

ifx's compiler-selected YMM build had a much lower CPI and Core Bound percentage than GFortran's forced-ZMM build, producing much better runtime despite lower reported vector capacity.

### 13.6 Working-set size changed the memory story

The 3 MiB nominal matrix working set exceeded private L1/L2 capacity. The forced-ZMM profile saw meaningful L1/L2 and TLB pressure. However, DRAM Bound remained only 3.2%, so the experiment was not primarily a main-memory bandwidth benchmark.

### 13.7 Compiler winners are kernel-dependent

- ifx scalar dominated the earlier reduction study.
- GFortran scalar beat ifx scalar in the native SGEMM study.
- ifx SIMD then dominated GFortran SIMD in SGEMM.

No compiler should be labelled universally superior from one kernel.

---

## 14. Current interpretation

The SGEMM study supports the following provisional interpretation:

> GFortran benefits coherently from wider SIMD on a native SGEMM workload and reaches full reported vector capacity with forced AVX-512. Intel ifx generates substantially faster SIMD kernels through stronger unrolling, accumulator management, FMA scheduling, and instruction-level parallelism. Its default AVX-512 target favors 256-bit arithmetic with the expanded AVX-512 register architecture, while an explicit 512-bit width preference produces a genuine ZMM microkernel. The forced-ZMM kernel retires far fewer instructions and runs faster, but incurs higher CPI, Core Bound pressure, and L1/L2-related stalls. Therefore, vector width, vector-capacity utilization, and instruction count must be interpreted together with CPI, clockticks, and Top-Down microarchitecture metrics.

---

## 15. Limitations

- Single Tiger Lake CPU only.
- Single compiler versions per family.
- Single-threaded benchmark only.
- Native naive SGEMM, not a blocked or packed production BLAS implementation.
- No bare-metal VTune Memory Access analysis under WSL/virtualization.
- VTune Hotspots and Microarchitecture Exploration use separate runs with different overheads.
- Random initialization means output values are not expected to match across separately compiled executions unless the same seed/data are fixed.
- No statistical multi-run timing distribution has yet been reported for every configuration.
- No power or energy measurement.
- No frequency trace across the complete run.
- No comparison yet with OpenBLAS, oneMKL, AOCL-BLAS, or BLIS.

---

## 16. Evidence to retain

For each configuration, preserve:

- exact compiler version;
- complete compile command;
- executable hash;
- standalone timing output;
- VTune Hotspots result directory;
- VTune Microarchitecture Exploration result directory;
- emitted assembly or `objdump` extract;
- vectorization report;
- CPU model and exposed ISA flags;
- benchmark source revision;
- matrix dimension and iteration count.

---

## 17. Next actions

1. Export all confirmed metrics into one CSV.
2. Run repeated standalone timings for median, minimum, maximum, and variation.
3. Generate plots for:
   - elapsed time;
   - clockticks;
   - instructions retired;
   - CPI;
   - Retiring;
   - Core Bound and Memory Bound;
   - Vector Capacity Usage.
4. Keep default ifx AVX-512 and forced-width ifx AVX-512 as separate configurations.
5. Avoid adding threading until the single-core interpretation is complete.
6. Use the paired inner-product and SGEMM investigations as the core of a technical report.

---

## 18. Report-level takeaway

The study began with a simple expectation:

```text
Scalar < AVX2 < AVX-512
```

The actual result was richer:

```text
Inner-product reduction:
  compiler strategy dominated SIMD width

Native SGEMM:
  wider SIMD helped substantially
  but compiler-generated scheduling still dominated

GFortran ZMM:
  100% Vector Capacity Usage
  slower overall

ifx YMM under AVX-512 target:
  ~50% Vector Capacity Usage
  much lower CPI and Core Bound
  much faster overall

ifx forced ZMM:
  100% Vector Capacity Usage
  fewer dynamic instructions
  slightly fewer clockticks
  higher CPI and cache/core pressure
```

The central lesson is:

```text
Performance is not determined by SIMD width alone.
```

For these kernels, performance emerged from the interaction of:

- workload structure;
- arithmetic intensity;
- vector width;
- unrolling;
- independent accumulators;
- FMA scheduling;
- register use;
- cache and TLB behavior;
- instruction count;
- CPI;
- Core Bound and Memory Bound pressure;
- compiler optimization strategy.