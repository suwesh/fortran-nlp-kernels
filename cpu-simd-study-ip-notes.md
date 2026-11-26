# CPU SIMD Study Notes (2026-11-25)

## Objective
Investigate how compiler choice, SIMD width (Scalar, AVX, AVX2, AVX-512), and Intel VTune microarchitecture metrics affect a simple cache-resident Fortran inner-product kernel from `fortran-nlp-kernels`.

## Kernel Under Study
Source: `fnlp_vector_search_core::inner_product`

```fortran
similarity = similarity + \
    query_vector(dimension_index) * \
    database_vector(dimension_index)
```

This is a reduction kernel (dot product).

## Benchmark Setup
- Project: `fortran-nlp-kernels`
- Branch: `cpu-simd-study`
- CPU: Intel Core i5-1135G7 (Tiger Lake)
- Profiling Tool: Intel VTune
- Kernel: `inner_product`
- Iterations: 100,000,000
- Same source used across compiler experiments.

---

# Key Headline Finding

The biggest surprise of the entire study was not AVX-512.

```text
ifx Scalar > GFortran AVX2
ifx Scalar > GFortran AVX512
```

For this kernel, compiler optimization strategy mattered more than SIMD width.

---

# Runtime Summary

```text
GFortran Scalar      ≈ 84 s
GFortran AVX         ≈ 72–110 s
GFortran AVX2        ≈ 72–110 s
GFortran AVX512      ≈ 95 s

ifx Scalar           ≈ 55 s
ifx AVX2             ≈ 4.4 s
ifx AVX512           ≈ 4.3 s
```

Observation:

```text
ifx AVX2 ≈ ifx AVX512
```

Once ifx correctly vectorized and scheduled the reduction, AVX-512 added very little beyond AVX2.

---

# GFortran Investigation

## Scalar

```text
Retiring:              27.4%
Memory Bound:          16.5%
Core Bound:            53.9%
Vector Capacity:        6.2%
```

Assembly:

```text
movss
mulss
addss
```

## AVX / AVX2

Compiler report:

```text
loop vectorized using 32 byte vectors
```

Assembly contained:

```text
vmulps
vaddps
ymm
```

VTune:

```text
Memory Bound ≈ 17%
Core Bound ≈ 64%
Vector Capacity ≈ 11%
```

Important:

```text
GFortran DID vectorize the loop.
```

The issue was not absence of SIMD.

The issue was low SIMD utilization.

## AVX512

`-march=tigerlake` alone still emitted YMM registers.

AVX512 required:

```bash
-march=tigerlake
-mprefer-vector-width=512
```

Observed:

```text
zmm
vmovups
vmulps
```

AVX512 instructions were genuinely emitted and executed.

However:

```text
AVX512 ≈ AVX2
```

for performance and VTune metrics.

---

# Intel ifx Investigation

## Scalar

Runtime:

```text
≈55 s
```

VTune:

```text
Retiring:      33.0%
Memory Bound:  10.7%
Core Bound:    42.9%
CPI:           0.56
```

Already faster than every tested GFortran build.

## AVX2

Runtime:

```text
≈4.4 s
```

VTune:

```text
Retiring:           36.2%
Memory Bound:        7.6%
Core Bound:         17.9%
Vector Capacity:    49.7%
CPI:                0.51
```

Observed assembly:

```text
vmovups
vfmadd231ps
ymm0
ymm1
ymm2
ymm3
```

Notable behavior:

- Multiple independent accumulators
- Heavy FMA usage
- Much better execution-unit utilization

## AVX512

Runtime:

```text
≈4.3 s
```

VTune:

```text
Retiring:           34.3%
Memory Bound:        4.7%
Core Bound:         16.6%
Vector Capacity:    49.6%
CPI:                0.51
```

Result:

```text
ifx AVX2 ≈ ifx AVX512
```

---

# Assembly Findings

## GFortran

SIMD instructions existed but vector-capacity utilization remained roughly:

```text
11%
```

## ifx

Observed patterns such as:

```asm
vfmadd231ps (...) %ymm4, %ymm3
vfmadd231ps (...) %ymm5, %ymm0
vfmadd231ps (...) %ymm6, %ymm1
vfmadd231ps (...) %ymm7, %ymm2
```

This suggests multiple independent accumulator chains rather than a single reduction dependency chain.

---

# VTune Findings

## Vector Capacity Usage

```text
GFortran Scalar     6.2%
GFortran SIMD      ~11%

ifx AVX2           49.7%
ifx AVX512         49.6%
```

This was the single most informative metric observed.

## Memory Bound

Across all runs:

```text
~5% to ~17%
```

The workload is largely cache-resident.

The benchmark repeatedly accesses:

```text
768 FP32 values
768 FP32 values
```

which comfortably fit inside cache.

The workload is primarily:

```text
Compute bound
```

rather than DRAM-bandwidth bound.

---

# Major Lessons Learned

## 1. SIMD Width != Performance

Expected:

```text
Scalar < AVX < AVX2 < AVX512
```

Observed:

```text
ifx AVX2 ≈ ifx AVX512
```

## 2. Compiler Strategy Matters More Than SIMD Width

Observed:

```text
ifx Scalar > GFortran SIMD
```

and

```text
ifx AVX2 ≈ ifx AVX512
```

## 3. Reduction Kernels Are Special

```fortran
similarity = similarity + x(i) * y(i)
```

requires reducing many SIMD lanes into one scalar result.

Reduction behavior becomes more important than simply widening vectors.

## 4. AVX512 Was Verified

AVX512 support existed on the CPU.

`zmm` instructions were observed after forcing 512-bit vectors.

AVX512 executed successfully, but did not materially outperform AVX2.

## 5. VTune Was Essential

The investigation evolved from:

```text
Did the compiler vectorize?
```

to:

```text
Why does ifx utilize the vector hardware so much better?
```

This became a compiler and microarchitecture study rather than a SIMD-width study.

---

# Final Conclusion

The original goal was to learn AVX and AVX512.

The most important lesson turned out to be:

```text
Compiler optimization strategy
can matter more than SIMD width.
```

A particularly surprising result was that:

```text
ifx Scalar > GFortran AVX2/AVX512
```

while:

```text
ifx AVX2 ≈ ifx AVX512
```

showing that effective vectorization, instruction scheduling, accumulator management, and execution-unit utilization were more important for this kernel than simply increasing SIMD width.