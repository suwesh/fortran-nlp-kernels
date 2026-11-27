#!/usr/bin/env bash
set -e

echo "Creating compiler module directories"
mkdir -p cpu-simd-study/build-modules/gfortran
mkdir -p cpu-simd-study/build-modules/ifx
mkdir -p cpu-simd-study/reports
mkdir -p cpu-simd-study/builds

echo "Removing stale compiler module files"
rm -f fnlp_topk_core.mod
rm -f fnlp_vector_search_core.mod
rm -f cpu-simd-study/build-modules/gfortran/*.mod
rm -f cpu-simd-study/build-modules/ifx/*.mod

echo "=========================================="
echo "Compiling inner-product configurations"
echo "=========================================="

echo "Compiling inner product with Gfortran Scalar"
gfortran -O3 -fno-tree-vectorize -J cpu-simd-study/build-modules/gfortran src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o cpu-simd-study/builds/bench_ip_gfortran_scalar
echo "Compiling inner product with Gfortran AVX2"
gfortran -O3 -mavx2 -mfma -J cpu-simd-study/build-modules/gfortran src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o cpu-simd-study/builds/bench_ip_gfortran_avx2
echo "Compiling inner product with Gfortran forced AVX-512 ZMM width"
gfortran -O3 -march=tigerlake -mprefer-vector-width=512 -fopt-info-vec-all=cpu-simd-study/reports/ip-gfortran-avx512-zmm.vec -J cpu-simd-study/build-modules/gfortran src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm

echo "Compiling inner product with IFX Scalar"
ifx -O3 -no-vec -module cpu-simd-study/build-modules/ifx src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o cpu-simd-study/builds/bench_ip_ifx_scalar
echo "Compiling inner product with IFX AVX2"
ifx -O3 -xCORE-AVX2 -module cpu-simd-study/build-modules/ifx src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o cpu-simd-study/builds/bench_ip_ifx_avx2
echo "Compiling inner product with IFX AVX-512 auto width selection"
ifx -O3 -xCORE-AVX512 -module cpu-simd-study/build-modules/ifx src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o cpu-simd-study/builds/bench_ip_ifx_avx512_auto
echo "Compiling inner product with IFX forced AVX-512 ZMM width"
ifx -O3 -xCORE-AVX512 -mprefer-vector-width=512 -qopt-report=3 -qopt-report-phase=vec -qopt-report-file=cpu-simd-study/reports/ip-ifx-avx512-zmm.optrpt -module cpu-simd-study/build-modules/ifx src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o cpu-simd-study/builds/bench_ip_ifx_avx512_zmm

echo "=========================================="
echo "Compiling native GEMM configurations"
echo "=========================================="

echo "Compiling native GEMM with Gfortran Scalar"
gfortran -O3 -fno-tree-vectorize -J cpu-simd-study/build-modules/gfortran cpu-simd-study/benchmark_nativesgemm.f90 -o cpu-simd-study/builds/bench_nativesgemm_gfortran_scalar
echo "Compiling native GEMM with Gfortran AVX2"
gfortran -O3 -mavx2 -mfma -J cpu-simd-study/build-modules/gfortran cpu-simd-study/benchmark_nativesgemm.f90 -o cpu-simd-study/builds/bench_nativesgemm_gfortran_avx2
echo "Compiling native GEMM with Gfortran forced AVX-512 ZMM width"
gfortran -O3 -march=tigerlake -mprefer-vector-width=512 -fopt-info-vec-all=cpu-simd-study/reports/nativesgemm-gfortran-avx512-zmm.vec -J cpu-simd-study/build-modules/gfortran cpu-simd-study/benchmark_nativesgemm.f90 -o cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm

echo "Compiling native GEMM with IFX Scalar"
ifx -O3 -no-vec -module cpu-simd-study/build-modules/ifx cpu-simd-study/benchmark_nativesgemm.f90 -o cpu-simd-study/builds/bench_nativesgemm_ifx_scalar
echo "Compiling native GEMM with IFX AVX2"
ifx -O3 -xCORE-AVX2 -module cpu-simd-study/build-modules/ifx cpu-simd-study/benchmark_nativesgemm.f90 -o cpu-simd-study/builds/bench_nativesgemm_ifx_avx2
echo "Compiling native GEMM with IFX AVX-512 auto width selection"
ifx -O3 -xCORE-AVX512 -module cpu-simd-study/build-modules/ifx cpu-simd-study/benchmark_nativesgemm.f90 -o cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto
echo "Compiling native GEMM with IFX forced AVX-512 ZMM width"
ifx -O3 -xCORE-AVX512 -mprefer-vector-width=512 -qopt-report=3 -qopt-report-phase=vec -qopt-report-file=cpu-simd-study/reports/nativesgemm-ifx-avx512-zmm.optrpt -module cpu-simd-study/build-modules/ifx cpu-simd-study/benchmark_nativesgemm.f90 -o cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm

echo "=========================================="
echo "Compilation successful, proceed for profiling with 1 unmeasured warup each"
echo "=========================================="
