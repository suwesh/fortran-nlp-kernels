#!/usr/bin/env bash

echo "Compiling native GEMM kernel with GFortran and fno-tree-vectorize(scalar)"
gfortran -O3 -fno-tree-vectorize cpu-simd-study/benchmark_nativesgemm.f90 -o bench_nativesgemm_gfortran_scalar

echo "Compiling native GEMM kernel with GFortran and AVX2 vectorization"
gfortran -O3 -mavx2 cpu-simd-study/benchmark_nativesgemm.f90 -o bench_nativesgemm_gfortran_avx2

echo "Compiling native GEMM kernel with GFortran and AVX512 vectorization"
gfortran -O3 -march=tigerlake -mprefer-vector-width=512 cpu-simd-study/benchmark_nativesgemm.f90 -o bench_nativesgemm_gfortran_avx512

echo "Compiling native GEMM kernel with IFX and fno-tree-vectorize(scalar)"
ifx -O3 -no-vec cpu-simd-study/benchmark_nativesgemm.f90 -o bench_nativesgemm_ifx_scalar

echo "Compiling native GEMM kernel with IFX and AVX2 vectorization"
ifx -O3 -xCORE-AVX2 cpu-simd-study/benchmark_nativesgemm.f90 -o bench_nativesgemm_ifx_avx2

echo "Compiling native GEMM kernel with IFX and AVX512 vectorization"
ifx -O3 -xCORE-AVX512 cpu-simd-study/benchmark_nativesgemm.f90 -o bench_nativesgemm_ifx_avx512
