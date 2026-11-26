#!/usr/bin/env bash

echo "Profiling native GEMM kernel with GFortran and fno-tree-vectorize(scalar)"
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran/scalar/gfortran-scalar-hs ./bench_nativesgemm_gfortran_scalar
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/gfortran/scalar/gfortran-scalar-ue ./bench_nativesgemm_gfortran_scalar

echo "Profiling native GEMM kernel with GFortran and AVX2 vectorization"
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran/avx2/gfortran-avx2-hs ./bench_nativesgemm_gfortran_avx2
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/gfortran/avx2/gfortran-avx2-ue ./bench_nativesgemm_gfortran_avx2

echo "Profiling native GEMM kernel with GFortran and AVX512 vectorization"
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran/avx512/gfortran-avx512-hs ./bench_nativesgemm_gfortran_avx512
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/gfortran/avx512/gfortran-avx512-ue ./bench_nativesgemm_gfortran_avx512

echo "Profiling native GEMM kernel with IFX and no-vec(scalar)"
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx/scalar/ifx-scalar-hs ./bench_nativesgemm_ifx_scalar
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/ifx/scalar/ifx-scalar-ue ./bench_nativesgemm_ifx_scalar

echo "Profiling native GEMM kernel with IFX and AVX2 vectorization"
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx/avx2/ifx-avx2-hs ./bench_nativesgemm_ifx_avx2
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/ifx/avx2/ifx-avx2-ue ./bench_nativesgemm_ifx_avx2

echo "Profiling native GEMM kernel with IFX and AVX512 vectorization"
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx/avx512/ifx-avx512-hs ./bench_nativesgemm_ifx_avx512
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/ifx/avx512/ifx-avx512-ue ./bench_nativesgemm_ifx_avx512