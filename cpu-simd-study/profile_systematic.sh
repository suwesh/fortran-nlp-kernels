#!/usr/bin/env bash
set -e

echo "Creating inner-product profile directories"
mkdir -p cpu-simd-study/inner_product_profiles/gfortran-scalar
mkdir -p cpu-simd-study/inner_product_profiles/gfortran-avx2
mkdir -p cpu-simd-study/inner_product_profiles/gfortran-avx512-zmm
mkdir -p cpu-simd-study/inner_product_profiles/ifx-scalar
mkdir -p cpu-simd-study/inner_product_profiles/ifx-avx2
mkdir -p cpu-simd-study/inner_product_profiles/ifx-avx512-auto
mkdir -p cpu-simd-study/inner_product_profiles/ifx-avx512-zmm

echo "Creating native GEMM profile directories"
mkdir -p cpu-simd-study/native_sgemm_profiles/gfortran-scalar
mkdir -p cpu-simd-study/native_sgemm_profiles/gfortran-avx2
mkdir -p cpu-simd-study/native_sgemm_profiles/gfortran-avx512-zmm
mkdir -p cpu-simd-study/native_sgemm_profiles/ifx-scalar
mkdir -p cpu-simd-study/native_sgemm_profiles/ifx-avx2
mkdir -p cpu-simd-study/native_sgemm_profiles/ifx-avx512-auto
mkdir -p cpu-simd-study/native_sgemm_profiles/ifx-avx512-zmm

echo "=========================================="
echo "Compilation successful, proceeding for profiling with 1 unmeasured warup each"
echo "=========================================="
echo "Profiling inner-product configurations"
echo "=========================================="

echo "Profiling inner product: GFortran scalar"
taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_scalar >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-scalar/ip-gfortran-scalar-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_scalar
taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_scalar >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/inner_product_profiles/gfortran-scalar/ip-gfortran-scalar-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_scalar

echo "Profiling inner product: GFortran AVX2"
taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx2 >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx2/ip-gfortran-avx2-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx2
taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx2 >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx2/ip-gfortran-avx2-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx2

echo "Profiling inner product: GFortran forced AVX-512 ZMM width"
taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx512-zmm/ip-gfortran-avx512-zmm-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm
taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx512-zmm/ip-gfortran-avx512-zmm-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm

echo "Profiling inner product: IFX scalar"
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_scalar >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-scalar/ip-ifx-scalar-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_scalar
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_scalar >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/inner_product_profiles/ifx-scalar/ip-ifx-scalar-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_scalar

echo "Profiling inner product: IFX AVX2"
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx2 >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx2/ip-ifx-avx2-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx2
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx2 >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/inner_product_profiles/ifx-avx2/ip-ifx-avx2-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx2

echo "Profiling inner product: IFX AVX-512 auto width selection"
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_auto >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-auto/ip-ifx-avx512-auto-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_auto
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_auto >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-auto/ip-ifx-avx512-auto-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_auto

echo "Profiling inner product: IFX AVX-512 forced ZMM width"
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_zmm >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-zmm/ip-ifx-avx512-zmm-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_zmm
taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_zmm >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-zmm/ip-ifx-avx512-zmm-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_ip_ifx_avx512_zmm

echo "=========================================="
echo "Profiling native SGEMM configurations"
echo "=========================================="

echo "Profiling native sgemm: GFortran scalar"
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_scalar >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-scalar/sgemm-gfortran-scalar-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_scalar
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_scalar >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-scalar/sgemm-gfortran-scalar-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_scalar

echo "Profiling native SGEMM: GFortran AVX2"
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx2 >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx2/sgemm-gfortran-avx2-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx2
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx2 >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx2/sgemm-gfortran-avx2-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx2

echo "Profiling native SGEMM: GFortran forced AVX-512 ZMM width"
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx512-zmm/sgemm-gfortran-avx512-zmm-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx512-zmm/sgemm-gfortran-avx512-zmm-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm

echo "Profiling native SGEMM: IFX scalar"
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_scalar >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-scalar/sgemm-ifx-scalar-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_scalar
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_scalar >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/ifx-scalar/sgemm-ifx-scalar-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_scalar

echo "Profiling native SGEMM: IFX AVX2"
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx2 >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx2/sgemm-ifx-avx2-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx2
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx2 >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx2/sgemm-ifx-avx2-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx2

echo "Profiling native SGEMM: IFX AVX-512 auto width selection"
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-auto/sgemm-ifx-avx512-auto-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-auto/sgemm-ifx-avx512-auto-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto

echo "Profiling native SGEMM: IFX AVX-512 forced ZMM width"
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm >/dev/null
vtune -collect hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-zmm/sgemm-ifx-avx512-zmm-hs -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm
taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm >/dev/null
vtune -collect uarch-exploration -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-zmm/sgemm-ifx-avx512-zmm-ue -- taskset -c 0 ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm

echo "============================================================"
echo "Compilation and VTune profiling completed"
echo "============================================================"