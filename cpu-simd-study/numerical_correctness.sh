#!/usr/bin/env bash
set -e

mkdir -p cpu-simd-study/measurements/correctness

CSV="cpu-simd-study/measurements/correctness/correctness_outputs.csv"

echo "kernel,compiler,configuration,output" > "$CSV"

echo "Validating inner-product outputs in parallel"

echo "inner_product,gfortran,scalar,$(./cpu-simd-study/builds/bench_ip_gfortran_scalar | xargs)" >> "$CSV" &
echo "inner_product,gfortran,avx2,$(./cpu-simd-study/builds/bench_ip_gfortran_avx2 | xargs)" >> "$CSV" &
echo "inner_product,gfortran,avx512-zmm,$(./cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm | xargs)" >> "$CSV" &

echo "inner_product,ifx,scalar,$(./cpu-simd-study/builds/bench_ip_ifx_scalar | xargs)" >> "$CSV" &
echo "inner_product,ifx,avx2,$(./cpu-simd-study/builds/bench_ip_ifx_avx2 | xargs)" >> "$CSV" & 
echo "inner_product,ifx,avx512-auto,$(./cpu-simd-study/builds/bench_ip_ifx_avx512_auto | xargs)" >> "$CSV" &
echo "inner_product,ifx,avx512-zmm,$(./cpu-simd-study/builds/bench_ip_ifx_avx512_zmm | xargs)" >> "$CSV" &

wait
echo "Validating native-SGEMM checksums in parallel"

echo "native_sgemm,gfortran,scalar,$(./cpu-simd-study/builds/bench_nativesgemm_gfortran_scalar | xargs)" >> "$CSV" &
echo "native_sgemm,gfortran,avx2,$(./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx2 | xargs)" >> "$CSV" &
echo "native_sgemm,gfortran,avx512-zmm,$(./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm | xargs)" >> "$CSV" &

echo "native_sgemm,ifx,scalar,$(./cpu-simd-study/builds/bench_nativesgemm_ifx_scalar | xargs)" >> "$CSV" &
echo "native_sgemm,ifx,avx2,$(./cpu-simd-study/builds/bench_nativesgemm_ifx_avx2 | xargs)" >> "$CSV" &
echo "native_sgemm,ifx,avx512-auto,$(./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto | xargs)" >> "$CSV" &
echo "native_sgemm,ifx,avx512-zmm,$(./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm | xargs)" >> "$CSV" &

wait
echo "Correctness outputs written to:"
echo "$CSV"