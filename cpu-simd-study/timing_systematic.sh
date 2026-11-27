#!/usr/bin/env bash
set -e
mkdir -p cpu-simd-study/measurements/raw

CSV="cpu-simd-study/measurements/raw/timing_results.csv"
TMP="cpu-simd-study/measurements/raw/current_time.txt"

echo "kernel,compiler,configuration,run,elapsed_s,user_s,system_s" > "$CSV"

measure_binary () {
    KERNEL="$1"
    COMPILER="$2"
    CONFIGURATION="$3"
    BINARY="$4"

    echo "Warming up: $KERNEL $COMPILER $CONFIGURATION"
    taskset -c 0 "$BINARY" > /dev/null
    taskset -c 0 "$BINARY" > /dev/null

    echo "Measuring: $KERNEL $COMPILER $CONFIGURATION"
    for RUN in 1 2 3 4 5 6 7
    do
        /usr/bin/time \
          -f "%e,%U,%S" \
          -o "$TMP" \
          taskset -c 0 "$BINARY" \
          > /dev/null
        echo "$KERNEL,$COMPILER,$CONFIGURATION,$RUN,$(cat "$TMP")" \
          >> "$CSV"
    done
}

measure_binary \
  inner_product \
  gfortran \
  scalar \
  ./cpu-simd-study/builds/bench_ip_gfortran_scalar

measure_binary \
  inner_product \
  gfortran \
  avx2 \
  ./cpu-simd-study/builds/bench_ip_gfortran_avx2

measure_binary \
  inner_product \
  gfortran \
  avx512-zmm \
  ./cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm

measure_binary \
  inner_product \
  ifx \
  scalar \
  ./cpu-simd-study/builds/bench_ip_ifx_scalar

measure_binary \
  inner_product \
  ifx \
  avx2 \
  ./cpu-simd-study/builds/bench_ip_ifx_avx2

measure_binary \
  inner_product \
  ifx \
  avx512-auto \
  ./cpu-simd-study/builds/bench_ip_ifx_avx512_auto

measure_binary \
  inner_product \
  ifx \
  avx512-zmm \
  ./cpu-simd-study/builds/bench_ip_ifx_avx512_zmm

measure_binary \
  native_sgemm \
  gfortran \
  scalar \
  ./cpu-simd-study/builds/bench_nativesgemm_gfortran_scalar

measure_binary \
  native_sgemm \
  gfortran \
  avx2 \
  ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx2

measure_binary \
  native_sgemm \
  gfortran \
  avx512-zmm \
  ./cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm

measure_binary \
  native_sgemm \
  ifx \
  scalar \
  ./cpu-simd-study/builds/bench_nativesgemm_ifx_scalar

measure_binary \
  native_sgemm \
  ifx \
  avx2 \
  ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx2

measure_binary \
  native_sgemm \
  ifx \
  avx512-auto \
  ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto

measure_binary \
  native_sgemm \
  ifx \
  avx512-zmm \
  ./cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm

rm -f "$TMP"

echo "Timing measurements written to:"
echo "$CSV"