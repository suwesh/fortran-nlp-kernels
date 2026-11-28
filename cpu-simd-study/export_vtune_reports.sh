#!/usr/bin/env bash
set -e

mkdir -p cpu-simd-study/measurements/vtune-csv

echo "Exporting inner-product VTune reports"

echo "Exporting GF-SCALAR inner product"
vtune -report summary -result-dir cpu-simd-study/inner_product_profiles/gfortran-scalar/ip-gfortran-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-scalar-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-scalar/ip-gfortran-scalar-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-scalar-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/inner_product_profiles/gfortran-scalar/ip-gfortran-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-scalar-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-scalar/ip-gfortran-scalar-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-scalar-hs-functions.csv

echo "Exporting GF-AVX2 inner product"
vtune -report summary -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx2/ip-gfortran-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx2-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx2/ip-gfortran-avx2-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx2-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx2/ip-gfortran-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx2-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx2/ip-gfortran-avx2-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx2-hs-functions.csv

echo "Exporting GF-AVX512-ZMM inner product"
vtune -report summary -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx512-zmm/ip-gfortran-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx512-zmm-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx512-zmm/ip-gfortran-avx512-zmm-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx512-zmm-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx512-zmm/ip-gfortran-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx512-zmm-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/gfortran-avx512-zmm/ip-gfortran-avx512-zmm-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-gfortran-avx512-zmm-hs-functions.csv

echo "Exporting IFX-SCALAR inner product"
vtune -report summary -result-dir cpu-simd-study/inner_product_profiles/ifx-scalar/ip-ifx-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-scalar-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-scalar/ip-ifx-scalar-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-scalar-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/inner_product_profiles/ifx-scalar/ip-ifx-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-scalar-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-scalar/ip-ifx-scalar-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-scalar-hs-functions.csv

echo "Exporting IFX-AVX2 inner product"
vtune -report summary -result-dir cpu-simd-study/inner_product_profiles/ifx-avx2/ip-ifx-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx2-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx2/ip-ifx-avx2-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx2-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/inner_product_profiles/ifx-avx2/ip-ifx-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx2-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx2/ip-ifx-avx2-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx2-hs-functions.csv

echo "Exporting IFX-AVX512-AUTO inner product"
vtune -report summary -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-auto/ip-ifx-avx512-auto-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-auto-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-auto/ip-ifx-avx512-auto-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-auto-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-auto/ip-ifx-avx512-auto-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-auto-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-auto/ip-ifx-avx512-auto-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-auto-hs-functions.csv

echo "Exporting IFX-AVX512-ZMM inner product"
vtune -report summary -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-zmm/ip-ifx-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-zmm-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-zmm/ip-ifx-avx512-zmm-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-zmm-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-zmm/ip-ifx-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-zmm-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/inner_product_profiles/ifx-avx512-zmm/ip-ifx-avx512-zmm-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/ip-ifx-avx512-zmm-hs-functions.csv


echo "Exporting native-SGEMM VTune reports"

echo "Exporting GF-SCALAR native SGEMM"
vtune -report summary -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-scalar/sgemm-gfortran-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-scalar-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-scalar/sgemm-gfortran-scalar-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-scalar-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-scalar/sgemm-gfortran-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-scalar-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-scalar/sgemm-gfortran-scalar-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-scalar-hs-functions.csv

echo "Exporting GF-AVX2 native SGEMM"
vtune -report summary -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx2/sgemm-gfortran-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx2-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx2/sgemm-gfortran-avx2-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx2-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx2/sgemm-gfortran-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx2-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx2/sgemm-gfortran-avx2-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx2-hs-functions.csv

echo "Exporting GF-AVX512-ZMM native SGEMM"
vtune -report summary -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx512-zmm/sgemm-gfortran-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx512-zmm-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx512-zmm/sgemm-gfortran-avx512-zmm-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx512-zmm-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx512-zmm/sgemm-gfortran-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx512-zmm-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/gfortran-avx512-zmm/sgemm-gfortran-avx512-zmm-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-gfortran-avx512-zmm-hs-functions.csv

echo "Exporting IFX-SCALAR native SGEMM"
vtune -report summary -result-dir cpu-simd-study/native_sgemm_profiles/ifx-scalar/sgemm-ifx-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-scalar-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-scalar/sgemm-ifx-scalar-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-scalar-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/native_sgemm_profiles/ifx-scalar/sgemm-ifx-scalar-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-scalar-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-scalar/sgemm-ifx-scalar-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-scalar-hs-functions.csv

echo "Exporting IFX-AVX2 native SGEMM"
vtune -report summary -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx2/sgemm-ifx-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx2-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx2/sgemm-ifx-avx2-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx2-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx2/sgemm-ifx-avx2-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx2-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx2/sgemm-ifx-avx2-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx2-hs-functions.csv

echo "Exporting IFX-AVX512-AUTO native SGEMM"
vtune -report summary -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-auto/sgemm-ifx-avx512-auto-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-auto-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-auto/sgemm-ifx-avx512-auto-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-auto-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-auto/sgemm-ifx-avx512-auto-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-auto-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-auto/sgemm-ifx-avx512-auto-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-auto-hs-functions.csv

echo "Exporting IFX-AVX512-ZMM native SGEMM"
vtune -report summary -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-zmm/sgemm-ifx-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-zmm-ue-summary.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-zmm/sgemm-ifx-avx512-zmm-ue -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-zmm-ue-functions.csv
vtune -report hw-events -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-zmm/sgemm-ifx-avx512-zmm-ue -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-zmm-ue-events.csv
vtune -report hotspots -result-dir cpu-simd-study/native_sgemm_profiles/ifx-avx512-zmm/sgemm-ifx-avx512-zmm-hs -group-by=function -format csv -report-output cpu-simd-study/measurements/vtune-csv/sgemm-ifx-avx512-zmm-hs-functions.csv

echo "VTune CSV exports completed"