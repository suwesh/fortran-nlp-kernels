# compile
ifx -O3 -xCORE-AVX512 src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o bench_ip_ifx_avx512
---
gfortran -O3 -march=tigerlake -fopt-info-vec-all src/native/cpu/vector_search/topk_core.f90 src/native/cpu/vector_search/vector_search_core.f90 cpu-simd-study/benchmark_ip.f90 -o bench_ip_gfortran_avx512_force

# VTune profile
vtune --collect hotspots ./bench_ip_gfortran_avx512_force
---
vtune --collect uarch-exploration ./bench_ip_gfortran_avx512_force