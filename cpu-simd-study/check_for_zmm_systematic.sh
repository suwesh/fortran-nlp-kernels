# GFortran forced ZMM
objdump -d -M intel cpu-simd-study/builds/bench_ip_gfortran_avx512_zmm | grep -c zmm
objdump -d -M intel cpu-simd-study/builds/bench_nativesgemm_gfortran_avx512_zmm | grep -c zmm

# ifx AVX-512 automatic width
objdump -d -M intel cpu-simd-study/builds/bench_ip_ifx_avx512_auto | grep -c zmm
objdump -d -M intel cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_auto | grep -c zmm

# ifx forced ZMM
objdump -d -M intel cpu-simd-study/builds/bench_ip_ifx_avx512_zmm | grep -c zmm
objdump -d -M intel cpu-simd-study/builds/bench_nativesgemm_ifx_avx512_zmm | grep -c zmm