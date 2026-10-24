# Vector Single-Query CPU Benchmark

Values are medians across four production-derived queries at k=10. Timings include native scoring, exact O(Nk) top-k, binding, and output creation.

| Metric | Vectors | Serial median (ms) | OpenMP-6 median (ms) | SGEMV-6 median (ms) | OpenMP-6 speedup | SGEMV-6 speedup | Best backend | Estimated memory (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IP | 475 | 0.346 | 0.068 | 0.030 | 5.082 | 11.633 | OpenBLAS SGEMV-6 | 1.859 |
| IP | 950 | 0.694 | 0.133 | 0.056 | 5.206 | 12.342 | OpenBLAS SGEMV-6 | 3.715 |
| IP | 2,375 | 1.764 | 0.329 | 0.149 | 5.361 | 11.880 | OpenBLAS SGEMV-6 | 9.281 |
| IP | 4,750 | 3.623 | 0.688 | 0.531 | 5.271 | 6.841 | OpenBLAS SGEMV-6 | 18.559 |
| IP | 11,875 | 9.113 | 1.706 | 1.566 | 5.343 | 5.799 | OpenBLAS SGEMV-6 | 46.391 |
| IP | 23,750 | 18.195 | 3.395 | 3.244 | 5.359 | 5.617 | OpenBLAS SGEMV-6 | 92.777 |
| IP | 47,500 | 36.373 | 6.850 | 6.448 | 5.301 | 5.642 | OpenBLAS SGEMV-6 | 185.551 |
| IP | 118,750 | 90.693 | 16.706 | 16.869 | 5.429 | 5.363 | Fortran OpenMP-6 | 463.871 |
| L2 | 475 | 0.346 | 0.066 | 0.372 | 5.254 | 0.931 | Fortran OpenMP-6 | 1.859 |
| L2 | 950 | 0.695 | 0.131 | 0.740 | 5.319 | 0.940 | Fortran OpenMP-6 | 3.715 |
| L2 | 2,375 | 1.753 | 0.324 | 1.874 | 5.423 | 0.937 | Fortran OpenMP-6 | 9.281 |
| L2 | 4,750 | 3.648 | 0.697 | 4.125 | 5.233 | 0.885 | Fortran OpenMP-6 | 18.559 |
| L2 | 11,875 | 9.129 | 1.695 | 10.545 | 5.386 | 0.866 | Fortran OpenMP-6 | 46.391 |
| L2 | 23,750 | 18.261 | 3.424 | 21.017 | 5.363 | 0.871 | Fortran OpenMP-6 | 92.777 |
| L2 | 47,500 | 36.392 | 6.798 | 42.083 | 5.354 | 0.865 | Fortran OpenMP-6 | 185.551 |
| L2 | 118,750 | 91.286 | 16.842 | 105.246 | 5.427 | 0.869 | Fortran OpenMP-6 | 463.871 |
