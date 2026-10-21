#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "fortran_nlp_vector_search.h"

#define DIMENSION 3
#define VECTOR_COUNT 4
#define K 2
#define TOLERANCE 1.0e-6f


static int float_arrays_match(
    const float *actual,
    const float *expected,
    int count
)
{
    for (int index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > TOLERANCE) {
            return 0;
        }
    }

    return 1;
}


static int int64_arrays_match(
    const int64_t *actual,
    const int64_t *expected,
    int count
)
{
    for (int index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) {
            return 0;
        }
    }

    return 1;
}


int main(void)
{
    const float query_vector[DIMENSION] = {
        1.0f,
        2.0f,
        3.0f
    };

    /*
     * Fortran column-major database layout:
     *
     * vector 0 = [ 1, 2, 3]
     * vector 1 = [ 4, 5, 6]
     * vector 2 = [ 0, 0, 0]
     * vector 3 = [-1, 2, 1]
     *
     * Each complete vector is stored contiguously.
     */
    const float database_vectors[
        DIMENSION * VECTOR_COUNT
    ] = {
         1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         0.0f, 0.0f, 0.0f,
        -1.0f, 2.0f, 1.0f
    };

    const float expected_l2_values[K] = {
        0.0f,
        8.0f
    };

    const int64_t expected_l2_indices[K] = {
        0,
        3
    };

    const float expected_ip_values[K] = {
        32.0f,
        14.0f
    };

    const int64_t expected_ip_indices[K] = {
        1,
        0
    };


    float l2_serial_values[K];
    int64_t l2_serial_indices[K];

    float l2_openmp_values[K];
    int64_t l2_openmp_indices[K];

    float ip_serial_values[K];
    int64_t ip_serial_indices[K];

    float ip_openmp_values[K];
    int64_t ip_openmp_indices[K];


    fnlp_l2_search_serial(
        query_vector,
        database_vectors,
        DIMENSION,
        VECTOR_COUNT,
        K,
        l2_serial_values,
        l2_serial_indices
    );

    fnlp_l2_search_openmp(
        query_vector,
        database_vectors,
        DIMENSION,
        VECTOR_COUNT,
        K,
        l2_openmp_values,
        l2_openmp_indices
    );

    fnlp_ip_search_serial(
        query_vector,
        database_vectors,
        DIMENSION,
        VECTOR_COUNT,
        K,
        ip_serial_values,
        ip_serial_indices
    );

    fnlp_ip_search_openmp(
        query_vector,
        database_vectors,
        DIMENSION,
        VECTOR_COUNT,
        K,
        ip_openmp_values,
        ip_openmp_indices
    );


    if (!float_arrays_match(
        l2_serial_values,
        expected_l2_values,
        K
    )) {
        fprintf(stderr, "Serial L2 values are incorrect.\n");
        return EXIT_FAILURE;
    }

    if (!int64_arrays_match(
        l2_serial_indices,
        expected_l2_indices,
        K
    )) {
        fprintf(stderr, "Serial L2 indices are incorrect.\n");
        return EXIT_FAILURE;
    }

    if (!float_arrays_match(
        l2_openmp_values,
        expected_l2_values,
        K
    )) {
        fprintf(stderr, "OpenMP L2 values are incorrect.\n");
        return EXIT_FAILURE;
    }

    if (!int64_arrays_match(
        l2_openmp_indices,
        expected_l2_indices,
        K
    )) {
        fprintf(stderr, "OpenMP L2 indices are incorrect.\n");
        return EXIT_FAILURE;
    }


    if (!float_arrays_match(
        ip_serial_values,
        expected_ip_values,
        K
    )) {
        fprintf(stderr, "Serial IP values are incorrect.\n");
        return EXIT_FAILURE;
    }

    if (!int64_arrays_match(
        ip_serial_indices,
        expected_ip_indices,
        K
    )) {
        fprintf(stderr, "Serial IP indices are incorrect.\n");
        return EXIT_FAILURE;
    }

    if (!float_arrays_match(
        ip_openmp_values,
        expected_ip_values,
        K
    )) {
        fprintf(stderr, "OpenMP IP values are incorrect.\n");
        return EXIT_FAILURE;
    }

    if (!int64_arrays_match(
        ip_openmp_indices,
        expected_ip_indices,
        K
    )) {
        fprintf(stderr, "OpenMP IP indices are incorrect.\n");
        return EXIT_FAILURE;
    }


    printf(
        "Serial L2:   %.1f@%ld %.1f@%ld\n",
        l2_serial_values[0],
        (long) l2_serial_indices[0],
        l2_serial_values[1],
        (long) l2_serial_indices[1]
    );

    printf(
        "OpenMP L2:   %.1f@%ld %.1f@%ld\n",
        l2_openmp_values[0],
        (long) l2_openmp_indices[0],
        l2_openmp_values[1],
        (long) l2_openmp_indices[1]
    );

    printf(
        "Serial IP:   %.1f@%ld %.1f@%ld\n",
        ip_serial_values[0],
        (long) ip_serial_indices[0],
        ip_serial_values[1],
        (long) ip_serial_indices[1]
    );

    printf(
        "OpenMP IP:   %.1f@%ld %.1f@%ld\n",
        ip_openmp_values[0],
        (long) ip_openmp_indices[0],
        ip_openmp_values[1],
        (long) ip_openmp_indices[1]
    );

    printf("Vector-search C ABI correctness: PASS\n");
    printf("Serial/OpenMP C ABI equivalence: PASS\n");

    return EXIT_SUCCESS;
}