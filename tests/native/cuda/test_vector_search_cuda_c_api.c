#include "fortran_nlp_cuda_vector_search.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIMENSION 3
#define VECTOR_COUNT 5
#define QUERY_COUNT 3
#define K 3

static void assert_float_close(
    float actual,
    float expected,
    const char *label
) {
    const float absolute_tolerance = 1.0e-4f;
    const float relative_tolerance = 1.0e-4f;
    const float difference = fabsf(actual - expected);
    const float allowed = absolute_tolerance
        + relative_tolerance * fabsf(expected);

    if (difference > allowed) {
        fprintf(
            stderr,
            "FAIL: %s: actual=%g expected=%g error=%g allowed=%g\n",
            label,
            (double)actual,
            (double)expected,
            (double)difference,
            (double)allowed
        );
        exit(EXIT_FAILURE);
    }
}

static void assert_float_array_close(
    const float *actual,
    const float *expected,
    int count,
    const char *label
) {
    int index;

    for (index = 0; index < count; ++index) {
        char element_label[256];
        snprintf(
            element_label,
            sizeof(element_label),
            "%s at flattened index %d",
            label,
            index
        );
        assert_float_close(
            actual[index],
            expected[index],
            element_label
        );
    }
}

static void assert_int64_array_equal(
    const int64_t *actual,
    const int64_t *expected,
    int count,
    const char *label
) {
    int index;

    for (index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) {
            fprintf(
                stderr,
                "FAIL: %s at flattened index %d: "
                "actual=%lld expected=%lld\n",
                label,
                index,
                (long long)actual[index],
                (long long)expected[index]
            );
            exit(EXIT_FAILURE);
        }
    }
}

static void assert_nonnegative(
    const float *values,
    int count,
    const char *label
) {
    int index;

    for (index = 0; index < count; ++index) {
        if (values[index] < 0.0f) {
            fprintf(
                stderr,
                "FAIL: %s at flattened index %d is negative: %g\n",
                label,
                index,
                (double)values[index]
            );
            exit(EXIT_FAILURE);
        }
    }
}

static void test_single_known_results(void) {
    /*
     * Arrays use Fortran column-major layout.
     * database shape: (dimension=3, vector_count=5)
     */
    const float query[DIMENSION] = {
        1.0f, 2.0f, 3.0f
    };

    const float database[DIMENSION * VECTOR_COUNT] = {
        1.0f,  2.0f,  3.0f,
        2.0f,  2.0f,  3.0f,
        1.0f,  4.0f,  3.0f,
        0.0f,  0.0f,  0.0f,
       -1.0f, -2.0f, -3.0f
    };

    const float expected_l2[K] = {
        0.0f, 1.0f, 4.0f
    };

    const int64_t expected_l2_indices[K] = {
        0, 1, 2
    };

    const float expected_ip[K] = {
        18.0f, 15.0f, 14.0f
    };

    const int64_t expected_ip_indices[K] = {
        2, 1, 0
    };

    float cuda_l2[K];
    float cuda_ip[K];
    float cublas_l2[K];
    float cublas_ip[K];
    int64_t cuda_l2_indices[K];
    int64_t cuda_ip_indices[K];
    int64_t cublas_l2_indices[K];
    int64_t cublas_ip_indices[K];

    fnlp_l2_search_cuda(
        query,
        database,
        DIMENSION,
        VECTOR_COUNT,
        K,
        cuda_l2,
        cuda_l2_indices
    );

    fnlp_ip_search_cuda(
        query,
        database,
        DIMENSION,
        VECTOR_COUNT,
        K,
        cuda_ip,
        cuda_ip_indices
    );

    fnlp_l2_search_cublas(
        query,
        database,
        DIMENSION,
        VECTOR_COUNT,
        K,
        cublas_l2,
        cublas_l2_indices
    );

    fnlp_ip_search_cublas(
        query,
        database,
        DIMENSION,
        VECTOR_COUNT,
        K,
        cublas_ip,
        cublas_ip_indices
    );

    assert_float_array_close(
        cuda_l2, expected_l2, K, "CUDA C ABI single L2 values"
    );
    assert_int64_array_equal(
        cuda_l2_indices,
        expected_l2_indices,
        K,
        "CUDA C ABI single L2 indices"
    );
    assert_float_array_close(
        cuda_ip, expected_ip, K, "CUDA C ABI single IP values"
    );
    assert_int64_array_equal(
        cuda_ip_indices,
        expected_ip_indices,
        K,
        "CUDA C ABI single IP indices"
    );

    assert_float_array_close(
        cublas_l2, expected_l2, K, "cuBLAS C ABI single L2 values"
    );
    assert_int64_array_equal(
        cublas_l2_indices,
        expected_l2_indices,
        K,
        "cuBLAS C ABI single L2 indices"
    );
    assert_float_array_close(
        cublas_ip, expected_ip, K, "cuBLAS C ABI single IP values"
    );
    assert_int64_array_equal(
        cublas_ip_indices,
        expected_ip_indices,
        K,
        "cuBLAS C ABI single IP indices"
    );

    assert_float_array_close(
        cuda_l2, cublas_l2, K, "CUDA/cuBLAS single L2 agreement"
    );
    assert_float_array_close(
        cuda_ip, cublas_ip, K, "CUDA/cuBLAS single IP agreement"
    );

    printf("Single-query C ABI known results: PASS\n");
}

static void test_single_deterministic_ties(void) {
    enum {
        dimension = 2,
        vector_count = 4,
        k = 4
    };

    const float query[dimension] = {
        1.0f, 1.0f
    };

    const float database[dimension * vector_count] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    const float expected_values[k] = {
        1.0f, 1.0f, 1.0f, 1.0f
    };

    const int64_t expected_indices[k] = {
        0, 1, 2, 3
    };

    float cuda_l2[k];
    float cuda_ip[k];
    float cublas_l2[k];
    float cublas_ip[k];
    int64_t cuda_l2_indices[k];
    int64_t cuda_ip_indices[k];
    int64_t cublas_l2_indices[k];
    int64_t cublas_ip_indices[k];

    fnlp_l2_search_cuda(
        query, database, dimension, vector_count, k,
        cuda_l2, cuda_l2_indices
    );
    fnlp_ip_search_cuda(
        query, database, dimension, vector_count, k,
        cuda_ip, cuda_ip_indices
    );
    fnlp_l2_search_cublas(
        query, database, dimension, vector_count, k,
        cublas_l2, cublas_l2_indices
    );
    fnlp_ip_search_cublas(
        query, database, dimension, vector_count, k,
        cublas_ip, cublas_ip_indices
    );

    assert_float_array_close(
        cuda_l2, expected_values, k, "CUDA C ABI tied L2 values"
    );
    assert_float_array_close(
        cuda_ip, expected_values, k, "CUDA C ABI tied IP values"
    );
    assert_float_array_close(
        cublas_l2, expected_values, k, "cuBLAS C ABI tied L2 values"
    );
    assert_float_array_close(
        cublas_ip, expected_values, k, "cuBLAS C ABI tied IP values"
    );

    assert_int64_array_equal(
        cuda_l2_indices,
        expected_indices,
        k,
        "CUDA C ABI tied L2 indices"
    );
    assert_int64_array_equal(
        cuda_ip_indices,
        expected_indices,
        k,
        "CUDA C ABI tied IP indices"
    );
    assert_int64_array_equal(
        cublas_l2_indices,
        expected_indices,
        k,
        "cuBLAS C ABI tied L2 indices"
    );
    assert_int64_array_equal(
        cublas_ip_indices,
        expected_indices,
        k,
        "cuBLAS C ABI tied IP indices"
    );

    printf("Single-query deterministic ties and zero-based indices: PASS\n");
}

static void test_batch_known_results(void) {
    /* queries shape: (dimension=3, query_count=3), column-major */
    const float queries[DIMENSION * QUERY_COUNT] = {
         1.0f,  2.0f,  3.0f,
         2.0f, -1.0f,  0.5f,
        -2.0f,  1.0f,  4.0f
    };

    const float database[DIMENSION * VECTOR_COUNT] = {
         1.0f,  2.0f,  3.0f,
         2.0f,  2.0f,  3.0f,
         1.0f,  4.0f,  3.0f,
         0.0f,  0.0f,  0.0f,
        -1.0f, -2.0f, -3.0f
    };

    /* Output shape is (k, query_count), also column-major. */
    const float expected_l2[K * QUERY_COUNT] = {
         0.0f,  1.0f,  4.0f,
         5.25f, 15.25f, 16.25f,
        11.0f, 18.0f, 19.0f
    };

    const int64_t expected_l2_indices[K * QUERY_COUNT] = {
        0, 1, 2,
        3, 1, 0,
        0, 1, 2
    };

    const float expected_ip[K * QUERY_COUNT] = {
        18.0f, 15.0f, 14.0f,
         3.5f,  1.5f,  0.0f,
        14.0f, 12.0f, 10.0f
    };

    const int64_t expected_ip_indices[K * QUERY_COUNT] = {
        2, 1, 0,
        1, 0, 3,
        2, 0, 1
    };

    float cuda_l2[K * QUERY_COUNT];
    float cuda_ip[K * QUERY_COUNT];
    float cublas_l2[K * QUERY_COUNT];
    float cublas_ip[K * QUERY_COUNT];
    int64_t cuda_l2_indices[K * QUERY_COUNT];
    int64_t cuda_ip_indices[K * QUERY_COUNT];
    int64_t cublas_l2_indices[K * QUERY_COUNT];
    int64_t cublas_ip_indices[K * QUERY_COUNT];

    fnlp_l2_search_batch_cuda(
        queries,
        database,
        DIMENSION,
        QUERY_COUNT,
        VECTOR_COUNT,
        K,
        cuda_l2,
        cuda_l2_indices
    );

    fnlp_ip_search_batch_cuda(
        queries,
        database,
        DIMENSION,
        QUERY_COUNT,
        VECTOR_COUNT,
        K,
        cuda_ip,
        cuda_ip_indices
    );

    fnlp_l2_search_batch_cublas(
        queries,
        database,
        DIMENSION,
        QUERY_COUNT,
        VECTOR_COUNT,
        K,
        cublas_l2,
        cublas_l2_indices
    );

    fnlp_ip_search_batch_cublas(
        queries,
        database,
        DIMENSION,
        QUERY_COUNT,
        VECTOR_COUNT,
        K,
        cublas_ip,
        cublas_ip_indices
    );

    assert_float_array_close(
        cuda_l2,
        expected_l2,
        K * QUERY_COUNT,
        "CUDA C ABI batched L2 values"
    );
    assert_int64_array_equal(
        cuda_l2_indices,
        expected_l2_indices,
        K * QUERY_COUNT,
        "CUDA C ABI batched L2 indices"
    );
    assert_float_array_close(
        cuda_ip,
        expected_ip,
        K * QUERY_COUNT,
        "CUDA C ABI batched IP values"
    );
    assert_int64_array_equal(
        cuda_ip_indices,
        expected_ip_indices,
        K * QUERY_COUNT,
        "CUDA C ABI batched IP indices"
    );

    assert_float_array_close(
        cublas_l2,
        expected_l2,
        K * QUERY_COUNT,
        "cuBLAS C ABI batched L2 values"
    );
    assert_int64_array_equal(
        cublas_l2_indices,
        expected_l2_indices,
        K * QUERY_COUNT,
        "cuBLAS C ABI batched L2 indices"
    );
    assert_float_array_close(
        cublas_ip,
        expected_ip,
        K * QUERY_COUNT,
        "cuBLAS C ABI batched IP values"
    );
    assert_int64_array_equal(
        cublas_ip_indices,
        expected_ip_indices,
        K * QUERY_COUNT,
        "cuBLAS C ABI batched IP indices"
    );

    assert_float_array_close(
        cuda_l2,
        cublas_l2,
        K * QUERY_COUNT,
        "CUDA/cuBLAS batched L2 agreement"
    );
    assert_float_array_close(
        cuda_ip,
        cublas_ip,
        K * QUERY_COUNT,
        "CUDA/cuBLAS batched IP agreement"
    );

    printf("Batched C ABI known results and layout: PASS\n");
}

static void test_batch_size_one(void) {
    enum {
        dimension = 4,
        vector_count = 5,
        query_count = 1,
        k = 3
    };

    const float query[dimension] = {
        1.0f, -2.0f, 3.0f, 0.5f
    };

    const float queries[dimension * query_count] = {
        1.0f, -2.0f, 3.0f, 0.5f
    };

    const float database[dimension * vector_count] = {
         1.0f, -2.0f,  3.0f,  0.5f,
         0.0f,  0.0f,  0.0f,  0.0f,
        -1.0f,  2.0f, -3.0f, -0.5f,
         2.0f, -1.0f,  4.0f,  1.5f,
         5.0f,  5.0f,  5.0f,  5.0f
    };

    float single_cuda_l2[k];
    float batch_cuda_l2[k];
    float single_cuda_ip[k];
    float batch_cuda_ip[k];
    float single_cublas_l2[k];
    float batch_cublas_l2[k];
    float single_cublas_ip[k];
    float batch_cublas_ip[k];
    int64_t single_cuda_l2_indices[k];
    int64_t batch_cuda_l2_indices[k];
    int64_t single_cuda_ip_indices[k];
    int64_t batch_cuda_ip_indices[k];
    int64_t single_cublas_l2_indices[k];
    int64_t batch_cublas_l2_indices[k];
    int64_t single_cublas_ip_indices[k];
    int64_t batch_cublas_ip_indices[k];

    fnlp_l2_search_cuda(
        query, database, dimension, vector_count, k,
        single_cuda_l2, single_cuda_l2_indices
    );
    fnlp_l2_search_batch_cuda(
        queries, database, dimension, query_count, vector_count, k,
        batch_cuda_l2, batch_cuda_l2_indices
    );
    fnlp_ip_search_cuda(
        query, database, dimension, vector_count, k,
        single_cuda_ip, single_cuda_ip_indices
    );
    fnlp_ip_search_batch_cuda(
        queries, database, dimension, query_count, vector_count, k,
        batch_cuda_ip, batch_cuda_ip_indices
    );

    fnlp_l2_search_cublas(
        query, database, dimension, vector_count, k,
        single_cublas_l2, single_cublas_l2_indices
    );
    fnlp_l2_search_batch_cublas(
        queries, database, dimension, query_count, vector_count, k,
        batch_cublas_l2, batch_cublas_l2_indices
    );
    fnlp_ip_search_cublas(
        query, database, dimension, vector_count, k,
        single_cublas_ip, single_cublas_ip_indices
    );
    fnlp_ip_search_batch_cublas(
        queries, database, dimension, query_count, vector_count, k,
        batch_cublas_ip, batch_cublas_ip_indices
    );

    assert_float_array_close(
        batch_cuda_l2, single_cuda_l2, k,
        "CUDA C ABI batch-size-one L2 values"
    );
    assert_int64_array_equal(
        batch_cuda_l2_indices, single_cuda_l2_indices, k,
        "CUDA C ABI batch-size-one L2 indices"
    );
    assert_float_array_close(
        batch_cuda_ip, single_cuda_ip, k,
        "CUDA C ABI batch-size-one IP values"
    );
    assert_int64_array_equal(
        batch_cuda_ip_indices, single_cuda_ip_indices, k,
        "CUDA C ABI batch-size-one IP indices"
    );

    assert_float_array_close(
        batch_cublas_l2, single_cublas_l2, k,
        "cuBLAS C ABI batch-size-one L2 values"
    );
    assert_int64_array_equal(
        batch_cublas_l2_indices, single_cublas_l2_indices, k,
        "cuBLAS C ABI batch-size-one L2 indices"
    );
    assert_float_array_close(
        batch_cublas_ip, single_cublas_ip, k,
        "cuBLAS C ABI batch-size-one IP values"
    );
    assert_int64_array_equal(
        batch_cublas_ip_indices, single_cublas_ip_indices, k,
        "cuBLAS C ABI batch-size-one IP indices"
    );

    printf("C ABI batch size one: PASS\n");
}

static void test_batch_deterministic_ties(void) {
    enum {
        dimension = 2,
        query_count = 2,
        vector_count = 4,
        k = 4
    };

    const float queries[dimension * query_count] = {
         1.0f,  1.0f,
        -1.0f, -1.0f
    };

    const float database[dimension * vector_count] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    const int64_t expected_indices[k * query_count] = {
        0, 1, 2, 3,
        0, 1, 2, 3
    };

    float cuda_l2[k * query_count];
    float cuda_ip[k * query_count];
    float cublas_l2[k * query_count];
    float cublas_ip[k * query_count];
    int64_t cuda_l2_indices[k * query_count];
    int64_t cuda_ip_indices[k * query_count];
    int64_t cublas_l2_indices[k * query_count];
    int64_t cublas_ip_indices[k * query_count];

    fnlp_l2_search_batch_cuda(
        queries, database, dimension, query_count, vector_count, k,
        cuda_l2, cuda_l2_indices
    );
    fnlp_ip_search_batch_cuda(
        queries, database, dimension, query_count, vector_count, k,
        cuda_ip, cuda_ip_indices
    );
    fnlp_l2_search_batch_cublas(
        queries, database, dimension, query_count, vector_count, k,
        cublas_l2, cublas_l2_indices
    );
    fnlp_ip_search_batch_cublas(
        queries, database, dimension, query_count, vector_count, k,
        cublas_ip, cublas_ip_indices
    );

    assert_int64_array_equal(
        cuda_l2_indices,
        expected_indices,
        k * query_count,
        "CUDA C ABI batched L2 ties"
    );
    assert_int64_array_equal(
        cuda_ip_indices,
        expected_indices,
        k * query_count,
        "CUDA C ABI batched IP ties"
    );
    assert_int64_array_equal(
        cublas_l2_indices,
        expected_indices,
        k * query_count,
        "cuBLAS C ABI batched L2 ties"
    );
    assert_int64_array_equal(
        cublas_ip_indices,
        expected_indices,
        k * query_count,
        "cuBLAS C ABI batched IP ties"
    );

    printf("Batched deterministic ties and zero-based indices: PASS\n");
}

static void test_l2_nonnegative_rounding_clamp(void) {
    enum {
        dimension = 8,
        query_count = 2,
        vector_count = 2,
        k = 2
    };

    const float queries[dimension * query_count] = {
         0.1f,  0.2f,  0.3f,  0.4f,
         0.5f,  0.6f,  0.7f,  0.8f,
        -0.1f, -0.2f, -0.3f, -0.4f,
        -0.5f, -0.6f, -0.7f, -0.8f
    };

    const float database[dimension * vector_count] = {
         0.1f,  0.2f,  0.3f,  0.4f,
         0.5f,  0.6f,  0.7f,  0.8f,
        -0.1f, -0.2f, -0.3f, -0.4f,
        -0.5f, -0.6f, -0.7f, -0.8f
    };

    float cuda_values[k * query_count];
    float cublas_values[k * query_count];
    int64_t cuda_indices[k * query_count];
    int64_t cublas_indices[k * query_count];

    fnlp_l2_search_batch_cuda(
        queries, database, dimension, query_count, vector_count, k,
        cuda_values, cuda_indices
    );
    fnlp_l2_search_batch_cublas(
        queries, database, dimension, query_count, vector_count, k,
        cublas_values, cublas_indices
    );

    assert_nonnegative(
        cuda_values,
        k * query_count,
        "CUDA C ABI L2 values"
    );
    assert_nonnegative(
        cublas_values,
        k * query_count,
        "cuBLAS C ABI L2 values"
    );

    assert_float_close(
        cuda_values[0], 0.0f, "CUDA identical-vector L2 query 1"
    );
    assert_float_close(
        cuda_values[k], 0.0f, "CUDA identical-vector L2 query 2"
    );
    assert_float_close(
        cublas_values[0], 0.0f, "cuBLAS identical-vector L2 query 1"
    );
    assert_float_close(
        cublas_values[k], 0.0f, "cuBLAS identical-vector L2 query 2"
    );

    printf("C ABI L2 nonnegative rounding clamp: PASS\n");
}

int main(void) {
    test_single_known_results();
    test_single_deterministic_ties();
    test_batch_known_results();
    test_batch_size_one();
    test_batch_deterministic_ties();
    test_l2_nonnegative_rounding_clamp();

    printf("CUDA vector-search C ABI tests passed\n");
    return EXIT_SUCCESS;
}
