#ifndef FORTRAN_NLP_VECTOR_SEARCH_H
#define FORTRAN_NLP_VECTOR_SEARCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void fnlp_l2_search_serial(
    const float *query_vector,
    const float *database_vectors,
    int dimension,
    int vector_count,
    int k,
    float *topk_distances,
    int64_t *topk_indices
);

void fnlp_l2_search_openmp(
    const float *query_vector,
    const float *database_vectors,
    int dimension,
    int vector_count,
    int k,
    float *topk_distances,
    int64_t *topk_indices
);

void fnlp_ip_search_serial(
    const float *query_vector,
    const float *database_vectors,
    int dimension,
    int vector_count,
    int k,
    float *topk_similarities,
    int64_t *topk_indices
);

void fnlp_ip_search_openmp(
    const float *query_vector,
    const float *database_vectors,
    int dimension,
    int vector_count,
    int k,
    float *topk_similarities,
    int64_t *topk_indices
);

void fnlp_l2_search_blas(
    const float *query_vector,
    const float *database_vectors,
    int dimension,
    int vector_count,
    int k,
    float *topk_distances,
    int64_t *topk_indices
);

void fnlp_ip_search_blas(
    const float *query_vector,
    const float *database_vectors,
    int dimension,
    int vector_count,
    int k,
    float *topk_similarities,
    int64_t *topk_indices
);

void fnlp_l2_search_batch_blas(
    const float *query_vectors,
    const float *database_vectors,
    int dimension,
    int query_count,
    int vector_count,
    int k,
    float *topk_distances,
    int64_t *topk_indices
);

void fnlp_ip_search_batch_blas(
    const float *query_vectors,
    const float *database_vectors,
    int dimension,
    int query_count,
    int vector_count,
    int k,
    float *topk_similarities,
    int64_t *topk_indices
);

#ifdef __cplusplus
}
#endif
#endif