#ifndef FORTRAN_NLP_CUDA_LEXICAL_H
#define FORTRAN_NLP_CUDA_LEXICAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void fnlp_jaccard_corpus_cuda(
    const int64_t *candidate_shingles,
    int64_t candidate_count,
    const int64_t *corpus_shingles,
    int64_t corpus_count,
    const int64_t *document_offsets,
    int document_count,
    int64_t *intersection_counts,
    int64_t *union_counts,
    double *jaccard_scores
);

void fnlp_jaccard_batch_corpus_cuda(
    const int64_t *candidate_shingles,
    int64_t candidate_shingle_count,
    const int64_t *candidate_offsets,
    int candidate_count,
    const int64_t *corpus_shingles,
    int64_t corpus_shingle_count,
    const int64_t *document_offsets,
    int document_count,
    int64_t *intersection_counts,
    int64_t *union_counts,
    double *jaccard_scores
);

#ifdef __cplusplus
}
#endif
#endif