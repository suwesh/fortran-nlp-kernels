#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "fortran_nlp_lexical.h"


#define CANDIDATE_COUNT 2
#define CANDIDATE_SHINGLE_COUNT 5

#define DOCUMENT_COUNT 3
#define CORPUS_SHINGLE_COUNT 7

#define RESULT_COUNT \
    (CANDIDATE_COUNT * DOCUMENT_COUNT)

#define TOLERANCE 1.0e-12


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


static int double_arrays_match(
    const double *actual,
    const double *expected,
    int count
)
{
    for (int index = 0; index < count; ++index) {
        if (
            fabs(actual[index] - expected[index])
            > TOLERANCE
        ) {
            return 0;
        }
    }

    return 1;
}


int main(void)
{
    /*
     * Candidate 0 = [1, 2, 3]
     * Candidate 1 = [2, 4]
     */
    const int64_t candidate_shingles[
        CANDIDATE_SHINGLE_COUNT
    ] = {
        1,
        2,
        3,
        2,
        4
    };

    const int64_t candidate_offsets[
        CANDIDATE_COUNT + 1
    ] = {
        0,
        3,
        5
    };


    /*
     * Document 0 = [1, 2, 3]
     * Document 1 = [2, 3, 4]
     * Document 2 = [5]
     */
    const int64_t corpus_shingles[
        CORPUS_SHINGLE_COUNT
    ] = {
        1,
        2,
        3,
        2,
        3,
        4,
        5
    };

    const int64_t document_offsets[
        DOCUMENT_COUNT + 1
    ] = {
        0,
        3,
        6,
        7
    };


    /*
     * Fortran matrix shape:
     *
     *     (document_count, candidate_count)
     *
     * Fortran column-major flattened layout:
     *
     *     candidate 0 results for all documents,
     *     candidate 1 results for all documents.
     */

    const int64_t expected_intersections[
        RESULT_COUNT
    ] = {
        3,
        2,
        0,
        1,
        2,
        0
    };

    const int64_t expected_unions[
        RESULT_COUNT
    ] = {
        3,
        4,
        4,
        4,
        3,
        3
    };

    const double expected_scores[
        RESULT_COUNT
    ] = {
        1.0,
        0.5,
        0.0,
        0.25,
        2.0 / 3.0,
        0.0
    };


    int64_t intersection_counts[RESULT_COUNT];
    int64_t union_counts[RESULT_COUNT];
    double jaccard_scores[RESULT_COUNT];


    fnlp_jaccard_batch_corpus(
        candidate_shingles,
        CANDIDATE_SHINGLE_COUNT,
        candidate_offsets,
        CANDIDATE_COUNT,
        corpus_shingles,
        CORPUS_SHINGLE_COUNT,
        document_offsets,
        DOCUMENT_COUNT,
        intersection_counts,
        union_counts,
        jaccard_scores
    );


    if (!int64_arrays_match(
        intersection_counts,
        expected_intersections,
        RESULT_COUNT
    )) {
        fprintf(
            stderr,
            "Batched C ABI intersection counts are incorrect.\n"
        );

        return EXIT_FAILURE;
    }


    if (!int64_arrays_match(
        union_counts,
        expected_unions,
        RESULT_COUNT
    )) {
        fprintf(
            stderr,
            "Batched C ABI union counts are incorrect.\n"
        );

        return EXIT_FAILURE;
    }


    if (!double_arrays_match(
        jaccard_scores,
        expected_scores,
        RESULT_COUNT
    )) {
        fprintf(
            stderr,
            "Batched C ABI Jaccard scores are incorrect.\n"
        );

        return EXIT_FAILURE;
    }


    printf("Batch intersections:\n");

    for (
        int candidate_index = 0;
        candidate_index < CANDIDATE_COUNT;
        ++candidate_index
    ) {
        printf(
            "  Candidate %d:",
            candidate_index
        );

        for (
            int document_index = 0;
            document_index < DOCUMENT_COUNT;
            ++document_index
        ) {
            const int result_index =
                candidate_index * DOCUMENT_COUNT
                + document_index;

            printf(
                " %ld",
                (long) intersection_counts[result_index]
            );
        }

        printf("\n");
    }


    printf("Batch unions:\n");

    for (
        int candidate_index = 0;
        candidate_index < CANDIDATE_COUNT;
        ++candidate_index
    ) {
        printf(
            "  Candidate %d:",
            candidate_index
        );

        for (
            int document_index = 0;
            document_index < DOCUMENT_COUNT;
            ++document_index
        ) {
            const int result_index =
                candidate_index * DOCUMENT_COUNT
                + document_index;

            printf(
                " %ld",
                (long) union_counts[result_index]
            );
        }

        printf("\n");
    }


    printf("Batch scores:\n");

    for (
        int candidate_index = 0;
        candidate_index < CANDIDATE_COUNT;
        ++candidate_index
    ) {
        printf(
            "  Candidate %d:",
            candidate_index
        );

        for (
            int document_index = 0;
            document_index < DOCUMENT_COUNT;
            ++document_index
        ) {
            const int result_index =
                candidate_index * DOCUMENT_COUNT
                + document_index;

            printf(
                " %.12f",
                jaccard_scores[result_index]
            );
        }

        printf("\n");
    }


    printf(
        "Batched Jaccard C ABI correctness: PASS\n"
    );

    return EXIT_SUCCESS;
}