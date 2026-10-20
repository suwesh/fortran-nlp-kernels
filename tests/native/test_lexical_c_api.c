#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "fortran_nlp_lexical.h"

int main(void)
{
    const int64_t candidate[] = {10, 20, 40, 70};

    const int64_t corpus[] = {
        10, 20, 40,
        15, 20,
        10, 30, 60, 70
    };

    const int64_t offsets[] = {0, 3, 5, 9};

    const int64_t candidate_count = 4;
    const int64_t corpus_count = 9;
    const int document_count = 3;

    int64_t serial_intersections[3];
    int64_t serial_unions[3];
    double serial_scores[3];

    int64_t openmp_intersections[3];
    int64_t openmp_unions[3];
    double openmp_scores[3];

    fnlp_jaccard_corpus_serial(
        candidate,
        candidate_count,
        corpus,
        corpus_count,
        offsets,
        document_count,
        serial_intersections,
        serial_unions,
        serial_scores
    );

    fnlp_jaccard_corpus_openmp(
        candidate,
        candidate_count,
        corpus,
        corpus_count,
        offsets,
        document_count,
        openmp_intersections,
        openmp_unions,
        openmp_scores
    );

    for (int i = 0; i < document_count; ++i) {
        if (serial_intersections[i] != openmp_intersections[i]) {
            fprintf(stderr, "Intersection mismatch at document %d\n", i);
            return EXIT_FAILURE;
        }

        if (serial_unions[i] != openmp_unions[i]) {
            fprintf(stderr, "Union mismatch at document %d\n", i);
            return EXIT_FAILURE;
        }

        if (fabs(serial_scores[i] - openmp_scores[i]) > 1.0e-15) {
            fprintf(stderr, "Score mismatch at document %d\n", i);
            return EXIT_FAILURE;
        }
    }

    printf("Intersections:");
    for (int i = 0; i < document_count; ++i) {
        printf(" %ld", (long) serial_intersections[i]);
    }

    printf("\nUnions:");
    for (int i = 0; i < document_count; ++i) {
        printf(" %ld", (long) serial_unions[i]);
    }

    printf("\nScores:");
    for (int i = 0; i < document_count; ++i) {
        printf(" %.15f", serial_scores[i]);
    }

    printf("\nC ABI serial/OpenMP equivalence: PASS\n");

    return EXIT_SUCCESS;
}