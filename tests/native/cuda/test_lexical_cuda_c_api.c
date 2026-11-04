#include "fortran_nlp_cuda_lexical.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


static const double TOLERANCE = 1.0e-15;


static void fail(const char *label)
{
    fprintf(
        stderr,
        "CUDA lexical C API test failed: %s\n",
        label
    );

    exit(EXIT_FAILURE);
}


static void assert_int64_equal(
    const int64_t *actual,
    const int64_t *expected,
    int count,
    const char *label
)
{
    int index;

    for (index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) {
            fprintf(
                stderr,
                "%s mismatch at index %d: "
                "actual=%lld expected=%lld\n",
                label,
                index,
                (long long) actual[index],
                (long long) expected[index]
            );

            fail(label);
        }
    }
}


static void assert_double_close(
    const double *actual,
    const double *expected,
    int count,
    const char *label
)
{
    int index;

    for (index = 0; index < count; ++index) {
        const double difference = fabs(
            actual[index] - expected[index]
        );

        if (difference > TOLERANCE) {
            fprintf(
                stderr,
                "%s mismatch at index %d: "
                "actual=%.17g expected=%.17g "
                "difference=%.17g\n",
                label,
                index,
                actual[index],
                expected[index],
                difference
            );

            fail(label);
        }
    }
}


static void test_single_candidate(void)
{
    enum {
        DOCUMENT_COUNT = 4
    };

    const int64_t candidate_shingles[] = {
        10,
        20,
        40,
        70
    };

    const int64_t corpus_shingles[] = {
        10, 20, 40,
        15, 20,
        10, 30, 60, 70
    };

    /*
     * Document 1: [10, 20, 40]
     * Document 2: [15, 20]
     * Document 3: [10, 30, 60, 70]
     * Document 4: empty
     */
    const int64_t document_offsets[] = {
        0,
        3,
        5,
        9,
        9
    };

    const int64_t candidate_count = (
        (int64_t) (
            sizeof(candidate_shingles)
            / sizeof(candidate_shingles[0])
        )
    );

    const int64_t corpus_count = (
        (int64_t) (
            sizeof(corpus_shingles)
            / sizeof(corpus_shingles[0])
        )
    );

    int64_t intersection_counts[DOCUMENT_COUNT] = {0};
    int64_t union_counts[DOCUMENT_COUNT] = {0};
    double jaccard_scores[DOCUMENT_COUNT] = {0.0};

    const int64_t expected_intersections[DOCUMENT_COUNT] = {
        3,
        1,
        2,
        0
    };

    const int64_t expected_unions[DOCUMENT_COUNT] = {
        4,
        5,
        6,
        4
    };

    const double expected_scores[DOCUMENT_COUNT] = {
        0.75,
        0.20,
        1.0 / 3.0,
        0.0
    };

    fnlp_jaccard_corpus_cuda(
        candidate_shingles,
        candidate_count,
        corpus_shingles,
        corpus_count,
        document_offsets,
        DOCUMENT_COUNT,
        intersection_counts,
        union_counts,
        jaccard_scores
    );

    assert_int64_equal(
        intersection_counts,
        expected_intersections,
        DOCUMENT_COUNT,
        "single intersections"
    );

    assert_int64_equal(
        union_counts,
        expected_unions,
        DOCUMENT_COUNT,
        "single unions"
    );

    assert_double_close(
        jaccard_scores,
        expected_scores,
        DOCUMENT_COUNT,
        "single scores"
    );

    printf(
        "CUDA lexical single-candidate C API: PASS\n"
    );
}


static void test_empty_candidate(void)
{
    enum {
        DOCUMENT_COUNT = 4
    };

    /*
     * C does not support a portable zero-length array, so this
     * one-element placeholder is passed with candidate_count = 0.
     * The CUDA C ABI must not read the placeholder value.
     */
    const int64_t candidate_shingles[] = {
        0
    };

    const int64_t corpus_shingles[] = {
        10, 20, 40,
        15, 20,
        10, 30, 60, 70
    };

    const int64_t document_offsets[] = {
        0,
        3,
        5,
        9,
        9
    };

    const int64_t candidate_count = 0;

    const int64_t corpus_count = (
        (int64_t) (
            sizeof(corpus_shingles)
            / sizeof(corpus_shingles[0])
        )
    );

    int64_t intersection_counts[DOCUMENT_COUNT] = {0};
    int64_t union_counts[DOCUMENT_COUNT] = {0};
    double jaccard_scores[DOCUMENT_COUNT] = {0.0};

    const int64_t expected_intersections[DOCUMENT_COUNT] = {
        0,
        0,
        0,
        0
    };

    const int64_t expected_unions[DOCUMENT_COUNT] = {
        3,
        2,
        4,
        0
    };

    const double expected_scores[DOCUMENT_COUNT] = {
        0.0,
        0.0,
        0.0,
        0.0
    };

    fnlp_jaccard_corpus_cuda(
        candidate_shingles,
        candidate_count,
        corpus_shingles,
        corpus_count,
        document_offsets,
        DOCUMENT_COUNT,
        intersection_counts,
        union_counts,
        jaccard_scores
    );

    assert_int64_equal(
        intersection_counts,
        expected_intersections,
        DOCUMENT_COUNT,
        "empty-candidate intersections"
    );

    assert_int64_equal(
        union_counts,
        expected_unions,
        DOCUMENT_COUNT,
        "empty-candidate unions"
    );

    assert_double_close(
        jaccard_scores,
        expected_scores,
        DOCUMENT_COUNT,
        "empty-candidate scores"
    );

    printf(
        "CUDA lexical empty-candidate C API: PASS\n"
    );
}


static void test_batch(void)
{
    enum {
        CANDIDATE_COUNT = 3,
        DOCUMENT_COUNT = 4,
        RESULT_COUNT = CANDIDATE_COUNT * DOCUMENT_COUNT
    };

    /*
     * Candidate 1: [10, 20, 40, 70]
     * Candidate 2: [15, 20]
     * Candidate 3: empty
     */
    const int64_t candidate_shingles[] = {
        10, 20, 40, 70,
        15, 20
    };

    const int64_t candidate_offsets[] = {
        0,
        4,
        6,
        6
    };

    /*
     * Document 1: [10, 20, 40]
     * Document 2: [15, 20]
     * Document 3: [10, 30, 60, 70]
     * Document 4: empty
     */
    const int64_t corpus_shingles[] = {
        10, 20, 40,
        15, 20,
        10, 30, 60, 70
    };

    const int64_t document_offsets[] = {
        0,
        3,
        5,
        9,
        9
    };

    const int64_t candidate_shingle_count = (
        (int64_t) (
            sizeof(candidate_shingles)
            / sizeof(candidate_shingles[0])
        )
    );

    const int64_t corpus_shingle_count = (
        (int64_t) (
            sizeof(corpus_shingles)
            / sizeof(corpus_shingles[0])
        )
    );

    int64_t intersection_counts[RESULT_COUNT] = {0};
    int64_t union_counts[RESULT_COUNT] = {0};
    double jaccard_scores[RESULT_COUNT] = {0.0};

    /*
     * Flattened Fortran layout:
     *
     * candidate 1 against documents 1 through 4,
     * candidate 2 against documents 1 through 4,
     * candidate 3 against documents 1 through 4.
     */
    const int64_t expected_intersections[RESULT_COUNT] = {
        3, 1, 2, 0,
        1, 2, 0, 0,
        0, 0, 0, 0
    };

    const int64_t expected_unions[RESULT_COUNT] = {
        4, 5, 6, 4,
        4, 2, 6, 2,
        3, 2, 4, 0
    };

    const double expected_scores[RESULT_COUNT] = {
        0.75,
        0.20,
        1.0 / 3.0,
        0.0,

        0.25,
        1.0,
        0.0,
        0.0,

        0.0,
        0.0,
        0.0,
        0.0
    };

    fnlp_jaccard_batch_corpus_cuda(
        candidate_shingles,
        candidate_shingle_count,
        candidate_offsets,
        CANDIDATE_COUNT,
        corpus_shingles,
        corpus_shingle_count,
        document_offsets,
        DOCUMENT_COUNT,
        intersection_counts,
        union_counts,
        jaccard_scores
    );

    assert_int64_equal(
        intersection_counts,
        expected_intersections,
        RESULT_COUNT,
        "batch intersections"
    );

    assert_int64_equal(
        union_counts,
        expected_unions,
        RESULT_COUNT,
        "batch unions"
    );

    assert_double_close(
        jaccard_scores,
        expected_scores,
        RESULT_COUNT,
        "batch scores"
    );

    printf(
        "CUDA lexical batch C API: PASS\n"
    );
}


static void test_batch_size_one(void)
{
    enum {
        CANDIDATE_COUNT = 1,
        DOCUMENT_COUNT = 3,
        RESULT_COUNT = CANDIDATE_COUNT * DOCUMENT_COUNT
    };

    const int64_t candidate_shingles[] = {
        10,
        20,
        40,
        70
    };

    const int64_t candidate_offsets[] = {
        0,
        4
    };

    const int64_t corpus_shingles[] = {
        10, 20, 40,
        15, 20,
        10, 30, 60, 70
    };

    const int64_t document_offsets[] = {
        0,
        3,
        5,
        9
    };

    const int64_t candidate_shingle_count = (
        (int64_t) (
            sizeof(candidate_shingles)
            / sizeof(candidate_shingles[0])
        )
    );

    const int64_t corpus_shingle_count = (
        (int64_t) (
            sizeof(corpus_shingles)
            / sizeof(corpus_shingles[0])
        )
    );

    int64_t intersection_counts[RESULT_COUNT] = {0};
    int64_t union_counts[RESULT_COUNT] = {0};
    double jaccard_scores[RESULT_COUNT] = {0.0};

    const int64_t expected_intersections[RESULT_COUNT] = {
        3,
        1,
        2
    };

    const int64_t expected_unions[RESULT_COUNT] = {
        4,
        5,
        6
    };

    const double expected_scores[RESULT_COUNT] = {
        0.75,
        0.20,
        1.0 / 3.0
    };

    fnlp_jaccard_batch_corpus_cuda(
        candidate_shingles,
        candidate_shingle_count,
        candidate_offsets,
        CANDIDATE_COUNT,
        corpus_shingles,
        corpus_shingle_count,
        document_offsets,
        DOCUMENT_COUNT,
        intersection_counts,
        union_counts,
        jaccard_scores
    );

    assert_int64_equal(
        intersection_counts,
        expected_intersections,
        RESULT_COUNT,
        "batch-one intersections"
    );

    assert_int64_equal(
        union_counts,
        expected_unions,
        RESULT_COUNT,
        "batch-one unions"
    );

    assert_double_close(
        jaccard_scores,
        expected_scores,
        RESULT_COUNT,
        "batch-one scores"
    );

    printf(
        "CUDA lexical batch-size-one C API: PASS\n"
    );
}


int main(void)
{
    test_single_candidate();
    test_empty_candidate();
    test_batch();
    test_batch_size_one();

    printf(
        "CUDA lexical C API tests passed\n"
    );

    return EXIT_SUCCESS;
}