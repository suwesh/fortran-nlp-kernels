import numpy as np
import pytest

from fortran_nlp import _cuda


def _single_inputs():
    candidate = np.array(
        [10, 20, 40, 70],
        dtype=np.int64,
    )

    corpus = np.array(
        [
            10, 20, 40,
            15, 20,
            10, 30, 60, 70,
        ],
        dtype=np.int64,
    )

    document_offsets = np.array(
        [0, 3, 5, 9, 9],
        dtype=np.int64,
    )

    return candidate, corpus, document_offsets


def _batch_inputs():
    candidate_shingles = np.array(
        [
            10, 20, 40, 70,
            15, 20,
        ],
        dtype=np.int64,
    )

    candidate_offsets = np.array(
        [0, 4, 6, 6],
        dtype=np.int64,
    )

    corpus = np.array(
        [
            10, 20, 40,
            15, 20,
            10, 30, 60, 70,
        ],
        dtype=np.int64,
    )

    document_offsets = np.array(
        [0, 3, 5, 9, 9],
        dtype=np.int64,
    )

    return (
        candidate_shingles,
        candidate_offsets,
        corpus,
        document_offsets,
    )


def test_cuda_single_candidate_results():
    candidate, corpus, document_offsets = _single_inputs()

    intersections, unions, scores = _cuda.jaccard_corpus_cuda(
        candidate,
        corpus,
        document_offsets,
    )

    np.testing.assert_array_equal(
        intersections,
        np.array([3, 1, 2, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        unions,
        np.array([4, 5, 6, 4], dtype=np.int64),
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [0.75, 0.20, 1.0 / 3.0, 0.0],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_cuda_single_output_contract():
    candidate, corpus, document_offsets = _single_inputs()

    intersections, unions, scores = _cuda.jaccard_corpus_cuda(
        candidate,
        corpus,
        document_offsets,
    )

    assert intersections.shape == (4,)
    assert unions.shape == (4,)
    assert scores.shape == (4,)

    assert intersections.dtype == np.int64
    assert unions.dtype == np.int64
    assert scores.dtype == np.float64

    assert intersections.flags.c_contiguous
    assert unions.flags.c_contiguous
    assert scores.flags.c_contiguous


def test_cuda_empty_candidate():
    _, corpus, document_offsets = _single_inputs()
    candidate = np.empty(0, dtype=np.int64)

    intersections, unions, scores = _cuda.jaccard_corpus_cuda(
        candidate,
        corpus,
        document_offsets,
    )

    np.testing.assert_array_equal(
        intersections,
        np.array([0, 0, 0, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        unions,
        np.array([3, 2, 4, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        scores,
        np.zeros(4, dtype=np.float64),
    )


def test_cuda_batch_results():
    (
        candidate_shingles,
        candidate_offsets,
        corpus,
        document_offsets,
    ) = _batch_inputs()

    intersections, unions, scores = (
        _cuda.jaccard_batch_corpus_cuda(
            candidate_shingles,
            candidate_offsets,
            corpus,
            document_offsets,
        )
    )

    expected_intersections = np.array(
        [
            [3, 1, 0],
            [1, 2, 0],
            [2, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
        order="F",
    )

    expected_unions = np.array(
        [
            [4, 4, 3],
            [5, 2, 2],
            [6, 6, 4],
            [4, 2, 0],
        ],
        dtype=np.int64,
        order="F",
    )

    expected_scores = np.array(
        [
            [0.75, 0.25, 0.0],
            [0.20, 1.00, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
        order="F",
    )

    np.testing.assert_array_equal(
        intersections,
        expected_intersections,
    )

    np.testing.assert_array_equal(
        unions,
        expected_unions,
    )

    np.testing.assert_allclose(
        scores,
        expected_scores,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_cuda_batch_output_contract():
    inputs = _batch_inputs()

    intersections, unions, scores = (
        _cuda.jaccard_batch_corpus_cuda(*inputs)
    )

    assert intersections.shape == (4, 3)
    assert unions.shape == (4, 3)
    assert scores.shape == (4, 3)

    assert intersections.dtype == np.int64
    assert unions.dtype == np.int64
    assert scores.dtype == np.float64

    assert intersections.flags.f_contiguous
    assert unions.flags.f_contiguous
    assert scores.flags.f_contiguous


def test_cuda_batch_size_one():
    candidate, corpus, document_offsets = _single_inputs()
    candidate_offsets = np.array([0, 4], dtype=np.int64)

    intersections, unions, scores = (
        _cuda.jaccard_batch_corpus_cuda(
            candidate,
            candidate_offsets,
            corpus,
            document_offsets,
        )
    )

    assert intersections.shape == (4, 1)
    assert unions.shape == (4, 1)
    assert scores.shape == (4, 1)

    np.testing.assert_array_equal(
        intersections[:, 0],
        np.array([3, 1, 2, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        unions[:, 0],
        np.array([4, 5, 6, 4], dtype=np.int64),
    )

    np.testing.assert_allclose(
        scores[:, 0],
        np.array(
            [0.75, 0.20, 1.0 / 3.0, 0.0],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1.0e-15,
    )


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        (
            np.array([1, 3, 5, 9], dtype=np.int64),
            "document_offsets must begin with zero",
        ),
        (
            np.array([0, 5, 3, 9], dtype=np.int64),
            "document_offsets must be nondecreasing",
        ),
        (
            np.array([0, 3, 5, 8], dtype=np.int64),
            "final document offset must equal corpus length",
        ),
    ],
)
def test_cuda_single_rejects_invalid_document_offsets(
    offsets,
    message,
):
    candidate, corpus, _ = _single_inputs()

    with pytest.raises(ValueError, match=message):
        _cuda.jaccard_corpus_cuda(
            candidate,
            corpus,
            offsets,
        )


def test_cuda_batch_rejects_invalid_candidate_offsets():
    (
        candidate_shingles,
        _,
        corpus,
        document_offsets,
    ) = _batch_inputs()

    invalid_offsets = np.array(
        [0, 4, 3, 6],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="candidate_offsets must be nondecreasing",
    ):
        _cuda.jaccard_batch_corpus_cuda(
            candidate_shingles,
            invalid_offsets,
            corpus,
            document_offsets,
        )


def test_cuda_batch_rejects_empty_candidate_offsets():
    _, _, corpus, document_offsets = _batch_inputs()

    with pytest.raises(
        ValueError,
        match="candidate_offsets must describe at least one candidate",
    ):
        _cuda.jaccard_batch_corpus_cuda(
            np.empty(0, dtype=np.int64),
            np.array([0], dtype=np.int64),
            corpus,
            document_offsets,
        )


def test_cuda_converts_compatible_integer_dtype():
    candidate, corpus, document_offsets = _single_inputs()

    intersections, unions, scores = _cuda.jaccard_corpus_cuda(
        candidate.astype(np.int32),
        corpus,
        document_offsets,
    )

    np.testing.assert_array_equal(
        intersections,
        np.array([3, 1, 2, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        unions,
        np.array([4, 5, 6, 4], dtype=np.int64),
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [0.75, 0.20, 1.0 / 3.0, 0.0],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_cuda_accepts_noncontiguous_input_via_conversion():
    candidate = np.array(
        [10, 99, 20, 99, 40, 99, 70, 99],
        dtype=np.int64,
    )[::2]

    assert not candidate.flags.c_contiguous
    np.testing.assert_array_equal(
        candidate,
        np.array([10, 20, 40, 70], dtype=np.int64),
    )

    _, corpus, document_offsets = _single_inputs()

    intersections, unions, scores = _cuda.jaccard_corpus_cuda(
        candidate,
        corpus,
        document_offsets,
    )

    np.testing.assert_array_equal(
        intersections,
        np.array([3, 1, 2, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        unions,
        np.array([4, 5, 6, 4], dtype=np.int64),
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [0.75, 0.20, 1.0 / 3.0, 0.0],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1.0e-15,
    )
