import numpy as np
import pytest

from fortran_nlp import _cpu


@pytest.fixture
def packed_candidates():
    # Candidate 0 = [1, 2, 3]
    # Candidate 1 = [2, 4]
    candidate_shingles = np.array(
        [
            1,
            2,
            3,
            2,
            4,
        ],
        dtype=np.int64,
    )

    candidate_offsets = np.array(
        [
            0,
            3,
            5,
        ],
        dtype=np.int64,
    )

    return candidate_shingles, candidate_offsets


@pytest.fixture
def packed_corpus():
    # Document 0 = [1, 2, 3]
    # Document 1 = [2, 3, 4]
    # Document 2 = [5]
    corpus_shingles = np.array(
        [
            1,
            2,
            3,
            2,
            3,
            4,
            5,
        ],
        dtype=np.int64,
    )

    document_offsets = np.array(
        [
            0,
            3,
            6,
            7,
        ],
        dtype=np.int64,
    )

    return corpus_shingles, document_offsets


def test_native_jaccard_batch_correctness(
    packed_candidates,
    packed_corpus,
):
    candidate_shingles, candidate_offsets = (
        packed_candidates
    )

    corpus_shingles, document_offsets = packed_corpus

    (
        intersections,
        unions,
        scores,
    ) = _cpu.jaccard_batch_corpus(
        candidate_shingles,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    )

    # Native matrix layout:
    # rows    = corpus documents
    # columns = candidate documents
    expected_intersections = np.array(
        [
            [3, 1],
            [2, 2],
            [0, 0],
        ],
        dtype=np.int64,
        order="F",
    )

    expected_unions = np.array(
        [
            [3, 4],
            [4, 3],
            [4, 3],
        ],
        dtype=np.int64,
        order="F",
    )

    expected_scores = np.array(
        [
            [1.0, 0.25],
            [0.5, 2.0 / 3.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
        order="F",
    )

    assert intersections.shape == (3, 2)
    assert unions.shape == (3, 2)
    assert scores.shape == (3, 2)

    assert intersections.dtype == np.int64
    assert unions.dtype == np.int64
    assert scores.dtype == np.float64

    assert intersections.flags.f_contiguous
    assert unions.flags.f_contiguous
    assert scores.flags.f_contiguous

    assert np.array_equal(
        intersections,
        expected_intersections,
    )

    assert np.array_equal(
        unions,
        expected_unions,
    )

    assert np.allclose(
        scores,
        expected_scores,
        rtol=0.0,
        atol=1e-12,
    )


def test_native_batch_matches_repeated_serial_search(
    packed_candidates,
    packed_corpus,
):
    candidate_shingles, candidate_offsets = (
        packed_candidates
    )

    corpus_shingles, document_offsets = packed_corpus

    (
        batch_intersections,
        batch_unions,
        batch_scores,
    ) = _cpu.jaccard_batch_corpus(
        candidate_shingles,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    )

    candidate_count = len(candidate_offsets) - 1

    for candidate_index in range(candidate_count):
        start = candidate_offsets[candidate_index]
        end = candidate_offsets[candidate_index + 1]

        candidate = np.ascontiguousarray(
            candidate_shingles[start:end],
            dtype=np.int64,
        )

        (
            serial_intersections,
            serial_unions,
            serial_scores,
        ) = _cpu.jaccard_corpus_serial(
            candidate,
            corpus_shingles,
            document_offsets,
        )

        assert np.array_equal(
            batch_intersections[:, candidate_index],
            serial_intersections,
        )

        assert np.array_equal(
            batch_unions[:, candidate_index],
            serial_unions,
        )

        assert np.allclose(
            batch_scores[:, candidate_index],
            serial_scores,
            rtol=0.0,
            atol=1e-12,
        )


def test_native_batch_candidate_offsets_must_begin_with_zero(
    packed_corpus,
):
    corpus_shingles, document_offsets = packed_corpus

    candidate_shingles = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    invalid_offsets = np.array(
        [1, 3],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="candidate_offsets must begin with zero",
    ):
        _cpu.jaccard_batch_corpus(
            candidate_shingles,
            invalid_offsets,
            corpus_shingles,
            document_offsets,
        )


def test_native_batch_candidate_offsets_must_be_nondecreasing(
    packed_corpus,
):
    corpus_shingles, document_offsets = packed_corpus

    candidate_shingles = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    invalid_offsets = np.array(
        [0, 3, 2, 3],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="candidate_offsets must be nondecreasing",
    ):
        _cpu.jaccard_batch_corpus(
            candidate_shingles,
            invalid_offsets,
            corpus_shingles,
            document_offsets,
        )


def test_native_batch_final_candidate_offset_matches_length(
    packed_corpus,
):
    corpus_shingles, document_offsets = packed_corpus

    candidate_shingles = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    invalid_offsets = np.array(
        [0, 2],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "final candidate offset must equal "
            "candidate shingles length"
        ),
    ):
        _cpu.jaccard_batch_corpus(
            candidate_shingles,
            invalid_offsets,
            corpus_shingles,
            document_offsets,
        )


def test_native_batch_requires_at_least_one_candidate(
    packed_corpus,
):
    corpus_shingles, document_offsets = packed_corpus

    candidate_shingles = np.empty(
        0,
        dtype=np.int64,
    )

    candidate_offsets = np.array(
        [0],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="must describe at least one candidate",
    ):
        _cpu.jaccard_batch_corpus(
            candidate_shingles,
            candidate_offsets,
            corpus_shingles,
            document_offsets,
        )