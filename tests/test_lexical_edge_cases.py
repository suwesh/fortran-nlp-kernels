import numpy as np
import pytest

from fortran_nlp import _cpu
from fortran_nlp.lexical import JaccardIndex


@pytest.mark.parametrize(
    "backend",
    [
        "fortran",
        "fortran_openmp",
    ],
)
def test_empty_candidate_returns_zero_scores(backend):
    index = JaccardIndex(
        [
            "reset the account password",
            "update seller information",
        ],
        backend=backend,
    )

    results = index.search("")

    assert len(results) == 2

    for result in results:
        assert result.score == 0.0
        assert result.intersection_count == 0
        assert result.union_count >= 0


@pytest.mark.parametrize(
    "backend",
    [
        "fortran",
        "fortran_openmp",
    ],
)
def test_text_shorter_than_shingle_size(backend):
    index = JaccardIndex(
        [
            "one two",
            "alpha beta",
        ],
        backend=backend,
        shingle_size=3,
    )

    results = index.search(
        "short query",
        minimum_score=0.0,
    )

    assert len(results) == 2

    for result in results:
        assert result.score == 0.0
        assert result.intersection_count == 0
        assert result.union_count == 0


@pytest.mark.parametrize(
    "backend",
    [
        "fortran",
        "fortran_openmp",
    ],
)
def test_identical_document_has_score_one(backend):
    text = "reset the account password immediately"

    index = JaccardIndex(
        [
            text,
            "update seller information today",
        ],
        backend=backend,
    )

    results = index.search(
        text,
        k=1,
    )

    assert len(results) == 1
    assert results[0].document_id == 0
    assert results[0].score == pytest.approx(1.0)
    assert (
        results[0].intersection_count
        == results[0].union_count
    )


@pytest.mark.parametrize(
    "backend",
    [
        "fortran",
        "fortran_openmp",
    ],
)
def test_duplicate_documents_produce_equal_scores(backend):
    duplicate = "reset the account password immediately"

    index = JaccardIndex(
        [
            duplicate,
            duplicate,
            "update seller information today",
        ],
        backend=backend,
    )

    results = index.search(
        duplicate,
        minimum_score=0.01,
    )

    matching_results = [
        result
        for result in results
        if result.score == pytest.approx(1.0)
    ]

    assert len(matching_results) == 2

    assert {
        result.document_id
        for result in matching_results
    } == {0, 1}


def test_invalid_backend_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported backend",
    ):
        JaccardIndex(
            ["reset the account password"],
            backend="invalid",
        )


def test_invalid_shingle_size_is_rejected():
    with pytest.raises(
        ValueError,
        match="shingle_size must be at least 1",
    ):
        JaccardIndex(
            ["reset the account password"],
            shingle_size=0,
        )


def test_invalid_k_is_rejected():
    index = JaccardIndex(
        ["reset the account password"],
    )

    with pytest.raises(
        ValueError,
        match="k must be at least 1",
    ):
        index.search(
            "reset the account password",
            k=0,
        )


@pytest.mark.parametrize(
    "minimum_score",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_minimum_score_is_rejected(
    minimum_score,
):
    index = JaccardIndex(
        ["reset the account password"],
    )

    with pytest.raises(
        ValueError,
        match="minimum_score must be between 0 and 1",
    ):
        index.search(
            "reset the account password",
            minimum_score=minimum_score,
        )


def test_native_offsets_must_begin_with_zero():
    candidate = np.array(
        [10, 20],
        dtype=np.int64,
    )

    corpus = np.array(
        [10, 20],
        dtype=np.int64,
    )

    invalid_offsets = np.array(
        [1, 2],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="must begin with zero",
    ):
        _cpu.jaccard_corpus_serial(
            candidate,
            corpus,
            invalid_offsets,
        )


def test_native_offsets_must_be_nondecreasing():
    candidate = np.array(
        [10, 20],
        dtype=np.int64,
    )

    corpus = np.array(
        [10, 20, 30],
        dtype=np.int64,
    )

    invalid_offsets = np.array(
        [0, 2, 1, 3],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="must be nondecreasing",
    ):
        _cpu.jaccard_corpus_serial(
            candidate,
            corpus,
            invalid_offsets,
        )


def test_final_native_offset_must_equal_corpus_length():
    candidate = np.array(
        [10, 20],
        dtype=np.int64,
    )

    corpus = np.array(
        [10, 20, 30],
        dtype=np.int64,
    )

    invalid_offsets = np.array(
        [0, 2],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="must equal corpus length",
    ):
        _cpu.jaccard_corpus_serial(
            candidate,
            corpus,
            invalid_offsets,
        )

def test_lexical_auto_selects_openmp():
    index = JaccardIndex(
        ["reset the account password"],
        backend="auto",
    )

    assert index.backend == "fortran_openmp"
