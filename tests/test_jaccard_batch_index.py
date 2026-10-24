import pytest

from fortran_nlp.lexical import (
    JaccardIndex,
    JaccardResult,
)


@pytest.fixture
def corpus_texts():
    return [
        "reset the account password immediately",
        "update the seller information today",
        "create a new loan application online",
        "reset the account password immediately",
    ]


@pytest.fixture
def candidate_texts():
    return [
        "reset the account password immediately",
        "update the seller information today",
        "unrelated candidate document words",
    ]


@pytest.mark.parametrize(
    "backend",
    [
        "fortran_openmp",
        "auto",
    ],
)
def test_search_batch_matches_repeated_search(
    corpus_texts,
    candidate_texts,
    backend,
):
    index = JaccardIndex(
        corpus_texts,
        backend=backend,
    )

    batch_results = index.search_batch(
        candidate_texts,
        k=3,
        minimum_score=0.0,
    )

    repeated_results = [
        index.search(
            candidate_text,
            k=3,
            minimum_score=0.0,
        )
        for candidate_text in candidate_texts
    ]

    assert batch_results == repeated_results


@pytest.mark.parametrize(
    "thread_count",
    [
        1,
        4,
    ],
)
def test_search_batch_returns_expected_matches(
    corpus_texts,
    candidate_texts,
    thread_count,
    monkeypatch,
):
    monkeypatch.setenv(
        "OMP_NUM_THREADS",
        str(thread_count),
    )

    index = JaccardIndex(
        corpus_texts,
        backend="fortran_openmp",
    )

    results = index.search_batch(
        candidate_texts,
        k=2,
        minimum_score=0.0,
    )

    assert len(results) == len(candidate_texts)

    assert all(
        len(candidate_results) == 2
        for candidate_results in results
    )

    assert all(
        isinstance(result, JaccardResult)
        for candidate_results in results
        for result in candidate_results
    )

    # Candidate 0 exactly matches duplicate corpus documents
    # at positional IDs 0 and 3.
    assert [
        result.document_id
        for result in results[0]
    ] == [0, 3]

    assert [
        result.score
        for result in results[0]
    ] == pytest.approx([1.0, 1.0])

    # Candidate 1 exactly matches document 1.
    assert results[1][0].document_id == 1
    assert results[1][0].score == pytest.approx(1.0)


def test_search_batch_applies_k_per_candidate(
    corpus_texts,
    candidate_texts,
):
    index = JaccardIndex(
        corpus_texts,
        backend="auto",
    )

    results = index.search_batch(
        candidate_texts,
        k=1,
    )

    assert len(results) == len(candidate_texts)

    assert all(
        len(candidate_results) == 1
        for candidate_results in results
    )


def test_search_batch_applies_minimum_score_per_candidate(
    corpus_texts,
    candidate_texts,
):
    index = JaccardIndex(
        corpus_texts,
        backend="auto",
    )

    results = index.search_batch(
        candidate_texts,
        minimum_score=1.0,
    )

    assert [
        result.document_id
        for result in results[0]
    ] == [0, 3]

    assert [
        result.document_id
        for result in results[1]
    ] == [1]

    assert results[2] == []


def test_search_batch_preserves_lower_ids_for_ties():
    corpus = [
        "alpha beta gamma delta",
        "alpha beta gamma delta",
        "alpha beta gamma delta",
    ]

    index = JaccardIndex(
        corpus,
        backend="auto",
    )

    results = index.search_batch(
        [
            "alpha beta gamma delta",
        ],
        k=3,
    )

    assert [
        result.document_id
        for result in results[0]
    ] == [0, 1, 2]


def test_search_batch_rejects_empty_candidate_sequence(
    corpus_texts,
):
    index = JaccardIndex(
        corpus_texts,
        backend="auto",
    )

    with pytest.raises(
        ValueError,
        match="must contain at least one candidate",
    ):
        index.search_batch([])


def test_search_batch_rejects_invalid_k(
    corpus_texts,
    candidate_texts,
):
    index = JaccardIndex(
        corpus_texts,
        backend="auto",
    )

    with pytest.raises(
        ValueError,
        match="k must be at least 1",
    ):
        index.search_batch(
            candidate_texts,
            k=0,
        )


@pytest.mark.parametrize(
    "minimum_score",
    [
        -0.01,
        1.01,
    ],
)
def test_search_batch_rejects_invalid_minimum_score(
    corpus_texts,
    candidate_texts,
    minimum_score,
):
    index = JaccardIndex(
        corpus_texts,
        backend="auto",
    )

    with pytest.raises(
        ValueError,
        match="minimum_score must be between 0 and 1",
    ):
        index.search_batch(
            candidate_texts,
            minimum_score=minimum_score,
        )


def test_search_batch_supports_serial_backend(
    corpus_texts,
    candidate_texts,
):
    serial_index = JaccardIndex(
        corpus_texts,
        backend="fortran",
    )

    openmp_index = JaccardIndex(
        corpus_texts,
        backend="fortran_openmp",
    )

    serial_results = serial_index.search_batch(
        candidate_texts,
        k=3,
    )

    openmp_results = openmp_index.search_batch(
        candidate_texts,
        k=3,
    )

    assert serial_results == openmp_results


def test_search_batch_returns_independent_result_lists(
    corpus_texts,
    candidate_texts,
):
    index = JaccardIndex(
        corpus_texts,
        backend="auto",
    )

    results = index.search_batch(
        candidate_texts,
        k=2,
    )

    assert results[0] is not results[1]
    assert results[1] is not results[2]