import pytest

import fortran_nlp.lexical.index as lexical_index_module
from fortran_nlp.lexical import JaccardIndex, JaccardResult


CORPUS = [
    "reset the account password",
    "update contact information",
    "create a new insurance application",
    "reset password instructions",
]


def _result_signature(results):
    return [
        (
            result.document_id,
            result.score,
            result.intersection_count,
            result.union_count,
        )
        for result in results
    ]


def test_cuda_backend_is_accepted_and_retained():
    index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )

    assert index.backend == "cuda_fortran"
    assert index.document_count == len(CORPUS)


def test_cuda_single_search_matches_serial_cpu():
    cuda_index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )
    cpu_index = JaccardIndex(
        CORPUS,
        backend="fortran",
    )

    candidate = "how to reset the account password"

    cuda_results = cuda_index.search(candidate)
    cpu_results = cpu_index.search(candidate)

    assert _result_signature(cuda_results) == pytest.approx(
        _result_signature(cpu_results),
        abs=1.0e-15,
        rel=0.0,
    )


def test_cuda_batch_search_matches_serial_cpu():
    cuda_index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )
    cpu_index = JaccardIndex(
        CORPUS,
        backend="fortran",
    )

    candidates = [
        "how to reset the account password",
        "contact information update",
        "",
    ]

    cuda_results = cuda_index.search_batch(candidates)
    cpu_results = cpu_index.search_batch(candidates)

    assert len(cuda_results) == len(cpu_results)

    for cuda_candidate, cpu_candidate in zip(
        cuda_results,
        cpu_results,
        strict=True,
    ):
        assert _result_signature(cuda_candidate) == pytest.approx(
            _result_signature(cpu_candidate),
            abs=1.0e-15,
            rel=0.0,
        )


def test_cuda_search_respects_k():
    index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )

    results = index.search(
        "reset account password",
        k=2,
    )

    assert len(results) == 2
    assert all(isinstance(result, JaccardResult) for result in results)


def test_cuda_search_respects_minimum_score():
    index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )

    results = index.search(
        "reset the account password",
        minimum_score=0.40,
    )

    assert results
    assert all(result.score >= 0.40 for result in results)


def test_cuda_ties_preserve_document_order():
    index = JaccardIndex(
        [
            "alpha beta",
            "alpha beta",
            "unrelated words",
        ],
        backend="cuda_fortran",
        shingle_size=1,
    )

    results = index.search(
        "alpha beta",
        minimum_score=1.0,
    )

    assert [result.document_id for result in results] == [0, 1]
    assert [result.score for result in results] == [1.0, 1.0]


def test_cuda_empty_candidate_matches_serial_cpu():
    cuda_index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )
    cpu_index = JaccardIndex(
        CORPUS,
        backend="fortran",
    )

    cuda_results = cuda_index.search("")
    cpu_results = cpu_index.search("")

    assert _result_signature(cuda_results) == pytest.approx(
        _result_signature(cpu_results),
        abs=1.0e-15,
        rel=0.0,
    )


def test_cuda_batch_size_one_matches_single_search():
    index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )

    candidate = "reset the account password"

    single_results = index.search(candidate)
    batch_results = index.search_batch([candidate])

    assert len(batch_results) == 1
    assert _result_signature(batch_results[0]) == pytest.approx(
        _result_signature(single_results),
        abs=1.0e-15,
        rel=0.0,
    )


def test_cuda_backend_raises_clear_error_when_extension_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        lexical_index_module,
        "_cuda",
        None,
    )

    index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )

    with pytest.raises(
        RuntimeError,
        match="backend='cuda_fortran' requires the optional",
    ):
        index.search("reset account password")


def test_cuda_batch_validates_public_arguments():
    index = JaccardIndex(
        CORPUS,
        backend="cuda_fortran",
    )

    with pytest.raises(
        ValueError,
        match="candidate_texts must contain at least one candidate",
    ):
        index.search_batch([])

    with pytest.raises(ValueError, match="k must be at least 1"):
        index.search_batch(["reset password"], k=0)

    with pytest.raises(
        ValueError,
        match="minimum_score must be between 0 and 1",
    ):
        index.search_batch(
            ["reset password"],
            minimum_score=1.1,
        )
