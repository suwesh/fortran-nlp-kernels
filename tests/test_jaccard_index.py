import numpy as np

from fortran_nlp.lexical import JaccardIndex


def test_serial_and_openmp_indexes_match():
    corpus = [
        "reset the account password",
        "update seller information",
        "create a new loan application",
    ]

    serial_index = JaccardIndex(
        corpus,
        backend="fortran",
    )

    openmp_index = JaccardIndex(
        corpus,
        backend="fortran_openmp",
    )

    query = "how to reset the account password"

    serial_results = serial_index.search(
        query,
        minimum_score=0.01,
    )

    openmp_results = openmp_index.search(
        query,
        minimum_score=0.01,
    )

    assert len(serial_results) == 1
    assert len(openmp_results) == 1

    assert serial_results[0].document_id == 0
    assert serial_results[0].intersection_count == 2
    assert serial_results[0].union_count == 4
    assert np.isclose(serial_results[0].score, 0.5)

    assert serial_results == openmp_results