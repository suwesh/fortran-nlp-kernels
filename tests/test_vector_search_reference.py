import numpy as np

from fortran_nlp.vector_search.reference import (
    inner_product_scores,
    search_ip,
    search_l2,
    squared_l2_scores,
)


def test_numpy_vector_search_reference():
    query = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    # Python-facing layout:
    # shape = (vector_count, dimension)
    database = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
            [-1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    expected_l2_scores = np.array(
        [0.0, 27.0, 14.0, 8.0],
        dtype=np.float32,
    )

    expected_ip_scores = np.array(
        [14.0, 32.0, 0.0, 6.0],
        dtype=np.float32,
    )

    expected_l2_values = np.array(
        [0.0, 8.0],
        dtype=np.float32,
    )

    expected_l2_indices = np.array(
        [0, 3],
        dtype=np.int64,
    )

    expected_ip_values = np.array(
        [32.0, 14.0],
        dtype=np.float32,
    )

    expected_ip_indices = np.array(
        [1, 0],
        dtype=np.int64,
    )

    l2_scores = squared_l2_scores(
        query,
        database,
    )

    ip_scores = inner_product_scores(
        query,
        database,
    )

    l2_values, l2_indices = search_l2(
        query,
        database,
        k=2,
    )

    ip_values, ip_indices = search_ip(
        query,
        database,
        k=2,
    )

    assert l2_scores.dtype == np.float32
    assert ip_scores.dtype == np.float32

    assert np.allclose(
        l2_scores,
        expected_l2_scores,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.allclose(
        ip_scores,
        expected_ip_scores,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.allclose(
        l2_values,
        expected_l2_values,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        l2_indices,
        expected_l2_indices,
    )

    assert np.allclose(
        ip_values,
        expected_ip_values,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        ip_indices,
        expected_ip_indices,
    )