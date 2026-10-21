import numpy as np
import pytest

from fortran_nlp.vector_search import (
    FlatIndexCosine,
    FlatIndexIP,
    FlatIndexL2,
)

from fortran_nlp.vector_search.reference import (
    search_ip,
    search_l2,
)


BACKENDS = [
    "fortran",
    "fortran_openmp",
]


@pytest.fixture
def database():
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 1.0],
            [-1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def query():
    return np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_flat_index_ip_matches_numpy(
    database,
    query,
    backend,
):
    index = FlatIndexIP(
        database,
        backend=backend,
    )

    actual_values, actual_indices = index.search(
        query,
        k=2,
    )

    expected_values, expected_indices = search_ip(
        query,
        database,
        k=2,
    )

    assert actual_values.dtype == np.float32
    assert actual_indices.dtype == np.int64

    assert np.allclose(
        actual_values,
        expected_values,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        actual_indices,
        expected_indices,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_flat_index_l2_matches_numpy(
    database,
    query,
    backend,
):
    index = FlatIndexL2(
        database,
        backend=backend,
    )

    actual_values, actual_indices = index.search(
        query,
        k=2,
    )

    expected_values, expected_indices = search_l2(
        query,
        database,
        k=2,
    )

    assert actual_values.dtype == np.float32
    assert actual_indices.dtype == np.int64

    assert np.allclose(
        actual_values,
        expected_values,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        actual_indices,
        expected_indices,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_flat_index_cosine_matches_numpy(
    database,
    query,
    backend,
):
    index = FlatIndexCosine(
        database,
        backend=backend,
    )

    actual_values, actual_indices = index.search(
        query,
        k=2,
    )

    normalized_database = (
        database
        / np.linalg.norm(
            database,
            axis=1,
            keepdims=True,
        )
    ).astype(np.float32)

    normalized_query = (
        query / np.linalg.norm(query)
    ).astype(np.float32)

    expected_values, expected_indices = search_ip(
        normalized_query,
        normalized_database,
        k=2,
    )

    assert actual_values.dtype == np.float32
    assert actual_indices.dtype == np.int64

    assert np.allclose(
        actual_values,
        expected_values,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        actual_indices,
        expected_indices,
    )


@pytest.mark.parametrize(
    "index_class",
    [
        FlatIndexIP,
        FlatIndexL2,
        FlatIndexCosine,
    ],
)
def test_flat_indexes_validate_query_dimension(
    database,
    index_class,
):
    index = index_class(database)

    invalid_query = np.array(
        [1.0, 2.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="equal dimensions",
    ):
        index.search(
            invalid_query,
            k=1,
        )


@pytest.mark.parametrize(
    "index_class",
    [
        FlatIndexIP,
        FlatIndexL2,
        FlatIndexCosine,
    ],
)
def test_flat_indexes_validate_k(
    database,
    query,
    index_class,
):
    index = index_class(database)

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        index.search(
            query,
            k=0,
        )

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        index.search(
            query,
            k=len(database) + 1,
        )