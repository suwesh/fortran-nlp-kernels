import numpy as np
import pytest

from fortran_nlp.vector_search import (
    FlatIndexCosine,
    FlatIndexIP,
    FlatIndexL2,
)


INDEX_CLASSES = [
    FlatIndexIP,
    FlatIndexL2,
    FlatIndexCosine,
]


@pytest.fixture
def database():
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
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


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
def test_empty_database_is_rejected(index_class):
    empty_database = np.empty(
        (0, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least one vector",
    ):
        index_class(empty_database)


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
def test_zero_dimension_database_is_rejected(index_class):
    empty_dimension_database = np.empty(
        (3, 0),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        index_class(empty_dimension_database)


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
def test_database_must_be_two_dimensional(index_class):
    invalid_database = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="must be two-dimensional",
    ):
        index_class(invalid_database)


@pytest.mark.parametrize(
    "index_class",
    INDEX_CLASSES,
)
def test_invalid_backend_is_rejected(
    database,
    index_class,
):
    with pytest.raises(
        ValueError,
        match="Unsupported backend",
    ):
        index_class(
            database,
            backend="invalid",
        )


@pytest.mark.parametrize(
    "index_class",
    INDEX_CLASSES,
)
def test_float64_inputs_are_converted_to_float32(
    index_class,
):
    database = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )

    query = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    index = index_class(database)

    values, indices = index.search(
        query,
        k=1,
    )

    assert values.dtype == np.float32
    assert indices.dtype == np.int64


@pytest.mark.parametrize(
    "index_class",
    INDEX_CLASSES,
)
def test_noncontiguous_inputs_are_supported(
    index_class,
):
    source_database = np.arange(
        30,
        dtype=np.float32,
    ).reshape(5, 6)

    database = source_database[:, ::2]

    source_query = np.arange(
        6,
        dtype=np.float32,
    )

    query = source_query[::2]

    assert not database.flags.c_contiguous
    assert not query.flags.c_contiguous

    index = index_class(database)

    values, indices = index.search(
        query,
        k=2,
    )

    assert values.shape == (2,)
    assert indices.shape == (2,)


@pytest.mark.parametrize(
    "index_class",
    INDEX_CLASSES,
)
def test_query_must_be_one_dimensional(
    database,
    index_class,
):
    invalid_query = np.array(
        [[1.0, 2.0, 3.0]],
        dtype=np.float32,
    )

    index = index_class(database)

    with pytest.raises(
        ValueError,
        match="must be one-dimensional",
    ):
        index.search(
            invalid_query,
            k=1,
        )


def test_cosine_database_rejects_zero_vector():
    database = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="zero vectors",
    ):
        FlatIndexCosine(database)


def test_cosine_query_rejects_zero_vector(database):
    index = FlatIndexCosine(database)

    zero_query = np.zeros(
        3,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="zero vectors cannot be normalized",
    ):
        index.search(
            zero_query,
            k=1,
        )


def test_empty_query_batch_is_rejected(database):
    index = FlatIndexIP(
        database,
        backend="fortran_blas",
    )

    empty_batch = np.empty(
        (0, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least one vector",
    ):
        index.search_batch(
            empty_batch,
            k=1,
        )


def test_query_batch_must_be_two_dimensional(database):
    index = FlatIndexIP(
        database,
        backend="fortran_blas",
    )

    invalid_batch = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="must be two-dimensional",
    ):
        index.search_batch(
            invalid_batch,
            k=1,
        )


@pytest.mark.parametrize(
    "backend",
    [
        "fortran",
        "fortran_openmp",
    ],
)
@pytest.mark.parametrize(
    "index_class",
    INDEX_CLASSES,
)
def test_batch_search_requires_blas_backend(
    database,
    index_class,
    backend,
):
    index = index_class(
        database,
        backend=backend,
    )

    query_batch = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        NotImplementedError,
        match="requires.*fortran_blas",
    ):
        index.search_batch(
            query_batch,
            k=1,
        )


@pytest.mark.parametrize(
    "index_class",
    [
        FlatIndexIP,
        FlatIndexL2,
    ],
)
def test_ties_use_lower_positional_id(
    index_class,
):
    database = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    index = index_class(
        database,
        backend="fortran_blas",
    )

    _, indices = index.search(
        query,
        k=2,
    )

    assert np.array_equal(
        indices,
        np.array([0, 1], dtype=np.int64),
    )