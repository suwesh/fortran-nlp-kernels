import numpy as np
import pytest

from fortran_nlp import _cpu, _cuda


SINGLE_BACKENDS = (
    (_cuda.l2_search_cuda, _cpu.l2_search_serial, "l2"),
    (_cuda.ip_search_cuda, _cpu.ip_search_serial, "ip"),
    (_cuda.l2_search_cublas, _cpu.l2_search_serial, "l2"),
    (_cuda.ip_search_cublas, _cpu.ip_search_serial, "ip"),
)

BATCH_BACKENDS = (
    (_cuda.l2_search_batch_cuda, _cpu.l2_search_serial, "l2"),
    (_cuda.ip_search_batch_cuda, _cpu.ip_search_serial, "ip"),
    (_cuda.l2_search_batch_cublas, _cpu.l2_search_serial, "l2"),
    (_cuda.ip_search_batch_cublas, _cpu.ip_search_serial, "ip"),
)


def _single_inputs():
    query = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    # Native layout: (dimension, vector_count).
    database = np.asfortranarray(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 3.0],
                [1.0, 4.0, 3.0],
                [0.0, 0.0, 0.0],
                [-1.0, -2.0, -3.0],
            ],
            dtype=np.float32,
        ).T
    )

    return query, database


def _batch_inputs():
    # Native layout: (dimension, query_count).
    queries = np.asfortranarray(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, -1.0, 0.5],
                [-2.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        ).T
    )

    _, database = _single_inputs()
    return queries, database


def _repeated_cpu_reference(
    queries,
    database,
    k,
    cpu_function,
):
    values = []
    indices = []

    for query_index in range(queries.shape[1]):
        query = np.ascontiguousarray(
            queries[:, query_index],
            dtype=np.float32,
        )
        query_values, query_indices = cpu_function(
            query,
            database,
            k,
        )
        values.append(query_values)
        indices.append(query_indices)

    return (
        np.asfortranarray(np.stack(values, axis=1)),
        np.asfortranarray(np.stack(indices, axis=1)),
    )


@pytest.mark.parametrize(
    ("cuda_function", "cpu_function", "metric"),
    SINGLE_BACKENDS,
)
def test_cuda_vector_single_matches_cpu(
    cuda_function,
    cpu_function,
    metric,
):
    query, database = _single_inputs()

    actual_values, actual_indices = cuda_function(
        query,
        database,
        3,
    )
    expected_values, expected_indices = cpu_function(
        query,
        database,
        3,
    )

    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(
        actual_indices,
        expected_indices,
    )

    assert metric in {"l2", "ip"}


@pytest.mark.parametrize(
    ("cuda_function", "cpu_function", "metric"),
    BATCH_BACKENDS,
)
def test_cuda_vector_batch_matches_repeated_cpu(
    cuda_function,
    cpu_function,
    metric,
):
    queries, database = _batch_inputs()

    actual_values, actual_indices = cuda_function(
        queries,
        database,
        3,
    )
    expected_values, expected_indices = _repeated_cpu_reference(
        queries,
        database,
        3,
        cpu_function,
    )

    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    np.testing.assert_array_equal(
        actual_indices,
        expected_indices,
    )

    assert metric in {"l2", "ip"}


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_cuda,
        _cuda.ip_search_cuda,
        _cuda.l2_search_cublas,
        _cuda.ip_search_cublas,
    ],
)
def test_cuda_vector_single_output_contract(cuda_function):
    query, database = _single_inputs()
    values, indices = cuda_function(query, database, 3)

    assert values.shape == (3,)
    assert indices.shape == (3,)
    assert values.dtype == np.float32
    assert indices.dtype == np.int64
    assert values.flags.c_contiguous
    assert indices.flags.c_contiguous


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_batch_cuda,
        _cuda.ip_search_batch_cuda,
        _cuda.l2_search_batch_cublas,
        _cuda.ip_search_batch_cublas,
    ],
)
def test_cuda_vector_batch_output_contract(cuda_function):
    queries, database = _batch_inputs()
    values, indices = cuda_function(queries, database, 3)

    assert values.shape == (3, 3)
    assert indices.shape == (3, 3)
    assert values.dtype == np.float32
    assert indices.dtype == np.int64
    assert values.flags.f_contiguous
    assert indices.flags.f_contiguous


@pytest.mark.parametrize(
    ("single_function", "batch_function"),
    [
        (_cuda.l2_search_cuda, _cuda.l2_search_batch_cuda),
        (_cuda.ip_search_cuda, _cuda.ip_search_batch_cuda),
        (_cuda.l2_search_cublas, _cuda.l2_search_batch_cublas),
        (_cuda.ip_search_cublas, _cuda.ip_search_batch_cublas),
    ],
)
def test_cuda_vector_batch_size_one_matches_single(
    single_function,
    batch_function,
):
    query, database = _single_inputs()
    queries = np.asfortranarray(query[:, None])

    single_values, single_indices = single_function(
        query,
        database,
        3,
    )
    batch_values, batch_indices = batch_function(
        queries,
        database,
        3,
    )

    np.testing.assert_allclose(
        batch_values[:, 0],
        single_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(
        batch_indices[:, 0],
        single_indices,
    )


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_cuda,
        _cuda.ip_search_cuda,
        _cuda.l2_search_cublas,
        _cuda.ip_search_cublas,
    ],
)
def test_cuda_vector_single_deterministic_ties(cuda_function):
    query = np.array([1.0, 1.0], dtype=np.float32)
    database = np.asfortranarray(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ).T
    )

    values, indices = cuda_function(query, database, 4)

    np.testing.assert_allclose(
        values,
        np.ones(4, dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        indices,
        np.array([0, 1, 2, 3], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_batch_cuda,
        _cuda.ip_search_batch_cuda,
        _cuda.l2_search_batch_cublas,
        _cuda.ip_search_batch_cublas,
    ],
)
def test_cuda_vector_batch_deterministic_ties(cuda_function):
    queries = np.asfortranarray(
        np.array(
            [
                [1.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=np.float32,
        ).T
    )
    database = np.asfortranarray(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ).T
    )

    _, indices = cuda_function(queries, database, 4)

    expected = np.asfortranarray(
        np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            dtype=np.int64,
        )
    )
    np.testing.assert_array_equal(indices, expected)


@pytest.mark.parametrize(
    ("cuda_function", "cublas_function"),
    [
        (_cuda.l2_search_cuda, _cuda.l2_search_cublas),
        (_cuda.ip_search_cuda, _cuda.ip_search_cublas),
    ],
)
def test_handwritten_cuda_matches_cublas_single(
    cuda_function,
    cublas_function,
):
    query, database = _single_inputs()

    cuda_values, cuda_indices = cuda_function(query, database, 5)
    cublas_values, cublas_indices = cublas_function(
        query,
        database,
        5,
    )

    np.testing.assert_allclose(
        cuda_values,
        cublas_values,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    np.testing.assert_array_equal(cuda_indices, cublas_indices)


@pytest.mark.parametrize(
    ("cuda_function", "cublas_function"),
    [
        (
            _cuda.l2_search_batch_cuda,
            _cuda.l2_search_batch_cublas,
        ),
        (
            _cuda.ip_search_batch_cuda,
            _cuda.ip_search_batch_cublas,
        ),
    ],
)
def test_handwritten_cuda_matches_cublas_batch(
    cuda_function,
    cublas_function,
):
    queries, database = _batch_inputs()

    cuda_values, cuda_indices = cuda_function(
        queries,
        database,
        5,
    )
    cublas_values, cublas_indices = cublas_function(
        queries,
        database,
        5,
    )

    np.testing.assert_allclose(
        cuda_values,
        cublas_values,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    np.testing.assert_array_equal(cuda_indices, cublas_indices)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.ip_search_cuda,
        _cuda.ip_search_cublas,
    ],
)
def test_cuda_vector_normalized_single_ip_matches_cosine_reference(
    cuda_function,
):
    database_rows = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.5],
            [0.5, -2.0, 4.0],
        ],
        dtype=np.float32,
    )
    query = np.array([2.0, 1.0, 3.0], dtype=np.float32)

    normalized_database_rows = database_rows / np.linalg.norm(
        database_rows,
        axis=1,
        keepdims=True,
    )
    normalized_query = query / np.linalg.norm(query)

    database = np.asfortranarray(
        normalized_database_rows.T,
        dtype=np.float32,
    )
    normalized_query = np.ascontiguousarray(
        normalized_query,
        dtype=np.float32,
    )

    values, indices = cuda_function(
        normalized_query,
        database,
        4,
    )

    reference = normalized_database_rows @ normalized_query
    expected_indices = np.argsort(
        -reference,
        kind="stable",
    ).astype(np.int64)
    expected_values = reference[expected_indices]

    np.testing.assert_allclose(
        values,
        expected_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(indices, expected_indices)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.ip_search_batch_cuda,
        _cuda.ip_search_batch_cublas,
    ],
)
def test_cuda_vector_normalized_batch_ip_matches_cosine_reference(
    cuda_function,
):
    database_rows = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.5],
            [0.5, -2.0, 4.0],
        ],
        dtype=np.float32,
    )
    query_rows = np.array(
        [
            [2.0, 1.0, 3.0],
            [-1.0, 2.0, 0.5],
        ],
        dtype=np.float32,
    )

    normalized_database_rows = database_rows / np.linalg.norm(
        database_rows,
        axis=1,
        keepdims=True,
    )
    normalized_query_rows = query_rows / np.linalg.norm(
        query_rows,
        axis=1,
        keepdims=True,
    )

    database = np.asfortranarray(
        normalized_database_rows.T,
        dtype=np.float32,
    )
    queries = np.asfortranarray(
        normalized_query_rows.T,
        dtype=np.float32,
    )

    values, indices = cuda_function(queries, database, 4)

    reference = normalized_database_rows @ normalized_query_rows.T
    expected_indices = np.argsort(
        -reference,
        axis=0,
        kind="stable",
    ).astype(np.int64)
    expected_values = np.take_along_axis(
        reference,
        expected_indices,
        axis=0,
    )

    np.testing.assert_allclose(
        values,
        expected_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(indices, expected_indices)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_cuda,
        _cuda.ip_search_cuda,
        _cuda.l2_search_cublas,
        _cuda.ip_search_cublas,
    ],
)
@pytest.mark.parametrize("k", [0, -1, 6])
def test_cuda_vector_single_rejects_invalid_k(cuda_function, k):
    query, database = _single_inputs()

    message = (
        "k must be at least 1"
        if k < 1
        else "k cannot exceed the database size"
    )

    with pytest.raises(ValueError, match=message):
        cuda_function(query, database, k)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_batch_cuda,
        _cuda.ip_search_batch_cuda,
        _cuda.l2_search_batch_cublas,
        _cuda.ip_search_batch_cublas,
    ],
)
@pytest.mark.parametrize("k", [0, -1, 6])
def test_cuda_vector_batch_rejects_invalid_k(cuda_function, k):
    queries, database = _batch_inputs()

    message = (
        "k must be at least 1"
        if k < 1
        else "k cannot exceed the database size"
    )

    with pytest.raises(ValueError, match=message):
        cuda_function(queries, database, k)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_cuda,
        _cuda.ip_search_cuda,
        _cuda.l2_search_cublas,
        _cuda.ip_search_cublas,
    ],
)
def test_cuda_vector_single_rejects_dimension_mismatch(cuda_function):
    query, database = _single_inputs()
    invalid_query = query[:2]

    with pytest.raises(
        ValueError,
        match="query and database vectors must have equal dimensions",
    ):
        cuda_function(invalid_query, database, 1)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_batch_cuda,
        _cuda.ip_search_batch_cuda,
        _cuda.l2_search_batch_cublas,
        _cuda.ip_search_batch_cublas,
    ],
)
def test_cuda_vector_batch_rejects_dimension_mismatch(cuda_function):
    queries, database = _batch_inputs()
    invalid_queries = np.asfortranarray(queries[:2, :])

    with pytest.raises(
        ValueError,
        match="queries and database vectors must have equal dimensions",
    ):
        cuda_function(invalid_queries, database, 1)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_cuda,
        _cuda.ip_search_cuda,
        _cuda.l2_search_cublas,
        _cuda.ip_search_cublas,
    ],
)
def test_cuda_vector_single_rejects_empty_database(cuda_function):
    query, _ = _single_inputs()
    database = np.empty((query.size, 0), dtype=np.float32, order="F")

    with pytest.raises(
        ValueError,
        match="database must contain at least one vector",
    ):
        cuda_function(query, database, 1)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_batch_cuda,
        _cuda.ip_search_batch_cuda,
        _cuda.l2_search_batch_cublas,
        _cuda.ip_search_batch_cublas,
    ],
)
def test_cuda_vector_batch_rejects_empty_query_batch(cuda_function):
    _, database = _batch_inputs()
    queries = np.empty(
        (database.shape[0], 0),
        dtype=np.float32,
        order="F",
    )

    with pytest.raises(
        ValueError,
        match="query batch must contain at least one vector",
    ):
        cuda_function(queries, database, 1)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_cuda,
        _cuda.ip_search_cuda,
        _cuda.l2_search_cublas,
        _cuda.ip_search_cublas,
    ],
)
def test_cuda_vector_converts_compatible_float_dtype(cuda_function):
    query, database = _single_inputs()

    reference_values, reference_indices = cuda_function(
        query,
        database,
        3,
    )
    values, indices = cuda_function(
        query.astype(np.float64),
        database.astype(np.float64, order="F"),
        3,
    )

    np.testing.assert_allclose(
        values,
        reference_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(indices, reference_indices)


@pytest.mark.parametrize(
    "cuda_function",
    [
        _cuda.l2_search_batch_cuda,
        _cuda.ip_search_batch_cuda,
        _cuda.l2_search_batch_cublas,
        _cuda.ip_search_batch_cublas,
    ],
)
def test_cuda_vector_accepts_c_contiguous_matrices_via_conversion(
    cuda_function,
):
    queries, database = _batch_inputs()

    reference_values, reference_indices = cuda_function(
        queries,
        database,
        3,
    )

    c_queries = np.ascontiguousarray(queries)
    c_database = np.ascontiguousarray(database)
    assert c_queries.flags.c_contiguous
    assert c_database.flags.c_contiguous

    values, indices = cuda_function(
        c_queries,
        c_database,
        3,
    )

    np.testing.assert_allclose(
        values,
        reference_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(indices, reference_indices)
