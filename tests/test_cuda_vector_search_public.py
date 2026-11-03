import numpy as np
import pytest

import fortran_nlp.vector_search.flat as flat_module
from fortran_nlp.vector_search import (
    FlatIndexCosine,
    FlatIndexIP,
    FlatIndexL2,
)


CUDA_BACKENDS = (
    "cuda_fortran",
    "cuda_cublas",
)

INDEX_CLASSES = (
    FlatIndexL2,
    FlatIndexIP,
    FlatIndexCosine,
)

CPU_REFERENCE_BACKEND = "fortran"


def _database():
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [1.0, 4.0, 3.0],
            [0.0, 0.0, 1.0],
            [-1.0, -2.0, -3.0],
        ],
        dtype=np.float32,
    )


def _query():
    return np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )


def _queries():
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 0.5],
            [-2.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )


def _assert_result_equal(
    actual_values,
    actual_indices,
    expected_values,
    expected_indices,
):
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


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_single_matches_cpu(
    index_class,
    backend,
):
    database = _database()
    query = _query()

    cuda_index = index_class(database, backend=backend)
    cpu_index = index_class(
        database,
        backend=CPU_REFERENCE_BACKEND,
    )

    actual_values, actual_indices = cuda_index.search(
        query,
        k=4,
    )
    expected_values, expected_indices = cpu_index.search(
        query,
        k=4,
    )

    _assert_result_equal(
        actual_values,
        actual_indices,
        expected_values,
        expected_indices,
    )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_batch_matches_cpu(
    index_class,
    backend,
):
    database = _database()
    queries = _queries()

    cuda_index = index_class(database, backend=backend)
    cpu_index = index_class(
        database,
        backend=CPU_REFERENCE_BACKEND,
    )

    actual_values, actual_indices = cuda_index.search_batch(
        queries,
        k=4,
    )
    expected_values, expected_indices = cpu_index.search_batch(
        queries,
        k=4,
    )

    _assert_result_equal(
        actual_values,
        actual_indices,
        expected_values,
        expected_indices,
    )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_single_output_contract(
    index_class,
    backend,
):
    index = index_class(_database(), backend=backend)
    values, indices = index.search(_query(), k=3)

    assert values.shape == (3,)
    assert indices.shape == (3,)
    assert values.dtype == np.float32
    assert indices.dtype == np.int64
    assert values.flags.c_contiguous
    assert indices.flags.c_contiguous


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_batch_output_contract(
    index_class,
    backend,
):
    index = index_class(_database(), backend=backend)
    values, indices = index.search_batch(_queries(), k=3)

    assert values.shape == (3, 3)
    assert indices.shape == (3, 3)
    assert values.dtype == np.float32
    assert indices.dtype == np.int64
    assert values.flags.c_contiguous
    assert indices.flags.c_contiguous


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_batch_size_one_matches_single(
    index_class,
    backend,
):
    index = index_class(_database(), backend=backend)
    query = _query()

    single_values, single_indices = index.search(query, k=3)
    batch_values, batch_indices = index.search_batch(
        query[None, :],
        k=3,
    )

    assert batch_values.shape == (1, 3)
    assert batch_indices.shape == (1, 3)

    _assert_result_equal(
        batch_values[0],
        batch_indices[0],
        single_values,
        single_indices,
    )


@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_l2_known_results(backend):
    index = FlatIndexL2(_database(), backend=backend)
    values, indices = index.search(_query(), k=5)

    np.testing.assert_allclose(
        values,
        np.array(
            [0.0, 1.0, 4.0, 9.0, 56.0],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(
        indices,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
    )


@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_ip_known_results(backend):
    index = FlatIndexIP(_database(), backend=backend)
    values, indices = index.search(_query(), k=5)

    np.testing.assert_allclose(
        values,
        np.array(
            [18.0, 15.0, 14.0, 3.0, -14.0],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(
        indices,
        np.array([2, 1, 0, 3, 4], dtype=np.int64),
    )


@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cosine_matches_numpy_reference(backend):
    database = _database()
    query = _query()
    index = FlatIndexCosine(database, backend=backend)

    values, indices = index.search(query, k=database.shape[0])

    normalized_database = database / np.linalg.norm(
        database,
        axis=1,
        keepdims=True,
    )
    normalized_query = query / np.linalg.norm(query)
    reference = normalized_database @ normalized_query
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


@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cosine_batch_matches_numpy_reference(backend):
    database = _database()
    queries = _queries()
    index = FlatIndexCosine(database, backend=backend)

    values, indices = index.search_batch(
        queries,
        k=database.shape[0],
    )

    normalized_database = database / np.linalg.norm(
        database,
        axis=1,
        keepdims=True,
    )
    normalized_queries = queries / np.linalg.norm(
        queries,
        axis=1,
        keepdims=True,
    )
    reference = normalized_queries @ normalized_database.T
    expected_indices = np.argsort(
        -reference,
        axis=1,
        kind="stable",
    ).astype(np.int64)
    expected_values = np.take_along_axis(
        reference,
        expected_indices,
        axis=1,
    )

    np.testing.assert_allclose(
        values,
        expected_values,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(indices, expected_indices)


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_deterministic_ties(
    index_class,
    backend,
):
    query = np.array([1.0, 1.0], dtype=np.float32)
    database = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = index_class(database, backend=backend)
    _, indices = index.search(query, k=4)

    np.testing.assert_array_equal(
        indices,
        np.array([0, 1, 2, 3], dtype=np.int64),
    )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_properties(
    index_class,
    backend,
):
    database = _database()
    index = index_class(database, backend=backend)

    assert index.backend == backend
    assert index.vector_count == database.shape[0]
    assert index.dimension == database.shape[1]


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_accepts_float64_inputs(
    index_class,
    backend,
):
    database = _database()
    query = _query()

    reference_index = index_class(database, backend=backend)
    reference_values, reference_indices = reference_index.search(
        query,
        k=3,
    )

    converted_index = index_class(
        database.astype(np.float64),
        backend=backend,
    )
    values, indices = converted_index.search(
        query.astype(np.float64),
        k=3,
    )

    _assert_result_equal(
        values,
        indices,
        reference_values,
        reference_indices,
    )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
@pytest.mark.parametrize("k", [0, -1, 6])
def test_public_cuda_rejects_invalid_k(
    index_class,
    backend,
    k,
):
    index = index_class(_database(), backend=backend)

    message = (
        "k must be at least 1"
        if k < 1
        else "k cannot exceed the database size"
    )

    with pytest.raises(ValueError, match=message):
        index.search(_query(), k=k)

    with pytest.raises(ValueError, match=message):
        index.search_batch(_queries(), k=k)


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_rejects_single_dimension_mismatch(
    index_class,
    backend,
):
    index = index_class(_database(), backend=backend)

    with pytest.raises(
        ValueError,
        match="query and database vectors must have equal dimensions",
    ):
        index.search(
            np.array([1.0, 2.0], dtype=np.float32),
            k=1,
        )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_rejects_batch_dimension_mismatch(
    index_class,
    backend,
):
    index = index_class(_database(), backend=backend)

    with pytest.raises(
        ValueError,
        match="queries and database vectors must have equal dimensions",
    ):
        index.search_batch(
            np.ones((2, 2), dtype=np.float32),
            k=1,
        )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_rejects_empty_query_batch(
    index_class,
    backend,
):
    index = index_class(_database(), backend=backend)

    message = (
        "database must contain at least one vector"
        if index_class is FlatIndexCosine
        else "query batch must contain at least one vector"
    )

    with pytest.raises(ValueError, match=message):
        index.search_batch(
            np.empty((0, 3), dtype=np.float32),
            k=1,
        )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_rejects_non_matrix_database(
    index_class,
    backend,
):
    message = (
        "database must be two-dimensional"
        if index_class is FlatIndexCosine
        else "database_vectors must be two-dimensional"
    )

    with pytest.raises(ValueError, match=message):
        index_class(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            backend=backend,
        )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_rejects_empty_database(
    index_class,
    backend,
):
    with pytest.raises(
        ValueError,
        match="database must contain at least one vector",
    ):
        index_class(
            np.empty((0, 3), dtype=np.float32),
            backend=backend,
        )


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_rejects_zero_dimension_database(
    index_class,
    backend,
):
    with pytest.raises(
        ValueError,
        match="vector dimension must be greater than zero",
    ):
        index_class(
            np.empty((3, 0), dtype=np.float32),
            backend=backend,
        )


@pytest.mark.parametrize("backend", CUDA_BACKENDS)
def test_public_cuda_reports_missing_optional_extension(
    backend,
    monkeypatch,
):
    monkeypatch.setattr(flat_module, "_cuda", None)
    index = FlatIndexIP(_database(), backend=backend)

    with pytest.raises(
        RuntimeError,
        match=(
            rf"backend='{backend}' requires the optional "
            r"fortran_nlp\._cuda extension"
        ),
    ):
        index.search(_query(), k=1)


@pytest.mark.parametrize("index_class", INDEX_CLASSES)
def test_public_cuda_backend_names_are_supported(index_class):
    for backend in CUDA_BACKENDS:
        index = index_class(_database(), backend=backend)
        assert index.backend == backend


def test_public_auto_remains_cpu_resolved(monkeypatch):
    database = _database()
    query = _query()

    def fail_if_cuda_is_requested(_backend):
        raise AssertionError(
            "backend='auto' must not request the CUDA extension"
        )

    monkeypatch.setattr(
        flat_module,
        "_require_cuda_extension",
        fail_if_cuda_is_requested,
    )

    index = FlatIndexIP(database, backend="auto")
    values, indices = index.search(query, k=2)

    assert values.shape == (2,)
    assert indices.shape == (2,)
