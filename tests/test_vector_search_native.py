import numpy as np

from fortran_nlp import _cpu


def test_vector_search_serial_openmp_equivalence():
    query = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    # Shape: (dimension, vector_count)
    # Each column is one database vector.
    database = np.asfortranarray(
        [
            [1.0, 4.0, 0.0, -1.0],
            [2.0, 5.0, 0.0, 2.0],
            [3.0, 6.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    k = 2

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

    l2_serial = _cpu.l2_search_serial(
        query,
        database,
        k,
    )

    l2_openmp = _cpu.l2_search_openmp(
        query,
        database,
        k,
    )

    ip_serial = _cpu.ip_search_serial(
        query,
        database,
        k,
    )

    ip_openmp = _cpu.ip_search_openmp(
        query,
        database,
        k,
    )

    l2_blas = _cpu.l2_search_blas(
        query,
        database,
        k,
    )

    ip_blas = _cpu.ip_search_blas(
        query,
        database,
        k,
    )

    assert np.allclose(
        l2_serial[0],
        expected_l2_values,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        l2_serial[1],
        expected_l2_indices,
    )

    assert np.allclose(
        ip_serial[0],
        expected_ip_values,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        ip_serial[1],
        expected_ip_indices,
    )

    assert np.allclose(
        l2_serial[0],
        l2_openmp[0],
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        l2_serial[1],
        l2_openmp[1],
    )

    assert np.allclose(
        ip_serial[0],
        ip_openmp[0],
        rtol=0.0,
        atol=1e-6,
    )

    assert np.array_equal(
        ip_serial[1],
        ip_openmp[1],
    )
    
    assert np.allclose(
        l2_blas[0],
        l2_serial[0],
        rtol=0.0,
        atol=1e-5,
    )

    assert np.array_equal(
        l2_blas[1],
        l2_serial[1],
    )

    assert np.allclose(
        ip_blas[0],
        ip_serial[0],
        rtol=0.0,
        atol=1e-5,
    )

    assert np.array_equal(
        ip_blas[1],
        ip_serial[1],
    )