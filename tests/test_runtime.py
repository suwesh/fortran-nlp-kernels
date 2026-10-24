import pytest

from fortran_nlp.runtime import (
    apply_openblas_thread_policy,
    apply_openmp_thread_policy,
    get_openblas_num_threads,
    get_openmp_max_threads,
    resolve_openblas_thread_policy,
    resolve_openmp_thread_policy,
)


def test_openmp_policy_resolution():
    assert resolve_openmp_thread_policy(None) is None
    assert resolve_openmp_thread_policy("single_thread") == 1
    assert resolve_openmp_thread_policy("physical_cores") >= 1
    assert resolve_openmp_thread_policy(2) == 2


def test_openblas_policy_resolution():
    assert resolve_openblas_thread_policy(None) is None
    assert resolve_openblas_thread_policy("single_thread") == 1
    assert resolve_openblas_thread_policy("physical_cores") >= 1
    assert resolve_openblas_thread_policy(2) == 2


@pytest.mark.parametrize(
    "resolver",
    [
        resolve_openmp_thread_policy,
        resolve_openblas_thread_policy,
    ],
)
def test_thread_policy_rejects_unsupported_value(
    resolver,
):
    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        resolver("unknown")


@pytest.mark.parametrize(
    "resolver",
    [
        resolve_openmp_thread_policy,
        resolve_openblas_thread_policy,
    ],
)
def test_thread_policy_rejects_invalid_integer(
    resolver,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        resolver(0)


def test_openmp_policy_restores_previous_count():
    original = get_openmp_max_threads()

    with apply_openmp_thread_policy(1):
        assert get_openmp_max_threads() == 1

    assert get_openmp_max_threads() == original


def test_openmp_policy_restores_after_exception():
    original = get_openmp_max_threads()

    with pytest.raises(
        RuntimeError,
        match="intentional failure",
    ):
        with apply_openmp_thread_policy(1):
            assert get_openmp_max_threads() == 1
            raise RuntimeError("intentional failure")

    assert get_openmp_max_threads() == original


def test_openblas_policy_restores_previous_count():
    original = get_openblas_num_threads()

    with apply_openblas_thread_policy(1):
        assert get_openblas_num_threads() == 1

    assert get_openblas_num_threads() == original


def test_openblas_policy_restores_after_exception():
    original = get_openblas_num_threads()

    with pytest.raises(
        RuntimeError,
        match="intentional failure",
    ):
        with apply_openblas_thread_policy(1):
            assert get_openblas_num_threads() == 1
            raise RuntimeError("intentional failure")

    assert get_openblas_num_threads() == original


def test_none_openmp_policy_preserves_count():
    original = get_openmp_max_threads()

    with apply_openmp_thread_policy(None) as selected:
        assert selected is None
        assert get_openmp_max_threads() == original

    assert get_openmp_max_threads() == original


def test_none_openblas_policy_preserves_count():
    original = get_openblas_num_threads()

    with apply_openblas_thread_policy(None) as selected:
        assert selected is None
        assert get_openblas_num_threads() == original

    assert get_openblas_num_threads() == original