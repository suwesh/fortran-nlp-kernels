from fortran_nlp.auto import (
    BackendDecision,
    get_shipped_policy,
)


def test_backend_decision_is_immutable():
    decision = BackendDecision(
        backend="fortran_openmp",
        strategy="openmp",
        source="fallback",
        reason="Test decision",
        thread_policy="physical_cores",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "openmp"
    assert decision.source == "fallback"
    assert decision.reason == "Test decision"
    assert decision.thread_policy == "physical_cores"

    try:
        decision.backend = "fortran"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "BackendDecision must be immutable"
        )


def test_shipped_policy_metadata():
    policy = get_shipped_policy()

    assert policy["schema_version"] == 1
    assert policy["policy_version"] == "cpu-fallback-v1"

    assert policy["package_compatibility"] == {
        "minimum_version": "0.1.0",
        "maximum_version_exclusive": "0.2.0",
    }


def test_shipped_policy_defaults():
    defaults = get_shipped_policy()["defaults"]

    assert defaults["openmp_thread_policy"] == "physical_cores"
    assert defaults["openblas_thread_policy"] == "single_thread"
    assert defaults["full_smt_automatic"] is False


def test_shipped_policy_contains_lexical_sections():
    lexical = get_shipped_policy()["lexical"]

    assert set(lexical) >= {
        "single",
        "batch",
        "memory",
    }

    assert lexical["single"]["small_work_strategy"] == "serial"
    assert lexical["single"]["large_work_strategy"] == "openmp"

    assert (
        lexical["batch"]["native_batch_strategy"]
        == "native_batch_openmp"
    )

    assert (
        lexical["memory"]["bytes_per_native_batch_result"]
        == 24
    )


def test_shipped_policy_contains_vector_metrics():
    vector = get_shipped_policy()["vector"]

    assert set(vector) >= {
        "ip",
        "cosine",
        "l2",
    }

    assert vector["ip"]["single"]["backend"] == "fortran_blas"
    assert vector["ip"]["batch"]["native_batch_min_queries"] == 8

    assert vector["cosine"]["policy"] == "normalized_ip"

    assert vector["l2"]["single"]["backend"] == "fortran_openmp"
    assert vector["l2"]["automatic_blas"] is False


def test_shipped_policy_returns_independent_copy():
    first = get_shipped_policy()
    second = get_shipped_policy()

    assert first is not second

    first["schema_version"] = 999
    first["lexical"]["single"]["small_work_strategy"] = "modified"

    third = get_shipped_policy()

    assert third["schema_version"] == 1
    assert (
        third["lexical"]["single"]["small_work_strategy"]
        == "serial"
    )