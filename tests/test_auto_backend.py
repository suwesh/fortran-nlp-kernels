from fortran_nlp.auto import (
    BackendDecision,
    PolicySelection,
    get_shipped_policy,
    select_dispatch_policy,
    VectorWorkload,
    resolve_vector_backend,
)
import pytest

from fortran_nlp.auto import (
    LexicalWorkload,
    resolve_lexical_backend,
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

def test_explicit_backend_skips_policy_loading(
    monkeypatch,
):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError(
            "Explicit backend must not load a machine profile"
        )

    monkeypatch.setattr(
        "fortran_nlp.auto.load_profile",
        fail_if_loaded,
    )

    selection = select_dispatch_policy(
        "fortran_openmp"
    )

    assert isinstance(selection, PolicySelection)
    assert selection.policy is None
    assert selection.source == "explicit"
    assert "fortran_openmp" in selection.reason


def test_auto_uses_shipped_policy_when_profile_is_missing(
    tmp_path,
):
    missing_profile = tmp_path / "missing.json"

    selection = select_dispatch_policy(
        "auto",
        profile_path=missing_profile,
    )

    assert selection.source == "fallback"
    assert selection.policy is not None

    assert (
        selection.policy["policy_version"]
        == "cpu-fallback-v1"
    )

    assert "missing" in selection.reason


def test_auto_uses_compatible_machine_profile(
    tmp_path,
):
    import json

    from fortran_nlp.profile import (
        get_machine_fingerprint,
        get_package_version,
    )

    profile_path = tmp_path / "cpu_profile.json"

    profile = {
        "schema_version": 1,
        "profile_version": "cpu-profile-v1",
        "package_version": get_package_version(),
        "machine_fingerprint": get_machine_fingerprint(),
        "decisions": {
            "lexical": {
                "single": {
                    "strategy": "serial",
                },
            },
            "vector": {
                "ip": {
                    "single": {
                        "strategy": "blas_single",
                    },
                },
            },
        },
    }

    profile_path.write_text(
        json.dumps(profile),
        encoding="utf-8",
    )

    selection = select_dispatch_policy(
        "auto",
        profile_path=profile_path,
    )

    assert selection.source == "profile"
    assert selection.policy == profile["decisions"]

    assert (
        selection.policy["lexical"]["single"]["strategy"]
        == "serial"
    )

    assert str(profile_path) in selection.reason


def test_auto_rejects_incompatible_profile_and_uses_fallback(
    tmp_path,
):
    import json

    from fortran_nlp.profile import (
        get_machine_fingerprint,
    )

    profile_path = tmp_path / "cpu_profile.json"

    profile = {
        "schema_version": 1,
        "profile_version": "cpu-profile-v1",
        "package_version": "0.0.0",
        "machine_fingerprint": get_machine_fingerprint(),
        "decisions": {
            "lexical": {},
            "vector": {},
        },
    }

    profile_path.write_text(
        json.dumps(profile),
        encoding="utf-8",
    )

    selection = select_dispatch_policy(
        "auto",
        profile_path=profile_path,
    )

    assert selection.source == "fallback"
    assert selection.policy is not None

    assert (
        selection.policy["policy_version"]
        == "cpu-fallback-v1"
    )

    assert "incompatible" in selection.reason


def test_auto_rejects_malformed_profile_and_uses_fallback(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    profile_path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    selection = select_dispatch_policy(
        "auto",
        profile_path=profile_path,
    )

    assert selection.source == "fallback"
    assert selection.policy is not None
    assert "malformed" in selection.reason


def test_requested_backend_must_be_string():
    try:
        select_dispatch_policy(None)
    except TypeError as error:
        assert "requested_backend" in str(error)
    else:
        raise AssertionError(
            "Non-string backend must raise TypeError"
        )

def make_lexical_workload(
    *,
    operation="single",
    document_count=1000,
    total_corpus_shingles=200_000,
    candidate_count=1,
    available_memory_bytes=8 * 1024**3,
):
    return LexicalWorkload(
        operation=operation,
        document_count=document_count,
        total_corpus_shingles=total_corpus_shingles,
        candidate_count=candidate_count,
        candidate_shingle_counts=(100,) * candidate_count,
        available_memory_bytes=available_memory_bytes,
    )


def test_lexical_workload_identifies_batch_api_with_one_candidate():
    workload = make_lexical_workload(
        operation="batch",
        candidate_count=1,
    )

    assert workload.is_batch is True


def test_lexical_workload_identifies_single_api():
    workload = make_lexical_workload(
        operation="single",
        candidate_count=1,
    )

    assert workload.is_batch is False


def test_lexical_single_operation_rejects_multiple_candidates():
    with pytest.raises(
        ValueError,
        match="exactly one candidate",
    ):
        make_lexical_workload(
            operation="single",
            candidate_count=2,
        )


@pytest.mark.parametrize(
    (
        "field",
        "value",
        "message",
    ),
    [
        (
            "document_count",
            -1,
            "document_count",
        ),
        (
            "total_corpus_shingles",
            -1,
            "total_corpus_shingles",
        ),
        (
            "candidate_count",
            0,
            "candidate_count",
        ),
    ],
)
def test_lexical_workload_rejects_invalid_counts(
    field,
    value,
    message,
):
    arguments = {
        "operation": "single",
        "document_count": 100,
        "total_corpus_shingles": 1000,
        "candidate_count": 1,
    }

    arguments[field] = value

    if field == "candidate_count":
        arguments["operation"] = "batch"

    with pytest.raises(
        ValueError,
        match=message,
    ):
        make_lexical_workload(**arguments)


def test_lexical_workload_rejects_mismatched_candidate_lengths():
    with pytest.raises(
        ValueError,
        match="length must equal",
    ):
        LexicalWorkload(
            operation="batch",
            document_count=100,
            total_corpus_shingles=1000,
            candidate_count=2,
            candidate_shingle_counts=(100,),
        )


def test_explicit_serial_single_bypasses_auto_policy(
    monkeypatch,
):
    def fail_if_selected(*args, **kwargs):
        raise AssertionError(
            "Explicit backend must bypass policy selection"
        )

    monkeypatch.setattr(
        "fortran_nlp.auto.select_dispatch_policy",
        fail_if_selected,
    )

    decision = resolve_lexical_backend(
        "fortran",
        make_lexical_workload(),
    )

    assert decision.backend == "fortran"
    assert decision.strategy == "serial"
    assert decision.source == "explicit"
    assert decision.thread_policy is None


def test_explicit_serial_batch_uses_repeated_serial():
    decision = resolve_lexical_backend(
        "fortran",
        make_lexical_workload(
            operation="batch",
            candidate_count=8,
        ),
    )

    assert decision.backend == "fortran"
    assert decision.strategy == "repeated_serial"
    assert decision.source == "explicit"


def test_explicit_openmp_single_uses_openmp():
    decision = resolve_lexical_backend(
        "fortran_openmp",
        make_lexical_workload(),
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "openmp"
    assert decision.source == "explicit"
    assert decision.thread_policy is None


def test_explicit_openmp_batch_uses_repeated_openmp():
    decision = resolve_lexical_backend(
        "fortran_openmp",
        make_lexical_workload(
            operation="batch",
            candidate_count=8,
        ),
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "repeated_openmp"
    assert decision.source == "explicit"


def test_auto_single_at_serial_boundary_uses_serial(
    tmp_path,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            document_count=93,
            total_corpus_shingles=16_359,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran"
    assert decision.strategy == "serial"
    assert decision.source == "fallback"


@pytest.mark.parametrize(
    (
        "document_count",
        "total_corpus_shingles",
    ),
    [
        (94, 16_359),
        (93, 16_360),
        (94, 16_360),
    ],
)
def test_auto_single_above_either_serial_limit_uses_openmp(
    tmp_path,
    document_count,
    total_corpus_shingles,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            document_count=document_count,
            total_corpus_shingles=total_corpus_shingles,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "openmp"
    assert decision.source == "fallback"
    assert decision.thread_policy == "physical_cores"


def test_auto_tiny_corpus_batch_repeats_selected_single_path(
    tmp_path,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            operation="batch",
            document_count=100,
            total_corpus_shingles=10_000,
            candidate_count=8,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "repeated_openmp"
    assert decision.source == "fallback"
    assert decision.thread_policy == "physical_cores"

def test_auto_small_corpus_batch_repeats_serial_path(
    tmp_path,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            operation="batch",
            document_count=50,
            total_corpus_shingles=8_000,
            candidate_count=8,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran"
    assert decision.strategy == "repeated_serial"
    assert decision.source == "fallback"
    assert decision.thread_policy is None

@pytest.mark.parametrize(
    "candidate_count",
    [
        2,
        16,
        17,
        31,
    ],
)
def test_auto_scaled_batch_before_native_crossover_uses_repeated_openmp(
    tmp_path,
    candidate_count,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            operation="batch",
            document_count=1000,
            total_corpus_shingles=200_000,
            candidate_count=candidate_count,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "repeated_openmp"
    assert decision.source == "fallback"


@pytest.mark.parametrize(
    "candidate_count",
    [
        32,
        64,
    ],
)
def test_auto_scaled_batch_at_native_crossover_uses_native_batch(
    tmp_path,
    candidate_count,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            operation="batch",
            document_count=1000,
            total_corpus_shingles=200_000,
            candidate_count=candidate_count,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "native_batch_openmp"
    assert decision.source == "fallback"
    assert decision.thread_policy == "physical_cores"


def test_unsupported_lexical_backend_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported lexical backend",
    ):
        resolve_lexical_backend(
            "unknown",
            make_lexical_workload(),
        )

def test_auto_scaled_batch_one_uses_native_batch(
    tmp_path,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            operation="batch",
            document_count=1000,
            total_corpus_shingles=200_000,
            candidate_count=1,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "native_batch_openmp"
    assert decision.source == "fallback"
    assert decision.thread_policy == "physical_cores"

def test_auto_tiny_corpus_batch_one_repeats_selected_path(
    tmp_path,
):
    decision = resolve_lexical_backend(
        "auto",
        make_lexical_workload(
            operation="batch",
            document_count=50,
            total_corpus_shingles=8_000,
            candidate_count=1,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran"
    assert decision.strategy == "repeated_serial"
    assert decision.source == "fallback"

def make_vector_workload(
    *,
    metric="ip",
    operation="single",
    dimension=1024,
    vector_count=100_000,
    query_count=1,
    k=10,
    available_memory_bytes=8 * 1024**3,
):
    return VectorWorkload(
        metric=metric,
        operation=operation,
        dimension=dimension,
        vector_count=vector_count,
        query_count=query_count,
        k=k,
        available_memory_bytes=available_memory_bytes,
    )


@pytest.mark.parametrize(
    (
        "field",
        "value",
        "message",
    ),
    [
        ("dimension", 0, "dimension"),
        ("vector_count", 0, "vector_count"),
        ("query_count", 0, "query_count"),
        ("k", 0, "k"),
    ],
)
def test_vector_workload_rejects_invalid_counts(
    field,
    value,
    message,
):
    arguments = {
        "metric": "ip",
        "operation": "single",
        "dimension": 1024,
        "vector_count": 100_000,
        "query_count": 1,
        "k": 10,
    }

    arguments[field] = value

    if field == "query_count":
        arguments["operation"] = "batch"

    with pytest.raises(
        ValueError,
        match=message,
    ):
        make_vector_workload(**arguments)


def test_vector_workload_rejects_k_larger_than_database():
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        make_vector_workload(
            vector_count=10,
            k=11,
        )


def test_vector_single_rejects_multiple_queries():
    with pytest.raises(
        ValueError,
        match="exactly one query",
    ):
        make_vector_workload(
            operation="single",
            query_count=2,
        )


def test_explicit_vector_serial_single_bypasses_policy(
    monkeypatch,
):
    def fail_if_selected(*args, **kwargs):
        raise AssertionError(
            "Explicit backend must bypass policy selection"
        )

    monkeypatch.setattr(
        "fortran_nlp.auto.select_dispatch_policy",
        fail_if_selected,
    )

    decision = resolve_vector_backend(
        "fortran",
        make_vector_workload(),
    )

    assert decision.backend == "fortran"
    assert decision.strategy == "serial"
    assert decision.source == "explicit"
    assert decision.thread_policy is None


@pytest.mark.parametrize(
    (
        "backend",
        "expected_strategy",
    ),
    [
        ("fortran", "repeated_serial"),
        ("fortran_openmp", "repeated_openmp"),
        ("fortran_blas", "blas_batch"),
    ],
)
def test_explicit_vector_batch_strategy(
    backend,
    expected_strategy,
):
    decision = resolve_vector_backend(
        backend,
        make_vector_workload(
            operation="batch",
            query_count=8,
        ),
    )

    assert decision.backend == backend
    assert decision.strategy == expected_strategy
    assert decision.source == "explicit"
    assert decision.thread_policy is None


@pytest.mark.parametrize(
    "metric",
    [
        "ip",
        "cosine",
    ],
)
def test_auto_ip_family_single_uses_blas_single(
    tmp_path,
    metric,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric=metric,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_blas"
    assert decision.strategy == "blas_single"
    assert decision.source == "fallback"
    assert decision.thread_policy == "single_thread"


@pytest.mark.parametrize(
    "query_count",
    [
        1,
        4,
        5,
        7,
    ],
)
def test_auto_ip_batch_before_sgemm_region_repeats_sgemv(
    tmp_path,
    query_count,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric="ip",
            operation="batch",
            query_count=query_count,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_blas"
    assert decision.strategy == "repeated_blas_single"
    assert decision.source == "fallback"
    assert decision.thread_policy == "single_thread"


@pytest.mark.parametrize(
    "query_count",
    [
        8,
        16,
        64,
    ],
)
def test_auto_ip_batch_from_sgemm_boundary_uses_blas_batch(
    tmp_path,
    query_count,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric="ip",
            operation="batch",
            query_count=query_count,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_blas"
    assert decision.strategy == "blas_batch"
    assert decision.source == "fallback"
    assert decision.thread_policy == "physical_cores"


def test_auto_l2_single_uses_openmp(
    tmp_path,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric="l2",
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "openmp"
    assert decision.source == "fallback"
    assert decision.thread_policy == "physical_cores"


def test_auto_l2_batch_uses_repeated_openmp(
    tmp_path,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric="l2",
            operation="batch",
            query_count=32,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_openmp"
    assert decision.strategy == "repeated_openmp"
    assert decision.source == "fallback"
    assert decision.thread_policy == "physical_cores"


def test_unsupported_vector_backend_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported vector backend",
    ):
        resolve_vector_backend(
            "unknown",
            make_vector_workload(),
        )


def test_unsupported_vector_metric_is_rejected():
    with pytest.raises(
        ValueError,
        match="metric",
    ):
        make_vector_workload(
            metric="unknown",
        )

def test_vector_blas_batch_falls_back_when_memory_unknown(
    tmp_path,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric="ip",
            operation="batch",
            vector_count=1_000_000,
            query_count=32,
            available_memory_bytes=None,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_blas"
    assert decision.strategy == "repeated_blas_single"
    assert decision.thread_policy == "single_thread"
    assert "could not be determined" in decision.reason


def test_vector_blas_batch_falls_back_when_memory_insufficient(
    tmp_path,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric="ip",
            operation="batch",
            vector_count=10_000_000,
            query_count=64,
            available_memory_bytes=1024**3,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_blas"
    assert decision.strategy == "repeated_blas_single"
    assert decision.thread_policy == "single_thread"
    assert "exceeds budget" in decision.reason


def test_vector_blas_batch_runs_when_memory_sufficient(
    tmp_path,
):
    decision = resolve_vector_backend(
        "auto",
        make_vector_workload(
            metric="ip",
            operation="batch",
            vector_count=100_000,
            query_count=32,
            available_memory_bytes=8 * 1024**3,
        ),
        profile_path=tmp_path / "missing.json",
    )

    assert decision.backend == "fortran_blas"
    assert decision.strategy == "blas_batch"
    assert decision.thread_policy == "physical_cores"
    assert "within" in decision.reason


def test_explicit_vector_blas_ignores_automatic_memory_guard():
    decision = resolve_vector_backend(
        "fortran_blas",
        make_vector_workload(
            metric="ip",
            operation="batch",
            vector_count=10_000_000,
            query_count=64,
            available_memory_bytes=1,
        ),
    )

    assert decision.backend == "fortran_blas"
    assert decision.strategy == "blas_batch"
    assert decision.source == "explicit"
    assert decision.thread_policy is None