"""CPU backend policy loading and runtime dispatch decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import os
from typing import Any, Literal

from .profile import load_profile


_POLICY_PACKAGE = "fortran_nlp.policies"
_POLICY_FILENAME = "cpu_fallback_v1.json"

_SUPPORTED_SCHEMA_VERSION = 1
_SUPPORTED_POLICY_VERSION = "cpu-fallback-v1"


DecisionSource = Literal[
    "explicit",
    "profile",
    "fallback",
]

LexicalOperation = Literal[
    "single",
    "batch",
]

VectorMetric = Literal[
    "ip",
    "cosine",
    "l2",
]

VectorOperation = Literal[
    "single",
    "batch",
]

class PolicyError(RuntimeError):
    """Raised when a CPU dispatch policy is unusable."""


@dataclass(frozen=True)
class BackendDecision:
    """Final backend decision for one search operation."""

    backend: str
    strategy: str
    source: DecisionSource
    reason: str
    thread_policy: str | None = None


@dataclass(frozen=True)
class PolicySelection:
    """Policy source selected for one dispatch operation."""

    policy: dict[str, Any] | None
    source: DecisionSource
    reason: str


@dataclass(frozen=True)
class LexicalWorkload:
    """Description of one lexical-search operation."""

    operation: LexicalOperation
    document_count: int
    total_corpus_shingles: int
    candidate_count: int
    candidate_shingle_counts: tuple[int, ...]
    available_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.operation not in {"single", "batch"}:
            raise ValueError("operation must be 'single' or 'batch'")

        if self.document_count < 0:
            raise ValueError("document_count must not be negative")

        if self.total_corpus_shingles < 0:
            raise ValueError(
                "total_corpus_shingles must not be negative"
            )

        if self.candidate_count < 1:
            raise ValueError("candidate_count must be at least 1")

        if (
            len(self.candidate_shingle_counts)
            != self.candidate_count
        ):
            raise ValueError(
                "candidate_shingle_counts length must equal "
                "candidate_count"
            )

        if any(
            count < 0
            for count in self.candidate_shingle_counts
        ):
            raise ValueError(
                "candidate shingle counts must not be negative"
            )

        if (
            self.operation == "single"
            and self.candidate_count != 1
        ):
            raise ValueError(
                "single operation must contain exactly one candidate"
            )

        if (
            self.available_memory_bytes is not None
            and self.available_memory_bytes < 0
        ):
            raise ValueError(
                "available_memory_bytes must not be negative"
            )

    @property
    def is_batch(self) -> bool:
        """Return whether search_batch() created this workload."""

        return self.operation == "batch"

@dataclass(frozen=True)
class VectorWorkload:
    """Description of one exact vector-search operation."""

    metric: VectorMetric
    operation: VectorOperation
    dimension: int
    vector_count: int
    query_count: int
    k: int
    available_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.metric not in {"ip", "cosine", "l2"}:
            raise ValueError(
                "metric must be 'ip', 'cosine', or 'l2'"
            )

        if self.operation not in {"single", "batch"}:
            raise ValueError(
                "operation must be 'single' or 'batch'"
            )

        if self.dimension < 1:
            raise ValueError(
                "dimension must be at least 1"
            )

        if self.vector_count < 1:
            raise ValueError(
                "vector_count must be at least 1"
            )

        if self.query_count < 1:
            raise ValueError(
                "query_count must be at least 1"
            )

        if self.k < 1:
            raise ValueError(
                "k must be at least 1"
            )

        if self.k > self.vector_count:
            raise ValueError(
                "k cannot exceed vector_count"
            )

        if (
            self.operation == "single"
            and self.query_count != 1
        ):
            raise ValueError(
                "single operation must contain exactly one query"
            )

        if (
            self.available_memory_bytes is not None
            and self.available_memory_bytes < 0
        ):
            raise ValueError(
                "available_memory_bytes must not be negative"
            )

    @property
    def is_batch(self) -> bool:
        """Return whether search_batch() created this workload."""

        return self.operation == "batch"

def _require_mapping(
    value: Any,
    *,
    field: str,
) -> dict[str, Any]:
    """Return a JSON object or raise a clear policy error."""

    if not isinstance(value, dict):
        raise PolicyError(
            f"CPU dispatch policy field '{field}' "
            "must be a JSON object"
        )

    return value


def _validate_shipped_policy(
    policy: Any,
) -> dict[str, Any]:
    """Validate the minimum structure required by dispatch."""

    root = _require_mapping(policy, field="<root>")

    schema_version = root.get("schema_version")

    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise PolicyError(
            "Unsupported CPU fallback policy schema version: "
            f"{schema_version!r}. Expected "
            f"{_SUPPORTED_SCHEMA_VERSION}."
        )

    policy_version = root.get("policy_version")

    if policy_version != _SUPPORTED_POLICY_VERSION:
        raise PolicyError(
            "Unsupported CPU fallback policy version: "
            f"{policy_version!r}. Expected "
            f"'{_SUPPORTED_POLICY_VERSION}'."
        )

    compatibility = _require_mapping(
        root.get("package_compatibility"),
        field="package_compatibility",
    )

    for field in (
        "minimum_version",
        "maximum_version_exclusive",
    ):
        value = compatibility.get(field)

        if not isinstance(value, str) or not value:
            raise PolicyError(
                "CPU fallback policy field "
                f"'package_compatibility.{field}' "
                "must be a non-empty string"
            )

    defaults = _require_mapping(
        root.get("defaults"),
        field="defaults",
    )

    required_defaults = {
        "openmp_thread_policy": str,
        "openblas_thread_policy": str,
        "full_smt_automatic": bool,
    }

    for field, expected_type in required_defaults.items():
        value = defaults.get(field)

        if not isinstance(value, expected_type):
            raise PolicyError(
                f"CPU fallback policy field 'defaults.{field}' "
                f"must be of type {expected_type.__name__}"
            )

    lexical = _require_mapping(
        root.get("lexical"),
        field="lexical",
    )

    for section in ("single", "batch", "memory"):
        _require_mapping(
            lexical.get(section),
            field=f"lexical.{section}",
        )

    vector = _require_mapping(
        root.get("vector"),
        field="vector",
    )

    for metric in ("ip", "cosine", "l2"):
        _require_mapping(
            vector.get(metric),
            field=f"vector.{metric}",
        )

    return root


@lru_cache(maxsize=1)
def _load_shipped_policy() -> dict[str, Any]:
    """Load and validate the immutable packaged fallback policy."""

    try:
        policy_resource = files(_POLICY_PACKAGE).joinpath(
            _POLICY_FILENAME
        )

        with policy_resource.open(
            "r",
            encoding="utf-8",
        ) as policy_file:
            raw_policy = json.load(policy_file)

    except FileNotFoundError as error:
        raise PolicyError(
            "Packaged CPU fallback policy is missing: "
            f"{_POLICY_PACKAGE}/{_POLICY_FILENAME}. "
            "The installation may be incomplete."
        ) from error

    except json.JSONDecodeError as error:
        raise PolicyError(
            "Packaged CPU fallback policy contains invalid JSON: "
            f"{error}"
        ) from error

    except (ModuleNotFoundError, OSError) as error:
        raise PolicyError(
            "Packaged CPU fallback policy could not be read: "
            f"{error}"
        ) from error

    return _validate_shipped_policy(raw_policy)


def get_shipped_policy() -> dict[str, Any]:
    """Return an independent copy of the shipped fallback policy."""

    return deepcopy(_load_shipped_policy())


def select_dispatch_policy(
    requested_backend: str,
    *,
    profile_path: str | os.PathLike[str] | None = None,
) -> PolicySelection:
    """Select explicit, calibrated-profile, or fallback policy data."""

    if not isinstance(requested_backend, str):
        raise TypeError("requested_backend must be a string")

    if requested_backend != "auto":
        return PolicySelection(
            policy=None,
            source="explicit",
            reason=(
                "Explicit backend override requested: "
                f"{requested_backend}"
            ),
        )

    profile_result = load_profile(profile_path)

    if profile_result.compatible:
        if profile_result.profile is None:
            raise RuntimeError(
                "Compatible profile result does not contain a profile"
            )

        return PolicySelection(
            policy=profile_result.profile["decisions"],
            source="profile",
            reason=(
                "Using compatible machine calibration profile: "
                f"{profile_result.path}"
            ),
        )

    return PolicySelection(
        policy=get_shipped_policy(),
        source="fallback",
        reason=(
            "Using packaged CPU fallback policy because the machine "
            f"profile is {profile_result.status}: "
            f"{profile_result.reason}"
        ),
    )


def _resolve_explicit_lexical_backend(
    requested_backend: str,
    workload: LexicalWorkload,
) -> BackendDecision:
    """Resolve an explicitly requested lexical backend."""

    if requested_backend == "fortran":
        return BackendDecision(
            backend="fortran",
            strategy=(
                "repeated_serial"
                if workload.is_batch
                else "serial"
            ),
            source="explicit",
            reason="Explicit serial Fortran backend override",
            thread_policy=None,
        )

    if requested_backend == "fortran_openmp":
        return BackendDecision(
            backend="fortran_openmp",
            strategy=(
                "repeated_openmp"
                if workload.is_batch
                else "openmp"
            ),
            source="explicit",
            reason="Explicit OpenMP Fortran backend override",
            thread_policy=None,
        )

    raise ValueError(
        f"Unsupported lexical backend '{requested_backend}'. "
        "Available backends: auto, fortran, fortran_openmp"
    )

def _resolve_explicit_vector_backend(
    requested_backend: str,
    workload: VectorWorkload,
) -> BackendDecision:
    """Resolve an explicitly requested vector backend."""

    if requested_backend == "fortran":
        return BackendDecision(
            backend="fortran",
            strategy=(
                "repeated_serial"
                if workload.is_batch
                else "serial"
            ),
            source="explicit",
            reason="Explicit serial Fortran backend override",
            thread_policy=None,
        )

    if requested_backend == "fortran_openmp":
        return BackendDecision(
            backend="fortran_openmp",
            strategy=(
                "repeated_openmp"
                if workload.is_batch
                else "openmp"
            ),
            source="explicit",
            reason="Explicit OpenMP Fortran backend override",
            thread_policy=None,
        )

    if requested_backend == "fortran_blas":
        return BackendDecision(
            backend="fortran_blas",
            strategy=(
                "blas_batch"
                if workload.is_batch
                else "blas_single"
            ),
            source="explicit",
            reason="Explicit OpenBLAS backend override",
            thread_policy=None,
        )

    raise ValueError(
        f"Unsupported vector backend '{requested_backend}'. "
        "Available backends: auto, fortran, "
        "fortran_openmp, fortran_blas"
    )

def _resolve_lexical_single(
    lexical_policy: dict[str, Any],
    *,
    workload: LexicalWorkload,
    source: DecisionSource,
    selection_reason: str,
) -> BackendDecision:
    """Resolve one lexical candidate against the corpus."""

    single_policy = _require_mapping(
        lexical_policy.get("single"),
        field="lexical.single",
    )

    serial_max_document_count = single_policy.get(
        "serial_max_document_count"
    )
    serial_max_corpus_shingles = single_policy.get(
        "serial_max_corpus_shingles"
    )

    if not isinstance(serial_max_document_count, int):
        raise PolicyError(
            "CPU dispatch policy field "
            "'lexical.single.serial_max_document_count' "
            "must be an integer"
        )

    if not isinstance(serial_max_corpus_shingles, int):
        raise PolicyError(
            "CPU dispatch policy field "
            "'lexical.single.serial_max_corpus_shingles' "
            "must be an integer"
        )

    use_serial = (
        workload.document_count <= serial_max_document_count
        and workload.total_corpus_shingles
        <= serial_max_corpus_shingles
    )

    if use_serial:
        return BackendDecision(
            backend="fortran",
            strategy="serial",
            source=source,
            reason=(
                f"{selection_reason}; lexical workload is within "
                "the serial fallback region"
            ),
            thread_policy=None,
        )

    thread_policy = single_policy.get(
        "thread_policy",
        "physical_cores",
    )

    return BackendDecision(
        backend="fortran_openmp",
        strategy="openmp",
        source=source,
        reason=(
            f"{selection_reason}; lexical workload exceeds the "
            "serial fallback region"
        ),
        thread_policy=thread_policy,
    )


def _lexical_native_batch_fits_memory(
    lexical_policy: dict[str, Any],
    workload: LexicalWorkload,
) -> tuple[bool, int, int | None]:
    """Check whether native lexical batch output fits its budget."""

    memory_policy = _require_mapping(
        lexical_policy.get("memory"),
        field="lexical.memory",
    )

    bytes_per_result = memory_policy.get(
        "bytes_per_native_batch_result"
    )
    available_fraction = memory_policy.get(
        "maximum_available_memory_fraction"
    )
    safety_margin_fraction = memory_policy.get(
        "safety_margin_fraction"
    )

    if (
        not isinstance(bytes_per_result, int)
        or isinstance(bytes_per_result, bool)
        or bytes_per_result < 1
    ):
        raise PolicyError(
            "CPU dispatch policy field "
            "'lexical.memory.bytes_per_native_batch_result' "
            "must be a positive integer"
        )

    if (
        not isinstance(available_fraction, (int, float))
        or isinstance(available_fraction, bool)
        or not 0.0 < available_fraction <= 1.0
    ):
        raise PolicyError(
            "CPU dispatch policy field "
            "'lexical.memory.maximum_available_memory_fraction' "
            "must be between 0 and 1"
        )

    if (
        not isinstance(safety_margin_fraction, (int, float))
        or isinstance(safety_margin_fraction, bool)
        or not 0.0 <= safety_margin_fraction < 1.0
    ):
        raise PolicyError(
            "CPU dispatch policy field "
            "'lexical.memory.safety_margin_fraction' "
            "must be between 0 and 1"
        )

    estimated_output_bytes = (
        workload.document_count
        * workload.candidate_count
        * bytes_per_result
    )

    if workload.available_memory_bytes is None:
        return False, estimated_output_bytes, None

    allowed_output_bytes = int(
        workload.available_memory_bytes
        * float(available_fraction)
        * (1.0 - float(safety_margin_fraction))
    )

    return (
        estimated_output_bytes <= allowed_output_bytes,
        estimated_output_bytes,
        allowed_output_bytes,
    )


def _resolve_lexical_batch(
    lexical_policy: dict[str, Any],
    *,
    workload: LexicalWorkload,
    source: DecisionSource,
    selection_reason: str,
) -> BackendDecision:
    """Resolve a batched lexical-search operation."""

    batch_policy = _require_mapping(
        lexical_policy.get("batch"),
        field="lexical.batch",
    )

    tiny_corpus_max_document_count = batch_policy.get(
        "tiny_corpus_max_document_count"
    )
    repeated_openmp_max_candidates = batch_policy.get(
        "repeated_openmp_max_candidates"
    )
    native_batch_min_candidates = batch_policy.get(
        "native_batch_min_candidates"
    )

    required_integer_fields = {
        "tiny_corpus_max_document_count": (
            tiny_corpus_max_document_count
        ),
        "repeated_openmp_max_candidates": (
            repeated_openmp_max_candidates
        ),
        "native_batch_min_candidates": (
            native_batch_min_candidates
        ),
    }

    for field, value in required_integer_fields.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise PolicyError(
                "CPU dispatch policy field "
                f"'lexical.batch.{field}' must be an integer"
            )

    thread_policy = batch_policy.get(
        "thread_policy",
        "physical_cores",
    )

    def native_batch_decision(
        reason: str,
    ) -> BackendDecision:
        (
            fits_memory,
            estimated_bytes,
            allowed_bytes,
        ) = _lexical_native_batch_fits_memory(
            lexical_policy,
            workload,
        )

        if not fits_memory:
            if allowed_bytes is None:
                memory_reason = (
                    "available memory could not be determined"
                )
            else:
                memory_reason = (
                    f"estimated output {estimated_bytes} bytes "
                    f"exceeds budget {allowed_bytes} bytes"
                )

            return BackendDecision(
                backend="fortran_openmp",
                strategy="repeated_openmp",
                source=source,
                reason=(
                    f"{selection_reason}; {reason}, but "
                    f"{memory_reason}; using repeated OpenMP"
                ),
                thread_policy=thread_policy,
            )

        return BackendDecision(
            backend="fortran_openmp",
            strategy="native_batch_openmp",
            source=source,
            reason=(
                f"{selection_reason}; {reason}; estimated native "
                f"output is {estimated_bytes} bytes within the "
                f"{allowed_bytes}-byte memory budget"
            ),
            thread_policy=thread_policy,
        )

    if (
        workload.document_count
        <= tiny_corpus_max_document_count
    ):
        single_decision = _resolve_lexical_single(
            lexical_policy,
            workload=LexicalWorkload(
                operation="single",
                document_count=workload.document_count,
                total_corpus_shingles=(
                    workload.total_corpus_shingles
                ),
                candidate_count=1,
                candidate_shingle_counts=(
                    workload.candidate_shingle_counts[0],
                ),
                available_memory_bytes=(
                    workload.available_memory_bytes
                ),
            ),
            source=source,
            selection_reason=selection_reason,
        )

        strategy = (
            "repeated_serial"
            if single_decision.backend == "fortran"
            else "repeated_openmp"
        )

        return BackendDecision(
            backend=single_decision.backend,
            strategy=strategy,
            source=source,
            reason=(
                f"{selection_reason}; tiny lexical corpus uses "
                f"the repeated {single_decision.strategy} path"
            ),
            thread_policy=single_decision.thread_policy,
        )

    if workload.candidate_count == 1:
        batch_one_strategy = batch_policy.get(
            "scaled_batch_one_strategy"
        )

        if batch_one_strategy != "native_batch_openmp":
            raise PolicyError(
                "CPU dispatch policy field "
                "'lexical.batch.scaled_batch_one_strategy' "
                "must be 'native_batch_openmp'"
            )

        return native_batch_decision(
            "scaled batch API workload with one candidate "
            "reached the native batch policy"
        )

    if (
        workload.candidate_count
        <= repeated_openmp_max_candidates
    ):
        return BackendDecision(
            backend="fortran_openmp",
            strategy="repeated_openmp",
            source=source,
            reason=(
                f"{selection_reason}; candidate count "
                f"{workload.candidate_count} is below the native "
                "batch crossover"
            ),
            thread_policy=thread_policy,
        )

    if (
        workload.candidate_count
        >= native_batch_min_candidates
    ):
        return native_batch_decision(
            f"candidate count {workload.candidate_count} "
            "reached the native batch crossover"
        )

    return BackendDecision(
        backend="fortran_openmp",
        strategy="repeated_openmp",
        source=source,
        reason=(
            f"{selection_reason}; unresolved crossover region uses "
            "the conservative repeated OpenMP path"
        ),
        thread_policy=thread_policy,
    )


def resolve_lexical_backend(
    requested_backend: str,
    workload: LexicalWorkload,
    *,
    profile_path: str | os.PathLike[str] | None = None,
) -> BackendDecision:
    """Resolve the backend and strategy for lexical search."""

    supported_backends = {
        "auto",
        "fortran",
        "fortran_openmp",
    }

    if requested_backend not in supported_backends:
        raise ValueError(
            f"Unsupported lexical backend "
            f"'{requested_backend}'. Available backends: "
            f"{', '.join(sorted(supported_backends))}"
        )

    if requested_backend != "auto":
        return _resolve_explicit_lexical_backend(
            requested_backend,
            workload,
        )

    selection = select_dispatch_policy(
        requested_backend,
        profile_path=profile_path,
    )

    if selection.policy is None:
        raise RuntimeError(
            "Automatic lexical dispatch selected no policy"
        )

    lexical_policy = _require_mapping(
        selection.policy.get("lexical"),
        field="lexical",
    )

    if not workload.is_batch:
        return _resolve_lexical_single(
            lexical_policy,
            workload=workload,
            source=selection.source,
            selection_reason=selection.reason,
        )

    return _resolve_lexical_batch(
        lexical_policy,
        workload=workload,
        source=selection.source,
        selection_reason=selection.reason,
    )

def _vector_blas_batch_fits_memory(
    vector_policy: dict[str, Any],
    workload: VectorWorkload,
) -> tuple[bool, int, int | None]:
    """
    Check whether automatic native BLAS batching fits its memory budget.

    The estimate includes the full float32 score matrix and the
    returned top-k values and indices.
    """

    memory_policy = _require_mapping(
        vector_policy.get("memory"),
        field="vector.memory",
    )

    bytes_per_score = memory_policy.get(
        "bytes_per_score"
    )
    bytes_per_result_value = memory_policy.get(
        "bytes_per_result_value"
    )
    bytes_per_result_index = memory_policy.get(
        "bytes_per_result_index"
    )
    available_fraction = memory_policy.get(
        "maximum_available_memory_fraction"
    )
    safety_margin_fraction = memory_policy.get(
        "safety_margin_fraction"
    )

    integer_fields = {
        "bytes_per_score": bytes_per_score,
        "bytes_per_result_value": bytes_per_result_value,
        "bytes_per_result_index": bytes_per_result_index,
    }

    for field, value in integer_fields.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise PolicyError(
                "CPU dispatch policy field "
                f"'vector.memory.{field}' "
                "must be a positive integer"
            )

    if (
        not isinstance(available_fraction, (int, float))
        or isinstance(available_fraction, bool)
        or not 0.0 < available_fraction <= 1.0
    ):
        raise PolicyError(
            "CPU dispatch policy field "
            "'vector.memory.maximum_available_memory_fraction' "
            "must be between 0 and 1"
        )

    if (
        not isinstance(safety_margin_fraction, (int, float))
        or isinstance(safety_margin_fraction, bool)
        or not 0.0 <= safety_margin_fraction < 1.0
    ):
        raise PolicyError(
            "CPU dispatch policy field "
            "'vector.memory.safety_margin_fraction' "
            "must be between 0 and 1"
        )

    score_matrix_bytes = (
        workload.vector_count
        * workload.query_count
        * bytes_per_score
    )

    result_bytes = (
        workload.query_count
        * workload.k
        * (
            bytes_per_result_value
            + bytes_per_result_index
        )
    )

    estimated_bytes = (
        score_matrix_bytes
        + result_bytes
    )

    if workload.available_memory_bytes is None:
        return False, estimated_bytes, None

    allowed_bytes = int(
        workload.available_memory_bytes
        * float(available_fraction)
        * (1.0 - float(safety_margin_fraction))
    )

    return (
        estimated_bytes <= allowed_bytes,
        estimated_bytes,
        allowed_bytes,
    )

def _resolve_vector_from_policy(
    vector_policy: dict[str, Any],
    *,
    workload: VectorWorkload,
    source: DecisionSource,
    selection_reason: str,
) -> BackendDecision:
    """Resolve automatic vector dispatch from selected policy data."""

    policy_metric = (
        "ip"
        if workload.metric == "cosine"
        else workload.metric
    )

    metric_policy = _require_mapping(
        vector_policy.get(policy_metric),
        field=f"vector.{policy_metric}",
    )

    if not workload.is_batch:
        single_policy = _require_mapping(
            metric_policy.get("single"),
            field=f"vector.{policy_metric}.single",
        )

        backend = single_policy.get("backend")
        strategy = single_policy.get("strategy")
        thread_policy = single_policy.get("thread_policy")

        if not isinstance(backend, str) or not backend:
            raise PolicyError(
                "CPU dispatch policy field "
                f"'vector.{policy_metric}.single.backend' "
                "must be a non-empty string"
            )

        if not isinstance(strategy, str) or not strategy:
            raise PolicyError(
                "CPU dispatch policy field "
                f"'vector.{policy_metric}.single.strategy' "
                "must be a non-empty string"
            )

        return BackendDecision(
            backend=backend,
            strategy=strategy,
            source=source,
            reason=(
                f"{selection_reason}; automatic "
                f"{workload.metric} single-query policy selected "
                f"{strategy}"
            ),
            thread_policy=thread_policy,
        )

    batch_policy = _require_mapping(
        metric_policy.get("batch"),
        field=f"vector.{policy_metric}.batch",
    )

    if policy_metric == "l2":
        backend = batch_policy.get("backend")
        strategy = batch_policy.get("strategy")
        thread_policy = batch_policy.get("thread_policy")

        if not isinstance(backend, str) or not backend:
            raise PolicyError(
                "CPU dispatch policy field "
                "'vector.l2.batch.backend' "
                "must be a non-empty string"
            )

        if not isinstance(strategy, str) or not strategy:
            raise PolicyError(
                "CPU dispatch policy field "
                "'vector.l2.batch.strategy' "
                "must be a non-empty string"
            )

        return BackendDecision(
            backend=backend,
            strategy=strategy,
            source=source,
            reason=(
                f"{selection_reason}; automatic L2 batch policy "
                f"selected {strategy}"
            ),
            thread_policy=thread_policy,
        )

    repeated_max_queries = batch_policy.get(
        "repeated_single_max_queries"
    )
    native_min_queries = batch_policy.get(
        "native_batch_min_queries"
    )

    for field, value in {
        "repeated_single_max_queries": repeated_max_queries,
        "native_batch_min_queries": native_min_queries,
    }.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise PolicyError(
                "CPU dispatch policy field "
                f"'vector.{policy_metric}.batch.{field}' "
                "must be an integer"
            )

    backend = batch_policy.get("backend")

    if not isinstance(backend, str) or not backend:
        raise PolicyError(
            "CPU dispatch policy field "
            f"'vector.{policy_metric}.batch.backend' "
            "must be a non-empty string"
        )

    if workload.query_count <= repeated_max_queries:
        strategy = batch_policy.get(
            "repeated_single_strategy"
        )

        if not isinstance(strategy, str) or not strategy:
            raise PolicyError(
                "CPU dispatch policy field "
                f"'vector.{policy_metric}.batch."
                "repeated_single_strategy' "
                "must be a non-empty string"
            )

        return BackendDecision(
            backend=backend,
            strategy=strategy,
            source=source,
            reason=(
                f"{selection_reason}; query count "
                f"{workload.query_count} is within the repeated "
                "single-query BLAS region"
            ),
            thread_policy=batch_policy.get(
                "repeated_single_thread_policy",
                "single_thread",
            ),
        )

    if workload.query_count >= native_min_queries:
        strategy = batch_policy.get(
            "native_batch_strategy"
        )
        thread_policy = batch_policy.get(
            "thread_policy"
        )

        if not isinstance(strategy, str) or not strategy:
            raise PolicyError(
                "CPU dispatch policy field "
                f"'vector.{policy_metric}.batch."
                "native_batch_strategy' "
                "must be a non-empty string"
            )

        (
            fits_memory,
            estimated_bytes,
            allowed_bytes,
        ) = _vector_blas_batch_fits_memory(
            vector_policy,
            workload,
        )

        if not fits_memory:
            if allowed_bytes is None:
                memory_reason = (
                    "available memory could not be determined"
                )
            else:
                memory_reason = (
                    f"estimated BLAS batch allocation "
                    f"{estimated_bytes} bytes exceeds budget "
                    f"{allowed_bytes} bytes"
                )

            return BackendDecision(
                backend=backend,
                strategy="repeated_blas_single",
                source=source,
                reason=(
                    f"{selection_reason}; query count "
                    f"{workload.query_count} reached the native "
                    f"SGEMM region, but {memory_reason}; using "
                    "repeated SGEMV"
                ),
                thread_policy=batch_policy.get(
                    "repeated_single_thread_policy",
                    "single_thread",
                ),
            )

        return BackendDecision(
            backend=backend,
            strategy=strategy,
            source=source,
            reason=(
                f"{selection_reason}; query count "
                f"{workload.query_count} reached the native "
                f"SGEMM batch region; estimated allocation "
                f"{estimated_bytes} bytes is within the "
                f"{allowed_bytes}-byte memory budget"
            ),
            thread_policy=thread_policy,
        )

    return BackendDecision(
        backend=backend,
        strategy="repeated_blas_single",
        source=source,
        reason=(
            f"{selection_reason}; query count "
            f"{workload.query_count} is in the conservative "
            "BLAS crossover region"
        ),
        thread_policy=batch_policy.get(
            "repeated_single_thread_policy",
            "single_thread",
        ),
    )

def resolve_vector_backend(
    requested_backend: str,
    workload: VectorWorkload,
    *,
    profile_path: str | os.PathLike[str] | None = None,
) -> BackendDecision:
    """
    Resolve the backend and execution strategy for vector search.

    Precedence:
        1. Explicit backend override
        2. Compatible calibrated machine profile
        3. Packaged fallback policy
    """

    supported_backends = {
        "auto",
        "fortran",
        "fortran_openmp",
        "fortran_blas",
    }

    if requested_backend not in supported_backends:
        raise ValueError(
            f"Unsupported vector backend "
            f"'{requested_backend}'. Available backends: "
            f"{', '.join(sorted(supported_backends))}"
        )

    if requested_backend != "auto":
        return _resolve_explicit_vector_backend(
            requested_backend,
            workload,
        )

    selection = select_dispatch_policy(
        requested_backend,
        profile_path=profile_path,
    )

    if selection.policy is None:
        raise RuntimeError(
            "Automatic vector dispatch selected no policy"
        )

    vector_policy = _require_mapping(
        selection.policy.get("vector"),
        field="vector",
    )

    return _resolve_vector_from_policy(
        vector_policy,
        workload=workload,
        source=selection.source,
        selection_reason=selection.reason,
    )