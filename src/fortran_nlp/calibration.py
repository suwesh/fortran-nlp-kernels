"""Explicit CPU calibration profile management and CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from .profile import (
    get_machine_fingerprint,
    get_package_version,
    get_profile_path,
    load_profile,
    validate_profile,
)
from copy import deepcopy
from statistics import median
from time import perf_counter_ns

import numpy as np

from fortran_nlp import _cpu
from .auto import get_shipped_policy
from .runtime import (
    apply_openblas_thread_policy,
    apply_openmp_thread_policy,
)

_PROFILE_VERSION = "cpu-profile-v1"
_PROFILE_SCHEMA_VERSION = 1
_CALIBRATION_REPEATS = 10
_CALIBRATION_WARMUPS = 4
_RANDOM_SEED = 20261024

def build_profile(
    decisions: dict[str, Any],
) -> dict[str, Any]:
    """
    Build and validate a machine-specific CPU profile.

    The decisions mapping must contain lexical and vector sections
    consumable by the runtime resolvers.
    """

    if not isinstance(decisions, dict):
        raise TypeError(
            "decisions must be a dictionary"
        )

    lexical = decisions.get("lexical")
    vector = decisions.get("vector")

    if not isinstance(lexical, dict):
        raise ValueError(
            "decisions.lexical must be a dictionary"
        )

    if not isinstance(vector, dict):
        raise ValueError(
            "decisions.vector must be a dictionary"
        )

    profile = {
        "schema_version": _PROFILE_SCHEMA_VERSION,
        "profile_version": _PROFILE_VERSION,
        "package_version": get_package_version(),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "machine_fingerprint": (
            get_machine_fingerprint()
        ),
        "decisions": {
            "lexical": lexical,
            "vector": vector,
            **(
                {
                    "calibration": decisions["calibration"],
                }
                if "calibration" in decisions
                else {}
            ),
        },
    }

    return validate_profile(profile)


def write_profile_atomic(
    profile: dict[str, Any],
    path: str | os.PathLike[str] | None = None,
) -> Path:
    """
    Validate and atomically write a CPU calibration profile.

    The shipped fallback policy is never modified.
    """

    validated_profile = validate_profile(
        profile
    )

    profile_path = get_profile_path(path)
    profile_directory = profile_path.parent

    profile_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{profile_path.name}.",
            suffix=".tmp",
            dir=profile_directory,
            text=True,
        )

        temporary_path = Path(
            temporary_name
        )

        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as profile_file:
            json.dump(
                validated_profile,
                profile_file,
                indent=2,
                sort_keys=True,
            )

            profile_file.write("\n")
            profile_file.flush()
            os.fsync(profile_file.fileno())

        os.replace(
            temporary_path,
            profile_path,
        )

    except Exception:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

        raise

    return profile_path


def show_profile(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """
    Return the compatible active profile, or None if unavailable.

    This function does not return malformed, stale, or incompatible
    profile contents as an active calibration profile.
    """

    result = load_profile(path)

    if not result.compatible:
        return None

    return result.profile


def reset_profile(
    path: str | os.PathLike[str] | None = None,
) -> bool:
    """
    Remove the selected user calibration profile.

    Returns True when a profile was removed and False when no profile
    existed. The packaged fallback policy is never touched.
    """

    profile_path = get_profile_path(path)

    try:
        profile_path.unlink()
    except FileNotFoundError:
        return False

    return True

def _thread_candidates() -> tuple[int, ...]:
    """Return conservative thread counts for calibration."""

    fingerprint = get_machine_fingerprint()

    physical_cores = fingerprint.get(
        "physical_core_count"
    )

    if (
        not isinstance(physical_cores, int)
        or physical_cores < 1
    ):
        physical_cores = 1

    candidates = {
        1,
        max(1, physical_cores // 2),
        physical_cores,
    }

    return tuple(sorted(candidates))


def _median_runtime_ns(
    operation,
    *,
    warmups: int = _CALIBRATION_WARMUPS,
    repeats: int = _CALIBRATION_REPEATS,
) -> int:
    """Return median operation runtime in nanoseconds."""

    if warmups < 0:
        raise ValueError(
            "warmups must not be negative"
        )

    if repeats < 1:
        raise ValueError(
            "repeats must be at least 1"
        )

    for _ in range(warmups):
        operation()

    samples = []

    for _ in range(repeats):
        start_ns = perf_counter_ns()
        operation()
        end_ns = perf_counter_ns()

        samples.append(
            end_ns - start_ns
        )

    return int(median(samples))


def _select_fastest_thread_count(
    benchmark,
    *,
    runtime: str,
) -> tuple[int, dict[str, int]]:
    """
    Select the fastest stable candidate thread count.

    A smaller thread count wins ties within 5 percent to avoid
    selecting additional threads for negligible improvement.
    """

    timings: dict[int, int] = {}

    for thread_count in _thread_candidates():
        if runtime == "openmp":
            with apply_openmp_thread_policy(
                thread_count
            ):
                timings[thread_count] = (
                    _median_runtime_ns(benchmark)
                )

        elif runtime == "openblas":
            with apply_openblas_thread_policy(
                thread_count
            ):
                timings[thread_count] = (
                    _median_runtime_ns(benchmark)
                )

        else:
            raise ValueError(
                f"Unsupported runtime: {runtime}"
            )

    fastest_time = min(timings.values())

    acceptable_counts = [
        thread_count
        for thread_count, runtime_ns in timings.items()
        if runtime_ns <= fastest_time * 1.05
    ]

    selected = min(acceptable_counts)

    serialized_timings = {
        str(thread_count): runtime_ns
        for thread_count, runtime_ns in timings.items()
    }

    return selected, serialized_timings
def _build_lexical_calibration_workload():
    """Build a deterministic lexical OpenMP workload."""

    rng = np.random.default_rng(
        _RANDOM_SEED
    )

    document_count = 4096
    shingles_per_document = 64

    corpus_matrix = rng.integers(
        low=0,
        high=np.iinfo(np.int64).max,
        size=(
            document_count,
            shingles_per_document,
        ),
        dtype=np.int64,
    )

    corpus_matrix.sort(axis=1)

    corpus_shingles = np.ascontiguousarray(
        corpus_matrix.reshape(-1),
        dtype=np.int64,
    )

    document_offsets = np.arange(
        0,
        corpus_shingles.size + 1,
        shingles_per_document,
        dtype=np.int64,
    )

    candidate = np.sort(
        rng.integers(
            low=0,
            high=np.iinfo(np.int64).max,
            size=64,
            dtype=np.int64,
        )
    )

    candidate = np.ascontiguousarray(
        candidate,
        dtype=np.int64,
    )

    return (
        candidate,
        corpus_shingles,
        document_offsets,
    )


def _build_vector_calibration_workload():
    """Build deterministic vector calibration data."""

    rng = np.random.default_rng(
        _RANDOM_SEED
    )

    vector_count = 20_000
    dimension = 256
    query_count = 16
    k = 10

    database = np.asfortranarray(
        rng.standard_normal(
            (
                dimension,
                vector_count,
            ),
            dtype=np.float32,
        )
    )

    query = np.ascontiguousarray(
        rng.standard_normal(
            dimension,
            dtype=np.float32,
        )
    )

    queries = np.asfortranarray(
        rng.standard_normal(
            (
                dimension,
                query_count,
            ),
            dtype=np.float32,
        )
    )

    return (
        query,
        queries,
        database,
        k,
    )
def _calibrate_thread_policies() -> dict[str, Any]:
    """Calibrate representative OpenMP and OpenBLAS thread counts."""

    (
        lexical_candidate,
        lexical_corpus,
        lexical_offsets,
    ) = _build_lexical_calibration_workload()

    (
        vector_query,
        vector_queries,
        vector_database,
        vector_k,
    ) = _build_vector_calibration_workload()

    lexical_threads, lexical_timings = (
        _select_fastest_thread_count(
            lambda: _cpu.jaccard_corpus_openmp(
                lexical_candidate,
                lexical_corpus,
                lexical_offsets,
            ),
            runtime="openmp",
        )
    )

    l2_threads, l2_timings = (
        _select_fastest_thread_count(
            lambda: _cpu.l2_search_openmp(
                vector_query,
                vector_database,
                vector_k,
            ),
            runtime="openmp",
        )
    )

    sgemv_threads, sgemv_timings = (
        _select_fastest_thread_count(
            lambda: _cpu.ip_search_blas(
                vector_query,
                vector_database,
                vector_k,
            ),
            runtime="openblas",
        )
    )

    sgemm_threads, sgemm_timings = (
        _select_fastest_thread_count(
            lambda: _cpu.ip_search_batch_blas(
                vector_queries,
                vector_database,
                vector_k,
            ),
            runtime="openblas",
        )
    )

    return {
        "lexical_openmp": lexical_threads,
        "l2_openmp": l2_threads,
        "sgemv_openblas": sgemv_threads,
        "sgemm_openblas": sgemm_threads,
        "timings_ns": {
            "lexical_openmp": lexical_timings,
            "l2_openmp": l2_timings,
            "sgemv_openblas": sgemv_timings,
            "sgemm_openblas": sgemm_timings,
        },
    }

def calibrate(
    *,
    output_path: str | os.PathLike[str] | None = None,
) -> Path:
    """
    Run bounded CPU calibration and write a machine profile.

    Version 0.1.0 calibrates thread policies while retaining the
    shipped benchmark-informed workload crossover rules.
    """

    print("Starting CPU calibration...")
    print(
        "Candidate thread counts: "
        f"{_thread_candidates()}"
    )

    thread_results = (
        _calibrate_thread_policies()
    )

    shipped_policy = get_shipped_policy()

    decisions = {
        "lexical": deepcopy(
            shipped_policy["lexical"]
        ),
        "vector": deepcopy(
            shipped_policy["vector"]
        ),
    }

    lexical_threads = thread_results[
        "lexical_openmp"
    ]
    l2_threads = thread_results[
        "l2_openmp"
    ]
    sgemv_threads = thread_results[
        "sgemv_openblas"
    ]
    sgemm_threads = thread_results[
        "sgemm_openblas"
    ]

    decisions["lexical"]["single"][
        "thread_policy"
    ] = lexical_threads

    decisions["lexical"]["batch"][
        "thread_policy"
    ] = lexical_threads

    decisions["vector"]["ip"]["single"][
        "thread_policy"
    ] = sgemv_threads

    decisions["vector"]["ip"]["batch"][
        "repeated_single_thread_policy"
    ] = sgemv_threads

    decisions["vector"]["ip"]["batch"][
        "thread_policy"
    ] = sgemm_threads

    decisions["vector"]["l2"]["single"][
        "thread_policy"
    ] = l2_threads

    decisions["vector"]["l2"]["batch"][
        "thread_policy"
    ] = l2_threads

    decisions["calibration"] = {
        "method": "bounded-thread-calibration-v1",
        "repeats": _CALIBRATION_REPEATS,
        "warmups": _CALIBRATION_WARMUPS,
        "thread_candidates": list(
            _thread_candidates()
        ),
        "selected_threads": {
            "lexical_openmp": lexical_threads,
            "l2_openmp": l2_threads,
            "sgemv_openblas": sgemv_threads,
            "sgemm_openblas": sgemm_threads,
        },
        "timings_ns": thread_results[
            "timings_ns"
        ],
        "crossover_source": (
            "shipped_cpu_fallback_v1"
        ),
    }

    profile = build_profile(
        decisions
    )

    written_path = write_profile_atomic(
        profile,
        output_path,
    )

    print("Selected thread counts:")
    print(
        f"  Lexical OpenMP: {lexical_threads}"
    )
    print(
        f"  L2 OpenMP: {l2_threads}"
    )
    print(
        f"  SGEMV OpenBLAS: {sgemv_threads}"
    )
    print(
        f"  SGEMM OpenBLAS: {sgemm_threads}"
    )

    return written_path

def _create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="fortran-nlp-calibrate",
        description=(
            "Create and manage machine-specific CPU calibration "
            "profiles for fortran-nlp-kernels."
        ),
    )

    action_group = parser.add_mutually_exclusive_group()

    action_group.add_argument(
        "--show",
        action="store_true",
        help=(
            "Show the active compatible calibration profile."
        ),
    )

    action_group.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete the selected user calibration profile."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Use a custom profile path instead of the default "
            "user configuration path."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the calibration command-line interface."""

    parser = _create_argument_parser()
    arguments = parser.parse_args(argv)

    profile_path = get_profile_path(
        arguments.output
    )

    if arguments.show:
        result = load_profile(profile_path)

        if result.profile is None:
            print(
                f"No active compatible profile: "
                f"{result.reason}"
            )
            print(f"Profile path: {result.path}")
            return 1

        print(
            json.dumps(
                result.profile,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    if arguments.reset:
        removed = reset_profile(profile_path)

        if removed:
            print(
                "Removed CPU calibration profile: "
                f"{profile_path}"
            )
        else:
            print(
                "No CPU calibration profile existed at: "
                f"{profile_path}"
            )

        return 0

    try:
        written_path = calibrate(
            output_path=profile_path,
        )
    except NotImplementedError as error:
        parser.error(str(error))

    print(
        "CPU calibration profile written to: "
        f"{written_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())