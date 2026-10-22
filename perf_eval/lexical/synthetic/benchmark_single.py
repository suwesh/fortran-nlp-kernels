from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import socket
import subprocess
from time import perf_counter_ns

import numpy as np

import fortran_nlp
from fortran_nlp import _cpu

from generate_workloads import (
    LexicalWorkload,
    generate_lexical_workload,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]

DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "perf_eval"
    / "results"
    / "raw"
    / "lexical_single_raw.csv"
)


CSV_FIELDS = [
    "timestamp_utc",
    "experiment_id",
    "backend",
    "thread_count",
    "seed",
    "length_mode",
    "target_overlap",
    "actual_mean_overlap",
    "candidate_shingles",
    "base_document_count",
    "replication_factor",
    "document_count",
    "total_corpus_shingles",
    "mean_document_shingles",
    "minimum_document_shingles",
    "maximum_document_shingles",
    "warmup_runs",
    "measured_runs",
    "measured_run_index",
    "elapsed_ns",
    "documents_per_second",
    "corpus_shingles_per_second",
    "package_version",
    "python_version",
    "numpy_version",
    "host_name",
    "git_commit",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one lexical Jaccard workload "
            "using one backend."
        )
    )

    parser.add_argument(
        "--backend",
        choices=[
            "serial",
            "openmp",
        ],
        required=True,
    )

    parser.add_argument(
        "--base-document-count",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--replication-factor",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--shingles-per-document",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--candidate-shingles",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--target-overlap",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--length-mode",
        choices=[
            "fixed",
            "variable",
        ],
        default="fixed",
    )

    parser.add_argument(
        "--minimum-length-ratio",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261022,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.base_document_count < 1:
        raise ValueError(
            "base-document-count must be at least 1"
        )

    if arguments.replication_factor < 1:
        raise ValueError(
            "replication-factor must be at least 1"
        )

    if arguments.shingles_per_document < 1:
        raise ValueError(
            "shingles-per-document must be at least 1"
        )

    if arguments.candidate_shingles < 1:
        raise ValueError(
            "candidate-shingles must be at least 1"
        )

    if not 0.0 <= arguments.target_overlap <= 1.0:
        raise ValueError(
            "target-overlap must be between 0 and 1"
        )

    if arguments.warmup_runs < 0:
        raise ValueError(
            "warmup-runs must not be negative"
        )

    if (
        arguments.measured_runs is not None
        and arguments.measured_runs < 1
    ):
        raise ValueError(
            "measured-runs must be at least 1"
        )


def choose_measured_runs(
    document_count: int,
) -> int:
    if document_count <= 1_000:
        return 500

    if document_count <= 10_000:
        return 200

    if document_count <= 100_000:
        return 50

    return 20


def replicate_workload(
    workload: LexicalWorkload,
    replication_factor: int,
) -> LexicalWorkload:
    if replication_factor == 1:
        return workload

    base_document_count = workload.document_count
    base_corpus_count = workload.corpus_shingle_count

    replicated_lengths = np.tile(
        workload.document_lengths,
        replication_factor,
    )

    replicated_offsets = np.empty(
        replicated_lengths.size + 1,
        dtype=np.int64,
    )

    replicated_offsets[0] = 0

    np.cumsum(
        replicated_lengths,
        dtype=np.int64,
        out=replicated_offsets[1:],
    )

    replicated_corpus = np.tile(
        workload.corpus_shingles,
        replication_factor,
    )

    expected_corpus_count = (
        base_corpus_count
        * replication_factor
    )

    expected_document_count = (
        base_document_count
        * replication_factor
    )

    if replicated_corpus.size != expected_corpus_count:
        raise AssertionError(
            "replicated corpus size is inconsistent"
        )

    if replicated_lengths.size != expected_document_count:
        raise AssertionError(
            "replicated document count is inconsistent"
        )

    return LexicalWorkload(
        candidate_shingles=np.ascontiguousarray(
            workload.candidate_shingles,
            dtype=np.int64,
        ),
        corpus_shingles=np.ascontiguousarray(
            replicated_corpus,
            dtype=np.int64,
        ),
        document_offsets=np.ascontiguousarray(
            replicated_offsets,
            dtype=np.int64,
        ),
        document_lengths=np.ascontiguousarray(
            replicated_lengths,
            dtype=np.int64,
        ),
        target_overlap=workload.target_overlap,
        seed=workload.seed,
    )


def calculate_mean_actual_overlap(
    workload: LexicalWorkload,
) -> float:
    candidate_size = (
        workload.candidate_shingle_count
    )

    intersections = np.empty(
        workload.document_count,
        dtype=np.int64,
    )

    candidate = workload.candidate_shingles

    for document_index in range(
        workload.document_count
    ):
        start = int(
            workload.document_offsets[
                document_index
            ]
        )

        end = int(
            workload.document_offsets[
                document_index + 1
            ]
        )

        document = workload.corpus_shingles[
            start:end
        ]

        intersections[document_index] = (
            np.intersect1d(
                candidate,
                document,
                assume_unique=True,
            ).size
        )

    unions = (
        candidate_size
        + workload.document_lengths
        - intersections
    )

    scores = np.divide(
        intersections,
        unions,
        out=np.zeros(
            workload.document_count,
            dtype=np.float64,
        ),
        where=unions != 0,
    )

    return float(np.mean(scores))


def run_backend(
    backend: str,
    workload: LexicalWorkload,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if backend == "serial":
        return _cpu.jaccard_corpus_serial(
            workload.candidate_shingles,
            workload.corpus_shingles,
            workload.document_offsets,
        )

    if backend == "openmp":
        return _cpu.jaccard_corpus_openmp(
            workload.candidate_shingles,
            workload.corpus_shingles,
            workload.document_offsets,
        )

    raise RuntimeError(
        f"Unsupported backend state: {backend}"
    )


def verify_backend(
    backend: str,
    workload: LexicalWorkload,
) -> None:
    serial_result = _cpu.jaccard_corpus_serial(
        workload.candidate_shingles,
        workload.corpus_shingles,
        workload.document_offsets,
    )

    actual_result = run_backend(
        backend,
        workload,
    )

    if not np.array_equal(
        serial_result[0],
        actual_result[0],
    ):
        raise AssertionError(
            "backend intersections differ from serial"
        )

    if not np.array_equal(
        serial_result[1],
        actual_result[1],
    ):
        raise AssertionError(
            "backend unions differ from serial"
        )

    if not np.allclose(
        serial_result[2],
        actual_result[2],
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise AssertionError(
            "backend scores differ from serial"
        )


def warm_up(
    backend: str,
    workload: LexicalWorkload,
    warmup_runs: int,
) -> None:
    for _ in range(warmup_runs):
        run_backend(
            backend,
            workload,
        )


def measure(
    backend: str,
    workload: LexicalWorkload,
    measured_runs: int,
) -> np.ndarray:
    timings = np.empty(
        measured_runs,
        dtype=np.int64,
    )

    for run_index in range(measured_runs):
        start_time = perf_counter_ns()

        run_backend(
            backend,
            workload,
        )

        end_time = perf_counter_ns()

        timings[run_index] = (
            end_time - start_time
        )

    return timings


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return "unknown"


def write_results(
    *,
    arguments: argparse.Namespace,
    workload: LexicalWorkload,
    measured_runs: int,
    timings: np.ndarray,
    actual_mean_overlap: float,
) -> None:
    output_path = arguments.output.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = output_path.exists()

    thread_count = os.environ.get(
        "OMP_NUM_THREADS",
        "runtime-default",
    )

    experiment_id = (
        f"lexical-single-"
        f"{arguments.backend}-"
        f"docs{workload.document_count}-"
        f"docsh{arguments.shingles_per_document}-"
        f"candsh{arguments.candidate_shingles}-"
        f"overlap{arguments.target_overlap:.2f}-"
        f"{arguments.length_mode}-"
        f"rf{arguments.replication_factor}-"
        f"threads{thread_count}-"
        f"seed{arguments.seed}"
    )

    common_values = {
        "timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "experiment_id": experiment_id,
        "backend": arguments.backend,
        "thread_count": thread_count,
        "seed": arguments.seed,
        "length_mode": arguments.length_mode,
        "target_overlap": (
            arguments.target_overlap
        ),
        "actual_mean_overlap": (
            actual_mean_overlap
        ),
        "candidate_shingles": (
            workload.candidate_shingle_count
        ),
        "base_document_count": (
            arguments.base_document_count
        ),
        "replication_factor": (
            arguments.replication_factor
        ),
        "document_count": workload.document_count,
        "total_corpus_shingles": (
            workload.corpus_shingle_count
        ),
        "mean_document_shingles": float(
            np.mean(workload.document_lengths)
        ),
        "minimum_document_shingles": int(
            np.min(workload.document_lengths)
        ),
        "maximum_document_shingles": int(
            np.max(workload.document_lengths)
        ),
        "warmup_runs": arguments.warmup_runs,
        "measured_runs": measured_runs,
        "package_version": (
            fortran_nlp.__version__
        ),
        "python_version": (
            platform.python_version()
        ),
        "numpy_version": np.__version__,
        "host_name": socket.gethostname(),
        "git_commit": get_git_commit(),
    }

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        for run_index, elapsed_ns in enumerate(
            timings
        ):
            elapsed_seconds = (
                int(elapsed_ns) / 1_000_000_000
            )

            row = dict(common_values)

            row.update(
                {
                    "measured_run_index": (
                        run_index
                    ),
                    "elapsed_ns": int(elapsed_ns),
                    "documents_per_second": (
                        workload.document_count
                        / elapsed_seconds
                    ),
                    "corpus_shingles_per_second": (
                        workload.corpus_shingle_count
                        / elapsed_seconds
                    ),
                }
            )

            writer.writerow(row)


def print_summary(
    *,
    arguments: argparse.Namespace,
    workload: LexicalWorkload,
    measured_runs: int,
    timings: np.ndarray,
    actual_mean_overlap: float,
) -> None:
    median_ns = int(np.median(timings))
    mean_ns = float(np.mean(timings))
    standard_deviation_ns = float(
        np.std(timings)
    )

    p25_ns, p75_ns, p95_ns = np.percentile(
        timings,
        [
            25,
            75,
            95,
        ],
    )

    coefficient_of_variation = (
        standard_deviation_ns / mean_ns
        if mean_ns > 0
        else 0.0
    )

    print("Lexical single-candidate benchmark")
    print("=" * 64)

    print(f"Backend: {arguments.backend}")
    print(
        "OMP_NUM_THREADS: "
        f"{os.environ.get('OMP_NUM_THREADS', 'runtime-default')}"
    )

    print(
        "OPENBLAS_NUM_THREADS: "
        f"{os.environ.get('OPENBLAS_NUM_THREADS', 'runtime-default')}"
    )

    print(
        f"Documents: {workload.document_count}"
    )

    print(
        "Corpus shingles: "
        f"{workload.corpus_shingle_count}"
    )

    print(
        "Candidate shingles: "
        f"{workload.candidate_shingle_count}"
    )

    print(
        "Document shingles min/mean/max: "
        f"{int(np.min(workload.document_lengths))}/"
        f"{float(np.mean(workload.document_lengths)):.2f}/"
        f"{int(np.max(workload.document_lengths))}"
    )

    print(
        "Target overlap: "
        f"{arguments.target_overlap:.6f}"
    )

    print(
        "Actual mean overlap: "
        f"{actual_mean_overlap:.6f}"
    )

    print(
        f"Length mode: {arguments.length_mode}"
    )

    print(
        "Replication factor: "
        f"{arguments.replication_factor}"
    )

    print(
        f"Warm-up runs: {arguments.warmup_runs}"
    )

    print(
        f"Measured runs: {measured_runs}"
    )

    print("-" * 64)

    print(
        f"Median: {median_ns / 1_000.0:.3f} us"
    )

    print(
        f"Mean: {mean_ns / 1_000.0:.3f} us"
    )

    print(
        "Standard deviation: "
        f"{standard_deviation_ns / 1_000.0:.3f} us"
    )

    print(
        f"P25: {p25_ns / 1_000.0:.3f} us"
    )

    print(
        f"P75: {p75_ns / 1_000.0:.3f} us"
    )

    print(
        f"P95: {p95_ns / 1_000.0:.3f} us"
    )

    print(
        "Coefficient of variation: "
        f"{coefficient_of_variation:.6f}"
    )

    median_seconds = (
        median_ns / 1_000_000_000
    )

    print(
        "Median documents/s: "
        f"{workload.document_count / median_seconds:,.2f}"
    )

    print(
        "Median corpus shingles/s: "
        f"{workload.corpus_shingle_count / median_seconds:,.2f}"
    )

    print("=" * 64)

    print(
        f"Raw results: {arguments.output.resolve()}"
    )


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    base_workload = generate_lexical_workload(
        document_count=(
            arguments.base_document_count
        ),
        shingles_per_document=(
            arguments.shingles_per_document
        ),
        candidate_shingle_count=(
            arguments.candidate_shingles
        ),
        target_overlap=(
            arguments.target_overlap
        ),
        variable_lengths=(
            arguments.length_mode == "variable"
        ),
        minimum_length_ratio=(
            arguments.minimum_length_ratio
        ),
        seed=arguments.seed,
    )

    workload = replicate_workload(
        base_workload,
        arguments.replication_factor,
    )

    measured_runs = (
        arguments.measured_runs
        if arguments.measured_runs is not None
        else choose_measured_runs(
            workload.document_count
        )
    )

    actual_mean_overlap = (
        calculate_mean_actual_overlap(
            workload
        )
    )

    verify_backend(
        arguments.backend,
        workload,
    )

    warm_up(
        arguments.backend,
        workload,
        arguments.warmup_runs,
    )

    timings = measure(
        arguments.backend,
        workload,
        measured_runs,
    )

    write_results(
        arguments=arguments,
        workload=workload,
        measured_runs=measured_runs,
        timings=timings,
        actual_mean_overlap=(
            actual_mean_overlap
        ),
    )

    print_summary(
        arguments=arguments,
        workload=workload,
        measured_runs=measured_runs,
        timings=timings,
        actual_mean_overlap=(
            actual_mean_overlap
        ),
    )


if __name__ == "__main__":
    main()