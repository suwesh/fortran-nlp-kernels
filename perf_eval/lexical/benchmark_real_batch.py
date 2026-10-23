from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import yaml

import fortran_nlp
from fortran_nlp import _cpu
from fortran_nlp.lexical.preprocessing import encode_text_shingles


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "perf_eval/config/lexical_cpu_matrix.local.yaml"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/lexical_real_batch_raw.csv"
)

METHODS = (
    "repeated_serial",
    "repeated_openmp",
    "native_batch_openmp",
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def load_candidates(
    candidate_path: Path,
    *,
    shingle_size: int,
    batch_size: int,
) -> tuple[list[dict], list[np.ndarray]]:
    base_candidates: list[dict] = []
    with candidate_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            row["shingles"] = np.ascontiguousarray(
                encode_text_shingles(
                    row["text"],
                    shingle_size=shingle_size,
                ),
                dtype=np.int64,
            )
            base_candidates.append(row)

    if not base_candidates:
        raise ValueError("Candidate file contains no candidates")

    selected_metadata: list[dict] = []
    selected_shingles: list[np.ndarray] = []
    for index in range(batch_size):
        source = base_candidates[index % len(base_candidates)]
        selected_metadata.append(source)
        selected_shingles.append(source["shingles"])

    return selected_metadata, selected_shingles


def load_workload(
    config_path: Path,
    *,
    replication_factor: int,
    batch_size: int,
) -> tuple[
    list[dict],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    config = load_config(config_path)
    packed_path = resolve_path(config["packed_corpus"]["output_path"])
    candidate_path = resolve_path(config["candidates"]["input_path"])

    with np.load(packed_path, allow_pickle=False) as packed:
        base_corpus = np.ascontiguousarray(
            packed["corpus_shingles"], dtype=np.int64
        )
        base_offsets = np.ascontiguousarray(
            packed["document_offsets"], dtype=np.int64
        )
        shingle_size = int(packed["shingle_size"])

    base_lengths = np.diff(base_offsets)
    document_lengths = np.tile(base_lengths, replication_factor)
    document_offsets = np.empty(
        document_lengths.size + 1,
        dtype=np.int64,
    )
    document_offsets[0] = 0
    np.cumsum(
        document_lengths,
        dtype=np.int64,
        out=document_offsets[1:],
    )
    corpus_shingles = np.ascontiguousarray(
        np.tile(base_corpus, replication_factor),
        dtype=np.int64,
    )

    candidate_metadata, candidate_arrays = load_candidates(
        candidate_path,
        shingle_size=shingle_size,
        batch_size=batch_size,
    )

    candidate_lengths = np.array(
        [candidate.size for candidate in candidate_arrays],
        dtype=np.int64,
    )
    candidate_offsets = np.empty(batch_size + 1, dtype=np.int64)
    candidate_offsets[0] = 0
    np.cumsum(
        candidate_lengths,
        dtype=np.int64,
        out=candidate_offsets[1:],
    )
    packed_candidates = np.ascontiguousarray(
        np.concatenate(candidate_arrays),
        dtype=np.int64,
    )

    return (
        candidate_metadata,
        candidate_arrays,
        packed_candidates,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    )


def run_method(
    method: str,
    candidate_arrays: list[np.ndarray],
    packed_candidates: np.ndarray,
    candidate_offsets: np.ndarray,
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
):
    if method == "repeated_serial":
        return [
            _cpu.jaccard_corpus_serial(
                candidate,
                corpus_shingles,
                document_offsets,
            )
            for candidate in candidate_arrays
        ]

    if method == "repeated_openmp":
        return [
            _cpu.jaccard_corpus_openmp(
                candidate,
                corpus_shingles,
                document_offsets,
            )
            for candidate in candidate_arrays
        ]

    if method == "native_batch_openmp":
        return _cpu.jaccard_batch_corpus(
            packed_candidates,
            candidate_offsets,
            corpus_shingles,
            document_offsets,
        )

    raise RuntimeError(f"Unsupported method: {method}")


def verify_equivalence(
    candidate_arrays: list[np.ndarray],
    packed_candidates: np.ndarray,
    candidate_offsets: np.ndarray,
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
) -> None:
    serial = run_method(
        "repeated_serial",
        candidate_arrays,
        packed_candidates,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    )
    repeated_openmp = run_method(
        "repeated_openmp",
        candidate_arrays,
        packed_candidates,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    )
    batch = run_method(
        "native_batch_openmp",
        candidate_arrays,
        packed_candidates,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    )

    for index, serial_result in enumerate(serial):
        openmp_result = repeated_openmp[index]
        batch_result = (
            batch[0][:, index],
            batch[1][:, index],
            batch[2][:, index],
        )
        for actual in (openmp_result, batch_result):
            if not np.array_equal(serial_result[0], actual[0]):
                raise AssertionError("Intersection results differ")
            if not np.array_equal(serial_result[1], actual[1]):
                raise AssertionError("Union results differ")
            if not np.allclose(
                serial_result[2], actual[2], rtol=0.0, atol=1e-15
            ):
                raise AssertionError("Jaccard scores differ")


def choose_runs(document_count: int, batch_size: int) -> int:
    comparisons = document_count * batch_size
    if comparisons <= 10_000:
        return 300
    if comparisons <= 100_000:
        return 100
    if comparisons <= 1_000_000:
        return 30
    return 10


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark real-corpus batched Jaccard scoring using repeated "
            "single-candidate calls and the native batched OpenMP routine."
        )
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--replication-factor", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the pre-timing equivalence check.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if arguments.replication_factor < 1:
        raise ValueError("replication-factor must be at least 1")
    if arguments.warmup_runs < 0:
        raise ValueError("warmup-runs must not be negative")

    (
        candidate_metadata,
        candidate_arrays,
        packed_candidates,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    ) = load_workload(
        arguments.config.expanduser().resolve(),
        replication_factor=arguments.replication_factor,
        batch_size=arguments.batch_size,
    )

    document_count = int(document_offsets.size - 1)
    comparison_count = document_count * arguments.batch_size
    measured_runs = (
        arguments.measured_runs
        if arguments.measured_runs is not None
        else choose_runs(document_count, arguments.batch_size)
    )

    if not arguments.skip_verification:
        verify_equivalence(
            candidate_arrays,
            packed_candidates,
            candidate_offsets,
            corpus_shingles,
            document_offsets,
        )

    for _ in range(arguments.warmup_runs):
        run_method(
            arguments.method,
            candidate_arrays,
            packed_candidates,
            candidate_offsets,
            corpus_shingles,
            document_offsets,
        )

    timings = np.empty(measured_runs, dtype=np.int64)
    for run_index in range(measured_runs):
        start = perf_counter_ns()
        run_method(
            arguments.method,
            candidate_arrays,
            packed_candidates,
            candidate_offsets,
            corpus_shingles,
            document_offsets,
        )
        timings[run_index] = perf_counter_ns() - start

    input_bytes = (
        packed_candidates.nbytes
        + candidate_offsets.nbytes
        + corpus_shingles.nbytes
        + document_offsets.nbytes
    )
    if arguments.method == "native_batch_openmp":
        peak_output_bytes = comparison_count * 24
    else:
        # Repeated calls release each result before the next timed call ends;
        # only one candidate's three result arrays are simultaneously needed.
        peak_output_bytes = document_count * 24
    logical_output_bytes = comparison_count * 24

    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    fields = [
        "package_version",
        "method",
        "batch_size",
        "unique_base_candidate_count",
        "minimum_candidate_shingles",
        "median_candidate_shingles",
        "mean_candidate_shingles",
        "maximum_candidate_shingles",
        "replication_factor",
        "document_count",
        "corpus_shingles",
        "candidate_document_comparisons",
        "thread_count",
        "omp_dynamic",
        "omp_proc_bind",
        "omp_places",
        "omp_wait_policy",
        "warmup_runs",
        "measured_runs",
        "run_index",
        "elapsed_ns",
        "candidates_per_second",
        "comparisons_per_second",
        "corpus_shingle_visits_per_second",
        "estimated_input_bytes",
        "estimated_peak_output_bytes",
        "logical_output_bytes",
    ]

    candidate_lengths = np.array(
        [candidate.size for candidate in candidate_arrays],
        dtype=np.int64,
    )
    unique_base_candidates = len(
        {item["candidate_id"] for item in candidate_metadata}
    )

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        for run_index, elapsed_ns in enumerate(timings):
            elapsed_seconds = int(elapsed_ns) / 1_000_000_000
            writer.writerow(
                {
                    "package_version": fortran_nlp.__version__,
                    "method": arguments.method,
                    "batch_size": arguments.batch_size,
                    "unique_base_candidate_count": unique_base_candidates,
                    "minimum_candidate_shingles": int(candidate_lengths.min()),
                    "median_candidate_shingles": float(
                        np.median(candidate_lengths)
                    ),
                    "mean_candidate_shingles": float(candidate_lengths.mean()),
                    "maximum_candidate_shingles": int(candidate_lengths.max()),
                    "replication_factor": arguments.replication_factor,
                    "document_count": document_count,
                    "corpus_shingles": int(corpus_shingles.size),
                    "candidate_document_comparisons": comparison_count,
                    "thread_count": os.environ.get(
                        "OMP_NUM_THREADS", "runtime-default"
                    ),
                    "omp_dynamic": os.environ.get("OMP_DYNAMIC", "unset"),
                    "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "unset"),
                    "omp_places": os.environ.get("OMP_PLACES", "unset"),
                    "omp_wait_policy": os.environ.get(
                        "OMP_WAIT_POLICY", "unset"
                    ),
                    "warmup_runs": arguments.warmup_runs,
                    "measured_runs": measured_runs,
                    "run_index": run_index,
                    "elapsed_ns": int(elapsed_ns),
                    "candidates_per_second": (
                        arguments.batch_size / elapsed_seconds
                    ),
                    "comparisons_per_second": (
                        comparison_count / elapsed_seconds
                    ),
                    "corpus_shingle_visits_per_second": (
                        corpus_shingles.size
                        * arguments.batch_size
                        / elapsed_seconds
                    ),
                    "estimated_input_bytes": input_bytes,
                    "estimated_peak_output_bytes": peak_output_bytes,
                    "logical_output_bytes": logical_output_bytes,
                }
            )

    median_ns = float(np.median(timings))
    median_seconds = median_ns / 1_000_000_000
    print("Real-corpus lexical batch benchmark")
    print("=" * 64)
    print(f"Method: {arguments.method}")
    print(f"Batch size: {arguments.batch_size}")
    print(f"Threads: {os.environ.get('OMP_NUM_THREADS', 'runtime-default')}")
    print(f"Replication factor: {arguments.replication_factor}")
    print(f"Documents: {document_count:,}")
    print(f"Corpus shingles: {corpus_shingles.size:,}")
    print(f"Candidate-document comparisons: {comparison_count:,}")
    print(f"Measured runs: {measured_runs}")
    print("-" * 64)
    print(f"Median batch latency: {median_ns / 1_000_000:.6f} ms")
    print(
        "P95 batch latency: "
        f"{np.percentile(timings, 95) / 1_000_000:.6f} ms"
    )
    print(
        "Median candidates/s: "
        f"{arguments.batch_size / median_seconds:,.2f}"
    )
    print(
        "Median comparisons/s: "
        f"{comparison_count / median_seconds:,.2f}"
    )
    print(
        "Estimated input memory: "
        f"{input_bytes / (1024 ** 2):,.3f} MiB"
    )
    print(
        "Estimated peak output memory: "
        f"{peak_output_bytes / (1024 ** 2):,.3f} MiB"
    )
    print(
        "Logical batch output memory: "
        f"{logical_output_bytes / (1024 ** 2):,.3f} MiB"
    )
    print("-" * 64)
    print(f"Raw CSV: {output_path}")


if __name__ == "__main__":
    main()
