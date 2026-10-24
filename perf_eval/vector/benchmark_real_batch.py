from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np

import fortran_nlp
from fortran_nlp import _cpu


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = (
    REPOSITORY_ROOT
    / "perf_eval/data/packed/vector_real_database.npz"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/vector_real_batch_raw.csv"
)

METRICS = ("ip", "l2")
METHODS = (
    "repeated_serial",
    "repeated_openmp",
    "repeated_sgemv",
    "native_sgemm",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact real-vector batch search using repeated native "
            "single-query calls or one native SGEMM batch call."
        )
    )
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--query-batch-size", type=int, required=True)
    parser.add_argument("--replication-factor", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip repeated-single versus SGEMM equivalence validation.",
    )
    return parser.parse_args()


def load_workload(
    database_path: Path,
    *,
    query_batch_size: int,
    replication_factor: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, int, int]:
    with np.load(database_path, allow_pickle=False) as packed:
        base_rows = np.ascontiguousarray(
            packed["database_vectors"],
            dtype=np.float32,
        )

    base_vector_count, dimension = base_rows.shape

    # Deterministic production-derived query sequence. Query batches larger
    # than the base database cycle through the same real vectors.
    query_indices = np.arange(query_batch_size, dtype=np.int64)
    query_indices %= base_vector_count

    query_rows = np.ascontiguousarray(
        base_rows[query_indices],
        dtype=np.float32,
    )
    query_list = [
        np.ascontiguousarray(row, dtype=np.float32)
        for row in query_rows
    ]
    queries_native = np.asfortranarray(
        query_rows.T,
        dtype=np.float32,
    )

    # Native ABI layout: (dimension, vector_count), Fortran-contiguous.
    database_native = np.asfortranarray(
        np.tile(base_rows.T, (1, replication_factor)),
        dtype=np.float32,
    )

    expected_database_shape = (
        dimension,
        base_vector_count * replication_factor,
    )
    if database_native.shape != expected_database_shape:
        raise AssertionError(
            f"Unexpected database shape: {database_native.shape}"
        )
    if queries_native.shape != (dimension, query_batch_size):
        raise AssertionError(
            f"Unexpected query batch shape: {queries_native.shape}"
        )
    if not database_native.flags.f_contiguous:
        raise AssertionError("Native database must be Fortran-contiguous")
    if not queries_native.flags.f_contiguous:
        raise AssertionError("Native query batch must be Fortran-contiguous")

    return (
        database_native,
        query_list,
        queries_native,
        base_vector_count,
        dimension,
    )


def single_function(metric: str, method: str):
    if metric == "ip":
        return {
            "repeated_serial": _cpu.ip_search_serial,
            "repeated_openmp": _cpu.ip_search_openmp,
            "repeated_sgemv": _cpu.ip_search_blas,
        }[method]

    return {
        "repeated_serial": _cpu.l2_search_serial,
        "repeated_openmp": _cpu.l2_search_openmp,
        "repeated_sgemv": _cpu.l2_search_blas,
    }[method]


def batch_function(metric: str):
    if metric == "ip":
        return _cpu.ip_search_batch_blas
    return _cpu.l2_search_batch_blas


def run_method(
    *,
    metric: str,
    method: str,
    query_list: list[np.ndarray],
    queries_native: np.ndarray,
    database_native: np.ndarray,
    k: int,
):
    if method == "native_sgemm":
        return batch_function(metric)(
            queries_native,
            database_native,
            k,
        )

    function = single_function(metric, method)
    return [
        function(query, database_native, k)
        for query in query_list
    ]


def verify_equivalence(
    *,
    metric: str,
    query_list: list[np.ndarray],
    queries_native: np.ndarray,
    database_native: np.ndarray,
    k: int,
) -> None:
    repeated = run_method(
        metric=metric,
        method="repeated_sgemv",
        query_list=query_list,
        queries_native=queries_native,
        database_native=database_native,
        k=k,
    )
    batched = run_method(
        metric=metric,
        method="native_sgemm",
        query_list=query_list,
        queries_native=queries_native,
        database_native=database_native,
        k=k,
    )

    expected_shape = (k, len(query_list))
    if batched[0].shape != expected_shape:
        raise AssertionError(
            f"Unexpected SGEMM values shape: {batched[0].shape}"
        )
    if batched[1].shape != expected_shape:
        raise AssertionError(
            f"Unexpected SGEMM IDs shape: {batched[1].shape}"
        )

    for query_index, repeated_result in enumerate(repeated):
        batch_values = np.ascontiguousarray(batched[0][:, query_index])
        batch_ids = np.ascontiguousarray(batched[1][:, query_index])

        if not np.array_equal(repeated_result[1], batch_ids):
            raise AssertionError(
                f"SGEMV/SGEMM IDs differ for query column {query_index}"
            )
        if not np.allclose(
            repeated_result[0],
            batch_values,
            rtol=1.0e-5,
            atol=2.0e-5,
        ):
            maximum_error = float(
                np.max(np.abs(repeated_result[0] - batch_values))
            )
            raise AssertionError(
                "SGEMV/SGEMM values differ for query column "
                f"{query_index}; maximum error={maximum_error}"
            )


def choose_runs(vector_count: int, query_batch_size: int) -> int:
    comparisons = vector_count * query_batch_size
    if comparisons <= 10_000:
        return 300
    if comparisons <= 100_000:
        return 100
    if comparisons <= 1_000_000:
        return 30
    return 10


def main() -> None:
    arguments = parse_arguments()

    if arguments.query_batch_size < 1:
        raise ValueError("query-batch-size must be at least 1")
    if arguments.replication_factor < 1:
        raise ValueError("replication-factor must be at least 1")
    if arguments.k < 1:
        raise ValueError("k must be at least 1")
    if arguments.warmup_runs < 0:
        raise ValueError("warmup-runs must not be negative")
    if arguments.measured_runs is not None and arguments.measured_runs < 1:
        raise ValueError("measured-runs must be at least 1")

    database_path = arguments.database.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()

    (
        database_native,
        query_list,
        queries_native,
        base_vector_count,
        dimension,
    ) = load_workload(
        database_path,
        query_batch_size=arguments.query_batch_size,
        replication_factor=arguments.replication_factor,
    )

    vector_count = int(database_native.shape[1])
    comparison_count = vector_count * arguments.query_batch_size

    if arguments.k > vector_count:
        raise ValueError("k cannot exceed the replicated database size")

    measured_runs = (
        arguments.measured_runs
        if arguments.measured_runs is not None
        else choose_runs(vector_count, arguments.query_batch_size)
    )

    if not arguments.skip_verification:
        verify_equivalence(
            metric=arguments.metric,
            query_list=query_list,
            queries_native=queries_native,
            database_native=database_native,
            k=arguments.k,
        )

    for _ in range(arguments.warmup_runs):
        run_method(
            metric=arguments.metric,
            method=arguments.method,
            query_list=query_list,
            queries_native=queries_native,
            database_native=database_native,
            k=arguments.k,
        )

    timings = np.empty(measured_runs, dtype=np.int64)
    for run_index in range(measured_runs):
        start = perf_counter_ns()
        run_method(
            metric=arguments.metric,
            method=arguments.method,
            query_list=query_list,
            queries_native=queries_native,
            database_native=database_native,
            k=arguments.k,
        )
        timings[run_index] = perf_counter_ns() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    input_bytes = int(
        database_native.nbytes + queries_native.nbytes
    )
    # Both repeated and native methods retain all returned top-k arrays until
    # the Python call completes, so use the same logical output estimate.
    output_bytes = int(
        arguments.query_batch_size
        * arguments.k
        * (4 + 8)
    )

    flop_estimate = int(
        2
        * dimension
        * vector_count
        * arguments.query_batch_size
    )

    fields = [
        "package_version",
        "metric",
        "method",
        "query_batch_size",
        "dimension",
        "k",
        "base_vector_count",
        "replication_factor",
        "vector_count",
        "query_vector_comparisons",
        "thread_count",
        "openblas_thread_count",
        "omp_dynamic",
        "omp_proc_bind",
        "omp_places",
        "omp_wait_policy",
        "warmup_runs",
        "measured_runs",
        "run_index",
        "elapsed_ns",
        "queries_per_second",
        "vectors_scored_per_second",
        "comparisons_per_second",
        "floating_point_operations_estimate",
        "estimated_gflops",
        "estimated_input_bytes",
        "estimated_output_bytes",
    ]

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        for run_index, elapsed_ns in enumerate(timings):
            elapsed_seconds = int(elapsed_ns) / 1_000_000_000
            writer.writerow(
                {
                    "package_version": fortran_nlp.__version__,
                    "metric": arguments.metric,
                    "method": arguments.method,
                    "query_batch_size": arguments.query_batch_size,
                    "dimension": dimension,
                    "k": arguments.k,
                    "base_vector_count": base_vector_count,
                    "replication_factor": arguments.replication_factor,
                    "vector_count": vector_count,
                    "query_vector_comparisons": comparison_count,
                    "thread_count": os.environ.get(
                        "OMP_NUM_THREADS", "runtime-default"
                    ),
                    "openblas_thread_count": os.environ.get(
                        "OPENBLAS_NUM_THREADS", "runtime-default"
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
                    "queries_per_second": (
                        arguments.query_batch_size / elapsed_seconds
                    ),
                    "vectors_scored_per_second": (
                        comparison_count / elapsed_seconds
                    ),
                    "comparisons_per_second": (
                        comparison_count / elapsed_seconds
                    ),
                    "floating_point_operations_estimate": flop_estimate,
                    "estimated_gflops": (
                        flop_estimate / elapsed_seconds / 1_000_000_000
                    ),
                    "estimated_input_bytes": input_bytes,
                    "estimated_output_bytes": output_bytes,
                }
            )

    median_ns = float(np.median(timings))
    median_seconds = median_ns / 1_000_000_000

    print("Real-vector batch benchmark")
    print("=" * 72)
    print(f"Metric: {arguments.metric}")
    print(f"Method: {arguments.method}")
    print(f"Query batch size: {arguments.query_batch_size}")
    print(f"Dimension: {dimension}")
    print(f"k: {arguments.k}")
    print(f"Replication factor: {arguments.replication_factor}")
    print(f"Vectors: {vector_count:,}")
    print(f"Query-vector comparisons: {comparison_count:,}")
    print(
        "OMP_NUM_THREADS: "
        f"{os.environ.get('OMP_NUM_THREADS', 'runtime-default')}"
    )
    print(
        "OPENBLAS_NUM_THREADS: "
        f"{os.environ.get('OPENBLAS_NUM_THREADS', 'runtime-default')}"
    )
    print(f"Measured runs: {measured_runs}")
    print("-" * 72)
    print(f"Median batch latency: {median_ns / 1_000_000:.6f} ms")
    print(
        "P95 batch latency: "
        f"{np.percentile(timings, 95) / 1_000_000:.6f} ms"
    )
    print(
        "Median queries/s: "
        f"{arguments.query_batch_size / median_seconds:,.2f}"
    )
    print(
        "Median comparisons/s: "
        f"{comparison_count / median_seconds:,.2f}"
    )
    print(
        "Estimated median GFLOP/s: "
        f"{flop_estimate / median_seconds / 1_000_000_000:,.3f}"
    )
    print(
        "Estimated input memory: "
        f"{input_bytes / (1024 ** 2):,.3f} MiB"
    )
    print("-" * 72)
    print(f"Raw CSV: {output_path}")


if __name__ == "__main__":
    main()
