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
    / "perf_eval/results/raw/vector_real_single_raw.csv"
)

METRICS = ("ip", "l2", "cosine")
BACKENDS = ("serial", "openmp", "blas")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one real vector query against one replicated database "
            "using one exact native CPU backend."
        )
    )
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--replication-factor", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the serial-reference equivalence check before timing.",
    )
    return parser.parse_args()


def load_workload(
    database_path: Path,
    *,
    query_index: int,
    replication_factor: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    with np.load(database_path, allow_pickle=False) as packed:
        base_rows = np.ascontiguousarray(
            packed["database_vectors"],
            dtype=np.float32,
        )

    base_vector_count, dimension = base_rows.shape

    if not 0 <= query_index < base_vector_count:
        raise ValueError(
            f"query-index must be in [0, {base_vector_count - 1}]"
        )

    query = np.ascontiguousarray(
        base_rows[query_index],
        dtype=np.float32,
    )

    # Native ABI layout is (dimension, vector_count), Fortran contiguous.
    # Replication is performed along the vector-count axis.
    database_native = np.asfortranarray(
        np.tile(base_rows.T, (1, replication_factor)),
        dtype=np.float32,
    )

    expected_shape = (
        dimension,
        base_vector_count * replication_factor,
    )
    if database_native.shape != expected_shape:
        raise AssertionError(
            f"Unexpected native database shape: {database_native.shape}"
        )
    if not database_native.flags.f_contiguous:
        raise AssertionError("Native database must be Fortran-contiguous")

    return query, database_native, base_vector_count, dimension


def run_backend(
    *,
    metric: str,
    backend: str,
    query: np.ndarray,
    database_native: np.ndarray,
    k: int,
):
    # The packed real vectors are unit-normalized, so cosine search uses
    # the exact inner-product kernels without timed normalization.
    effective_metric = "ip" if metric == "cosine" else metric

    if effective_metric == "ip":
        function = {
            "serial": _cpu.ip_search_serial,
            "openmp": _cpu.ip_search_openmp,
            "blas": _cpu.ip_search_blas,
        }[backend]
    else:
        function = {
            "serial": _cpu.l2_search_serial,
            "openmp": _cpu.l2_search_openmp,
            "blas": _cpu.l2_search_blas,
        }[backend]

    return function(query, database_native, k)


def choose_runs(vector_count: int) -> int:
    if vector_count <= 1_000:
        return 500
    if vector_count <= 10_000:
        return 200
    if vector_count <= 100_000:
        return 50
    return 20


def verify_backend(
    *,
    metric: str,
    backend: str,
    query: np.ndarray,
    database_native: np.ndarray,
    k: int,
) -> None:
    reference = run_backend(
        metric=metric,
        backend="serial",
        query=query,
        database_native=database_native,
        k=k,
    )
    actual = run_backend(
        metric=metric,
        backend=backend,
        query=query,
        database_native=database_native,
        k=k,
    )

    if not np.array_equal(reference[1], actual[1]):
        raise AssertionError("Backend top-k IDs differ from serial")
    if not np.allclose(
        reference[0],
        actual[0],
        rtol=1.0e-5,
        atol=2.0e-5,
    ):
        maximum_error = float(np.max(np.abs(reference[0] - actual[0])))
        raise AssertionError(
            f"Backend values differ from serial; maximum error={maximum_error}"
        )


def main() -> None:
    arguments = parse_arguments()

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

    query, database_native, base_vector_count, dimension = load_workload(
        database_path,
        query_index=arguments.query_index,
        replication_factor=arguments.replication_factor,
    )

    vector_count = int(database_native.shape[1])
    if arguments.k > vector_count:
        raise ValueError("k cannot exceed the replicated database size")

    measured_runs = (
        arguments.measured_runs
        if arguments.measured_runs is not None
        else choose_runs(vector_count)
    )

    if not arguments.skip_verification:
        verify_backend(
            metric=arguments.metric,
            backend=arguments.backend,
            query=query,
            database_native=database_native,
            k=arguments.k,
        )

    for _ in range(arguments.warmup_runs):
        run_backend(
            metric=arguments.metric,
            backend=arguments.backend,
            query=query,
            database_native=database_native,
            k=arguments.k,
        )

    timings = np.empty(measured_runs, dtype=np.int64)
    for run_index in range(measured_runs):
        start = perf_counter_ns()
        run_backend(
            metric=arguments.metric,
            backend=arguments.backend,
            query=query,
            database_native=database_native,
            k=arguments.k,
        )
        timings[run_index] = perf_counter_ns() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    input_bytes = int(query.nbytes + database_native.nbytes)
    output_bytes = int(arguments.k * (4 + 8))

    fields = [
        "package_version",
        "metric",
        "backend",
        "query_index",
        "dimension",
        "k",
        "base_vector_count",
        "replication_factor",
        "vector_count",
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
        "floating_point_operations_estimate",
        "estimated_gflops",
        "estimated_input_bytes",
        "estimated_output_bytes",
    ]

    # One dot product or squared-L2 comparison is approximately 2*dimension
    # floating-point operations. This estimate excludes top-k and wrapper work.
    flop_estimate = int(2 * dimension * vector_count)

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
                    "backend": arguments.backend,
                    "query_index": arguments.query_index,
                    "dimension": dimension,
                    "k": arguments.k,
                    "base_vector_count": base_vector_count,
                    "replication_factor": arguments.replication_factor,
                    "vector_count": vector_count,
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
                    "queries_per_second": 1.0 / elapsed_seconds,
                    "vectors_scored_per_second": vector_count / elapsed_seconds,
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

    print("Real-vector single-query benchmark")
    print("=" * 72)
    print(f"Metric: {arguments.metric}")
    print(f"Backend: {arguments.backend}")
    print(f"Query index: {arguments.query_index}")
    print(f"Dimension: {dimension}")
    print(f"k: {arguments.k}")
    print(f"Replication factor: {arguments.replication_factor}")
    print(f"Vectors: {vector_count:,}")
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
    print(f"Median latency: {median_ns / 1_000_000:.6f} ms")
    print(
        "P95 latency: "
        f"{np.percentile(timings, 95) / 1_000_000:.6f} ms"
    )
    print(
        "Median vectors scored/s: "
        f"{vector_count / median_seconds:,.2f}"
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
