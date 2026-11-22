#!/usr/bin/env python3
"""Benchmark direct CUDA L2 against cuBLAS norm-expansion L2.

This measures the existing public API implementations:

* cuda_fortran: direct sum((q - x) ** 2) CUDA Fortran kernel
* cuda_cublas: squared norms + SGEMV/SGEMM + L2 reconstruction kernel

The measured call includes the current public search path, including native
allocation, transfers, GPU scoring, score transfer, and deterministic CPU top-k.
Index construction and input generation are outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fortran_nlp import _cuda
from fortran_nlp.vector_search import FlatIndexL2


SUPPORTED_BACKENDS = ("cuda_fortran", "cuda_cublas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the two CUDA squared-L2 implementations."
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        required=True,
    )
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--database-size", type=int, required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Append the result row to this CSV file.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare one result with the other CUDA backend before timing.",
    )
    args = parser.parse_args()

    if args.dimension < 1:
        parser.error("--dimension must be at least 1")
    if args.database_size < 1:
        parser.error("--database-size must be at least 1")
    if args.query_count < 1:
        parser.error("--query-count must be at least 1")
    if args.k < 1 or args.k > args.database_size:
        parser.error("--k must be between 1 and --database-size")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    return args


def gpu_metadata() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, compute_capability, driver_version, memory_mib = [
            part.strip() for part in output.split(",", maxsplit=3)
        ]
        return {
            "gpu_name": name,
            "compute_capability": compute_capability,
            "driver_version": driver_version,
            "gpu_memory_mib": memory_mib,
        }
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {
            "gpu_name": "unknown",
            "compute_capability": "unknown",
            "driver_version": "unknown",
            "gpu_memory_mib": "unknown",
        }


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def run_search(
    index: FlatIndexL2,
    query: np.ndarray,
    queries: np.ndarray,
    query_count: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if query_count == 1:
        return index.search(query, k=k)
    return index.search_batch(queries, k=k)


def verify_results(
    database: np.ndarray,
    query: np.ndarray,
    queries: np.ndarray,
    query_count: int,
    k: int,
    measured_backend: str,
) -> None:
    reference_backend = (
        "cuda_cublas" if measured_backend == "cuda_fortran" else "cuda_fortran"
    )
    measured_index = FlatIndexL2(database, backend=measured_backend)
    reference_index = FlatIndexL2(database, backend=reference_backend)

    measured_values, measured_indices = run_search(
        measured_index, query, queries, query_count, k
    )
    reference_values, reference_indices = run_search(
        reference_index, query, queries, query_count, k
    )

    np.testing.assert_array_equal(measured_indices, reference_indices)
    np.testing.assert_allclose(
        measured_values,
        reference_values,
        rtol=2.0e-4,
        atol=2.0e-3,
    )


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    required_bindings = (
        "l2_search_cuda",
        "l2_search_batch_cuda",
        "l2_search_cublas",
        "l2_search_batch_cublas",
    )
    missing = [name for name in required_bindings if not hasattr(_cuda, name)]
    if missing:
        raise RuntimeError(f"Missing CUDA bindings: {missing}")

    rng = np.random.default_rng(args.seed)
    database = rng.standard_normal(
        (args.database_size, args.dimension),
        dtype=np.float32,
    )
    queries = rng.standard_normal(
        (args.query_count, args.dimension),
        dtype=np.float32,
    )
    query = np.ascontiguousarray(queries[0], dtype=np.float32)

    if args.verify:
        verify_results(
            database,
            query,
            queries,
            args.query_count,
            args.k,
            args.backend,
        )

    index = FlatIndexL2(database, backend=args.backend)

    for _ in range(args.warmups):
        run_search(index, query, queries, args.query_count, args.k)

    elapsed_ms: list[float] = []
    result_values: np.ndarray | None = None
    result_indices: np.ndarray | None = None

    for _ in range(args.repeats):
        start_ns = time.perf_counter_ns()
        result_values, result_indices = run_search(
            index,
            query,
            queries,
            args.query_count,
            args.k,
        )
        end_ns = time.perf_counter_ns()
        elapsed_ms.append((end_ns - start_ns) / 1_000_000.0)

    assert result_values is not None
    assert result_indices is not None

    median_ms = statistics.median(elapsed_ms)
    total_pairs = args.database_size * args.query_count
    pairs_per_second = total_pairs / (median_ms / 1_000.0)
    queries_per_second = args.query_count / (median_ms / 1_000.0)

    row: dict[str, object] = {
        "backend": args.backend,
        "dimension": args.dimension,
        "database_size": args.database_size,
        "query_count": args.query_count,
        "k": args.k,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "median_ms": median_ms,
        "minimum_ms": min(elapsed_ms),
        "p25_ms": percentile(elapsed_ms, 25.0),
        "p75_ms": percentile(elapsed_ms, 75.0),
        "queries_per_second": queries_per_second,
        "pairs_per_second": pairs_per_second,
        "result_shape": "x".join(str(value) for value in result_values.shape),
        "python_version": platform.python_version(),
        **gpu_metadata(),
    }

    print(json.dumps(row, indent=2))

    if args.output_csv is not None:
        append_csv(args.output_csv, row)
        print(f"Appended result to: {args.output_csv}")


if __name__ == "__main__":
    main()