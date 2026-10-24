from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "perf_eval/config/lexical_cpu_matrix.local.yaml"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/vector_real_batch_raw.csv"
)

METRICS = ["ip", "l2"]
K_VALUES = [1, 10, 50]
SELECTED_REPLICATION_FACTORS = [1, 10, 100, 250]
EXPECTED_QUERY_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
EXPECTED_PHYSICAL_THREADS = [1, 2, 4, 6]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def resolve_database_path(config: dict) -> Path | None:
    configured = config.get("vector_database", {}).get("output_path")
    if configured is None:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def build_execution_variants(
    physical_threads: list[int],
) -> list[tuple[str, int, int]]:
    """Return (method, OMP threads, OpenBLAS threads)."""
    variants: list[tuple[str, int, int]] = [
        ("repeated_serial", 1, 1),
    ]
    variants.extend(
        ("repeated_openmp", threads, 1)
        for threads in physical_threads
    )
    variants.extend(
        ("repeated_sgemv", 1, threads)
        for threads in physical_threads
    )
    variants.extend(
        ("native_sgemm", 1, threads)
        for threads in physical_threads
    )
    return variants


def completed_configurations(
    output_path: Path,
) -> set[tuple[str, str, int, int, int, int, int]]:
    if not output_path.exists():
        return set()

    required = {
        "metric",
        "method",
        "query_batch_size",
        "replication_factor",
        "k",
        "thread_count",
        "openblas_thread_count",
    }
    completed: set[
        tuple[str, str, int, int, int, int, int]
    ] = set()

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Existing vector batch CSV has no header")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Existing vector batch CSV has an incompatible schema. "
                "Missing columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            completed.add(
                (
                    row["metric"],
                    row["method"],
                    int(row["query_batch_size"]),
                    int(row["replication_factor"]),
                    int(row["k"]),
                    int(row["thread_count"]),
                    int(row["openblas_thread_count"]),
                )
            )
    return completed


def build_environment(
    *,
    method: str,
    omp_threads: int,
    openblas_threads: int,
    openmp_config: dict,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["OMP_NUM_THREADS"] = str(omp_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(openblas_threads)
    environment["OMP_DYNAMIC"] = str(
        openmp_config.get("dynamic", "FALSE")
    )

    if method == "repeated_openmp":
        environment["OMP_PROC_BIND"] = str(
            openmp_config.get("proc_bind", "close")
        )
        environment["OMP_PLACES"] = str(
            openmp_config.get("places", "cores")
        )
        environment["OMP_WAIT_POLICY"] = str(
            openmp_config.get("wait_policy", "active")
        )
    else:
        environment.pop("OMP_PROC_BIND", None)
        environment.pop("OMP_PLACES", None)
        environment.pop("OMP_WAIT_POLICY", None)

    return environment


def run_configuration(
    *,
    database_path: Path | None,
    output_path: Path,
    metric: str,
    method: str,
    query_batch_size: int,
    replication_factor: int,
    k: int,
    omp_threads: int,
    openblas_threads: int,
    openmp_config: dict,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "perf_eval.vector.benchmark_real_batch",
        "--output",
        str(output_path),
        "--metric",
        metric,
        "--method",
        method,
        "--query-batch-size",
        str(query_batch_size),
        "--replication-factor",
        str(replication_factor),
        "--k",
        str(k),
        "--skip-verification",
    ]
    if database_path is not None:
        command.extend(["--database", str(database_path)])

    environment = build_environment(
        method=method,
        omp_threads=omp_threads,
        openblas_threads=openblas_threads,
        openmp_config=openmp_config,
    )

    print(
        f"metric={metric} | batch={query_batch_size} | "
        f"rf={replication_factor} | k={k} | method={method} | "
        f"omp={omp_threads} | blas={openblas_threads}"
    )

    if dry_run:
        return

    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the real-vector physical-core batch matrix."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing formal batch CSV before starting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without executing benchmarks.",
    )
    arguments = parser.parse_args()

    config_path = arguments.config.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    config = load_config(config_path)

    configured_replication_factors = {
        int(value) for value in config["vector_replication_factors"]
    }
    for factor in SELECTED_REPLICATION_FACTORS:
        if factor not in configured_replication_factors:
            raise ValueError(
                f"Selected replication factor {factor} is absent from "
                "vector_replication_factors"
            )

    query_batch_sizes = [
        int(value) for value in config["vector_query_batch_sizes"]
    ]
    if query_batch_sizes != EXPECTED_QUERY_BATCH_SIZES:
        raise ValueError(
            "Expected vector_query_batch_sizes to be exactly "
            f"{EXPECTED_QUERY_BATCH_SIZES}; received {query_batch_sizes}"
        )

    physical_threads = [
        int(value) for value in config["physical_core_thread_counts"]
    ]
    if physical_threads != EXPECTED_PHYSICAL_THREADS:
        raise ValueError(
            "Expected physical_core_thread_counts to be exactly "
            f"{EXPECTED_PHYSICAL_THREADS}; received {physical_threads}"
        )

    execution_variants = build_execution_variants(physical_threads)
    database_path = resolve_database_path(config)
    openmp_config = config.get("openmp", {})

    if arguments.fresh and output_path.exists():
        output_path.unlink()

    completed = completed_configurations(output_path)
    total_configurations = (
        len(METRICS)
        * len(SELECTED_REPLICATION_FACTORS)
        * len(query_batch_sizes)
        * len(K_VALUES)
        * len(execution_variants)
    )

    executed = 0
    skipped = 0

    print("Real-vector physical-core batch matrix")
    print("=" * 80)
    print(f"Metrics: {METRICS}")
    print(f"Replication factors: {SELECTED_REPLICATION_FACTORS}")
    print(f"Query batch sizes: {query_batch_sizes}")
    print(f"k values: {K_VALUES}")
    print(f"Execution variants: {execution_variants}")
    print(f"Configurations: {total_configurations}")
    print(f"Output: {output_path}")
    print("=" * 80)

    for metric in METRICS:
        for replication_factor in SELECTED_REPLICATION_FACTORS:
            for query_batch_size in query_batch_sizes:
                for k in K_VALUES:
                    for method, omp_threads, blas_threads in execution_variants:
                        key = (
                            metric,
                            method,
                            query_batch_size,
                            replication_factor,
                            k,
                            omp_threads,
                            blas_threads,
                        )
                        if key in completed:
                            skipped += 1
                            continue

                        run_configuration(
                            database_path=database_path,
                            output_path=output_path,
                            metric=metric,
                            method=method,
                            query_batch_size=query_batch_size,
                            replication_factor=replication_factor,
                            k=k,
                            omp_threads=omp_threads,
                            openblas_threads=blas_threads,
                            openmp_config=openmp_config,
                            dry_run=arguments.dry_run,
                        )
                        executed += 1

    print("=" * 80)
    print(f"Executed configurations: {executed}")
    print(f"Skipped existing: {skipped}")
    if arguments.dry_run:
        print("Vector physical-core batch matrix: DRY RUN COMPLETE")
    else:
        print("Vector physical-core batch matrix: COMPLETE")


if __name__ == "__main__":
    main()
