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
    / "perf_eval/results/raw/vector_real_single_smt_raw.csv"
)

METRICS = ["ip", "l2"]
QUERY_INDICES = [0, 67, 208, 474]
K_VALUES = [1, 10, 50]

# Representative database sizes from the physical-core matrix:
# 4,750; 47,500; and 118,750 vectors.
SELECTED_REPLICATION_FACTORS = [10, 100, 250]
EXPECTED_SMT_THREADS = [8, 10, 12]

# Both handwritten OpenMP and OpenBLAS SGEMV receive SMT testing.
BACKENDS = ["openmp", "blas"]


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


def completed_configurations(
    output_path: Path,
) -> set[tuple[str, str, int, int, int, int, int]]:
    if not output_path.exists():
        return set()

    required_columns = {
        "metric",
        "backend",
        "query_index",
        "replication_factor",
        "k",
        "thread_count",
        "openblas_thread_count",
    }

    completed: set[
        tuple[str, str, int, int, int, int, int]
    ] = set()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Existing vector SMT CSV has no header")

        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Existing vector SMT CSV uses an incompatible schema. "
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            completed.add(
                (
                    row["metric"],
                    row["backend"],
                    int(row["query_index"]),
                    int(row["replication_factor"]),
                    int(row["k"]),
                    int(row["thread_count"]),
                    int(row["openblas_thread_count"]),
                )
            )

    return completed


def build_environment(
    *,
    backend: str,
    smt_threads: int,
    openmp_config: dict,
) -> tuple[dict[str, str], int, int]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    environment["OMP_DYNAMIC"] = str(
        openmp_config.get("dynamic", "FALSE")
    )

    if backend == "openmp":
        omp_threads = smt_threads
        openblas_threads = 1

        environment["OMP_NUM_THREADS"] = str(omp_threads)
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["OMP_PROC_BIND"] = str(
            openmp_config.get("proc_bind", "close")
        )
        environment["OMP_PLACES"] = "threads"
        environment["OMP_WAIT_POLICY"] = str(
            openmp_config.get("wait_policy", "active")
        )
    elif backend == "blas":
        omp_threads = 1
        openblas_threads = smt_threads

        environment["OMP_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = str(openblas_threads)
        environment.pop("OMP_PROC_BIND", None)
        environment.pop("OMP_PLACES", None)
        environment.pop("OMP_WAIT_POLICY", None)
    else:
        raise ValueError(f"Unsupported SMT backend: {backend}")

    return environment, omp_threads, openblas_threads


def run_configuration(
    *,
    database_path: Path | None,
    output_path: Path,
    metric: str,
    backend: str,
    query_index: int,
    replication_factor: int,
    k: int,
    smt_threads: int,
    openmp_config: dict,
    dry_run: bool,
) -> tuple[int, int]:
    environment, omp_threads, openblas_threads = build_environment(
        backend=backend,
        smt_threads=smt_threads,
        openmp_config=openmp_config,
    )

    command = [
        sys.executable,
        "-m",
        "perf_eval.vector.benchmark_real_single",
        "--output",
        str(output_path),
        "--metric",
        metric,
        "--backend",
        backend,
        "--query-index",
        str(query_index),
        "--replication-factor",
        str(replication_factor),
        "--k",
        str(k),
        "--skip-verification",
    ]

    if database_path is not None:
        command.extend(["--database", str(database_path)])

    print(
        f"metric={metric} | "
        f"query={query_index} | "
        f"rf={replication_factor} | "
        f"k={k} | "
        f"backend={backend} | "
        f"omp={omp_threads} | "
        f"blas={openblas_threads}"
    )

    if not dry_run:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )

    return omp_threads, openblas_threads


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run vector single-query SMT sensitivity for handwritten "
            "OpenMP and OpenBLAS SGEMV backends."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing SMT CSV before starting.",
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
        int(value)
        for value in config["vector_replication_factors"]
    }
    for factor in SELECTED_REPLICATION_FACTORS:
        if factor not in configured_replication_factors:
            raise ValueError(
                f"Selected replication factor {factor} is absent "
                "from vector_replication_factors"
            )

    smt_threads = [
        int(value)
        for value in config["smt_thread_counts"]
    ]
    if smt_threads != EXPECTED_SMT_THREADS:
        raise ValueError(
            "Expected smt_thread_counts to be exactly "
            f"{EXPECTED_SMT_THREADS}; received {smt_threads}"
        )

    database_path = resolve_database_path(config)
    openmp_config = config.get("openmp", {})

    if arguments.fresh and output_path.exists():
        output_path.unlink()

    completed = completed_configurations(output_path)

    total_configurations = (
        len(METRICS)
        * len(QUERY_INDICES)
        * len(SELECTED_REPLICATION_FACTORS)
        * len(K_VALUES)
        * len(BACKENDS)
        * len(smt_threads)
    )

    executed = 0
    skipped = 0

    print("Real-vector single-query SMT matrix")
    print("=" * 80)
    print(f"Metrics: {METRICS}")
    print(f"Queries: {QUERY_INDICES}")
    print(
        "Replication factors: "
        f"{SELECTED_REPLICATION_FACTORS}"
    )
    print(f"k values: {K_VALUES}")
    print(f"Backends: {BACKENDS}")
    print(f"SMT thread counts: {smt_threads}")
    print(f"Configurations: {total_configurations}")
    print(f"Output: {output_path}")
    print("=" * 80)

    for metric in METRICS:
        for query_index in QUERY_INDICES:
            for replication_factor in SELECTED_REPLICATION_FACTORS:
                for k in K_VALUES:
                    for backend in BACKENDS:
                        for smt_thread_count in smt_threads:
                            if backend == "openmp":
                                omp_threads = smt_thread_count
                                openblas_threads = 1
                            else:
                                omp_threads = 1
                                openblas_threads = smt_thread_count

                            key = (
                                metric,
                                backend,
                                query_index,
                                replication_factor,
                                k,
                                omp_threads,
                                openblas_threads,
                            )

                            if key in completed:
                                skipped += 1
                                continue

                            run_configuration(
                                database_path=database_path,
                                output_path=output_path,
                                metric=metric,
                                backend=backend,
                                query_index=query_index,
                                replication_factor=replication_factor,
                                k=k,
                                smt_threads=smt_thread_count,
                                openmp_config=openmp_config,
                                dry_run=arguments.dry_run,
                            )

                            executed += 1

    print("=" * 80)
    print(f"Executed configurations: {executed}")
    print(f"Skipped existing: {skipped}")

    if arguments.dry_run:
        print("Vector single-query SMT matrix: DRY RUN COMPLETE")
    else:
        print("Vector single-query SMT matrix: COMPLETE")


if __name__ == "__main__":
    main()
