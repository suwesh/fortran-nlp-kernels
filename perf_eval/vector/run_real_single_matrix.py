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
    / "perf_eval/results/raw/vector_real_single_raw.csv"
)

METRICS = ["ip", "l2"]
QUERY_INDICES = [0, 67, 208, 474]
K_VALUES = [1, 10, 50]
EXPECTED_REPLICATION_FACTORS = [1, 2, 5, 10, 25, 50, 100, 250]
EXPECTED_PHYSICAL_THREADS = [1, 2, 4, 6]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    return config


def build_execution_variants(
    physical_threads: list[int],
) -> list[tuple[str, int, int]]:
    """Return (backend, OMP threads, OpenBLAS threads) variants."""

    variants: list[tuple[str, int, int]] = [
        ("serial", 1, 1),
    ]

    variants.extend(
        ("openmp", thread_count, 1)
        for thread_count in physical_threads
    )

    variants.extend(
        ("blas", 1, thread_count)
        for thread_count in physical_threads
    )

    return variants


def completed_configurations(
    output_path: Path,
) -> set[tuple[str, str, int, int, int, int, int]]:
    """Return configurations already represented in the raw CSV."""

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
            raise ValueError("Existing vector CSV has no header")

        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Existing vector CSV uses an incompatible schema. "
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
    omp_threads: int,
    openblas_threads: int,
    openmp_config: dict,
) -> dict[str, str]:
    environment = os.environ.copy()

    # Ensure subprocesses measure the installed PyPI wheel.
    environment.pop("PYTHONPATH", None)

    environment["OMP_NUM_THREADS"] = str(omp_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(openblas_threads)
    environment["OMP_DYNAMIC"] = str(
        openmp_config.get("dynamic", "FALSE")
    )

    if backend == "openmp":
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
        # Serial and BLAS variants do not use the handwritten OpenMP path.
        environment.pop("OMP_PROC_BIND", None)
        environment.pop("OMP_PLACES", None)
        environment.pop("OMP_WAIT_POLICY", None)

    return environment


def run_configuration(
    *,
    config_path: Path,
    output_path: Path,
    metric: str,
    backend: str,
    query_index: int,
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

    vector_config = load_config(config_path).get("vector_database", {})
    database_path = vector_config.get("output_path")
    if database_path is not None:
        path = Path(database_path).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        command.extend(["--database", str(path.resolve())])

    environment = build_environment(
        backend=backend,
        omp_threads=omp_threads,
        openblas_threads=openblas_threads,
        openmp_config=openmp_config,
    )

    print(
        f"metric={metric} | "
        f"query={query_index} | "
        f"rf={replication_factor} | "
        f"k={k} | "
        f"backend={backend} | "
        f"omp={omp_threads} | "
        f"blas={openblas_threads}"
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
        description=(
            "Run the real-vector single-query physical-core matrix."
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
        help="Delete the existing formal CSV before starting.",
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

    replication_factors = [
        int(value)
        for value in config["vector_replication_factors"]
    ]
    if replication_factors != EXPECTED_REPLICATION_FACTORS:
        raise ValueError(
            "Expected vector_replication_factors to be exactly "
            f"{EXPECTED_REPLICATION_FACTORS}; received "
            f"{replication_factors}"
        )

    physical_threads = [
        int(value)
        for value in config["physical_core_thread_counts"]
    ]
    if physical_threads != EXPECTED_PHYSICAL_THREADS:
        raise ValueError(
            "Expected physical_core_thread_counts to be exactly "
            f"{EXPECTED_PHYSICAL_THREADS}; received {physical_threads}"
        )

    execution_variants = build_execution_variants(
        physical_threads
    )
    openmp_config = config.get("openmp", {})

    if arguments.fresh and output_path.exists():
        output_path.unlink()

    completed = completed_configurations(output_path)

    total_configurations = (
        len(METRICS)
        * len(QUERY_INDICES)
        * len(replication_factors)
        * len(K_VALUES)
        * len(execution_variants)
    )

    executed = 0
    skipped = 0

    print("Real-vector single-query physical-core matrix")
    print("=" * 80)
    print(f"Metrics: {METRICS}")
    print(f"Queries: {QUERY_INDICES}")
    print(f"Replication factors: {replication_factors}")
    print(f"k values: {K_VALUES}")
    print(f"Execution variants: {execution_variants}")
    print(f"Configurations: {total_configurations}")
    print(f"Output: {output_path}")
    print("=" * 80)

    for metric in METRICS:
        for query_index in QUERY_INDICES:
            for replication_factor in replication_factors:
                for k in K_VALUES:
                    for (
                        backend,
                        omp_threads,
                        openblas_threads,
                    ) in execution_variants:
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
                            config_path=config_path,
                            output_path=output_path,
                            metric=metric,
                            backend=backend,
                            query_index=query_index,
                            replication_factor=replication_factor,
                            k=k,
                            omp_threads=omp_threads,
                            openblas_threads=openblas_threads,
                            openmp_config=openmp_config,
                            dry_run=arguments.dry_run,
                        )

                        executed += 1

    print("=" * 80)
    print(f"Executed configurations: {executed}")
    print(f"Skipped existing: {skipped}")

    if arguments.dry_run:
        print("Vector single-query matrix: DRY RUN COMPLETE")
    else:
        print("Vector single-query matrix: COMPLETE")


if __name__ == "__main__":
    main()
