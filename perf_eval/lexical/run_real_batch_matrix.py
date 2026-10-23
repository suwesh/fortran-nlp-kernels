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
    / "perf_eval/results/raw/lexical_real_batch_raw.csv"
)

# Base corpus contains 372 documents. These factors produce:
#   1       -> 372 documents
#   10      -> 3,720 documents
#   100     -> 37,200 documents
#   1000    -> 372,000 documents
SELECTED_REPLICATION_FACTORS = [1, 10, 100, 1000]

METHODS = (
    "repeated_serial",
    "repeated_openmp",
    "native_batch_openmp",
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping"
        )

    return config


def completed_configurations(
    output_path: Path,
) -> set[tuple[str, int, int, int]]:
    """Return completed method/batch/replication/thread combinations."""

    if not output_path.exists():
        return set()

    required_columns = {
        "method",
        "batch_size",
        "replication_factor",
        "thread_count",
    }

    completed: set[tuple[str, int, int, int]] = set()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Existing batch CSV has no header"
            )

        missing = required_columns.difference(
            reader.fieldnames
        )

        if missing:
            raise ValueError(
                "Existing batch CSV uses an incompatible schema. "
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            completed.add(
                (
                    row["method"],
                    int(row["batch_size"]),
                    int(row["replication_factor"]),
                    int(row["thread_count"]),
                )
            )

    return completed


def build_environment(
    *,
    method: str,
    thread_count: int,
    openmp_config: dict,
) -> dict[str, str]:
    environment = os.environ.copy()

    # Force use of the installed immutable package rather than src/.
    environment.pop("PYTHONPATH", None)

    environment["OMP_NUM_THREADS"] = str(
        thread_count
    )
    environment["OMP_DYNAMIC"] = str(
        openmp_config.get("dynamic", "FALSE")
    )
    environment["OPENBLAS_NUM_THREADS"] = "1"

    if method in {
        "repeated_openmp",
        "native_batch_openmp",
    }:
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
    config_path: Path,
    output_path: Path,
    method: str,
    batch_size: int,
    replication_factor: int,
    thread_count: int,
    openmp_config: dict,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "perf_eval.lexical.benchmark_real_batch",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--method",
        method,
        "--batch-size",
        str(batch_size),
        "--replication-factor",
        str(replication_factor),
        "--skip-verification",
    ]

    environment = build_environment(
        method=method,
        thread_count=thread_count,
        openmp_config=openmp_config,
    )

    print(
        f"method={method} | "
        f"batch={batch_size} | "
        f"rf={replication_factor} | "
        f"threads={thread_count}"
    )

    if dry_run:
        return

    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def build_execution_variants(
    physical_threads: list[int],
) -> list[tuple[str, int]]:
    """
    Build non-redundant execution variants.

    repeated_serial is measured once with one thread.
    Both OpenMP methods are measured at 1, 2, 4, and 6 threads.
    """

    variants: list[tuple[str, int]] = [
        ("repeated_serial", 1),
    ]

    for method in (
        "repeated_openmp",
        "native_batch_openmp",
    ):
        variants.extend(
            (method, thread_count)
            for thread_count in physical_threads
        )

    return variants


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-corpus lexical physical-core batch matrix."
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
        help=(
            "Delete the existing formal batch CSV before starting."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running them.",
    )

    arguments = parser.parse_args()

    config_path = arguments.config.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    config = load_config(config_path)

    configured_replication_factors = {
        int(value)
        for value in config["replication_factors"]
    }

    for factor in SELECTED_REPLICATION_FACTORS:
        if factor not in configured_replication_factors:
            raise ValueError(
                "Selected batch replication factor "
                f"{factor} is absent from the configuration"
            )

    batch_sizes = [
        int(value)
        for value in config["candidate_batch_sizes"]
    ]

    expected_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if batch_sizes != expected_batch_sizes:
        raise ValueError(
            "Expected candidate_batch_sizes to be exactly "
            f"{expected_batch_sizes}; received {batch_sizes}"
        )

    physical_threads = [
        int(value)
        for value in config[
            "physical_core_thread_counts"
        ]
    ]

    if physical_threads != [1, 2, 4, 6]:
        raise ValueError(
            "Expected physical_core_thread_counts to be exactly "
            "[1, 2, 4, 6]"
        )

    execution_variants = build_execution_variants(
        physical_threads
    )
    openmp_config = config.get("openmp", {})

    if arguments.fresh and output_path.exists():
        output_path.unlink()

    completed = completed_configurations(output_path)

    total_configurations = (
        len(SELECTED_REPLICATION_FACTORS)
        * len(batch_sizes)
        * len(execution_variants)
    )

    executed = 0
    skipped = 0

    print("Real lexical physical-core batch matrix")
    print("=" * 72)
    print(
        "Replication factors: "
        f"{SELECTED_REPLICATION_FACTORS}"
    )
    print(f"Batch sizes: {batch_sizes}")
    print(
        "Execution variants: "
        f"{execution_variants}"
    )
    print(f"Configurations: {total_configurations}")
    print(f"Output: {output_path}")
    print("=" * 72)

    for replication_factor in (
        SELECTED_REPLICATION_FACTORS
    ):
        for batch_size in batch_sizes:
            for method, thread_count in (
                execution_variants
            ):
                key = (
                    method,
                    batch_size,
                    replication_factor,
                    thread_count,
                )

                if key in completed:
                    skipped += 1
                    continue

                run_configuration(
                    config_path=config_path,
                    output_path=output_path,
                    method=method,
                    batch_size=batch_size,
                    replication_factor=(
                        replication_factor
                    ),
                    thread_count=thread_count,
                    openmp_config=openmp_config,
                    dry_run=arguments.dry_run,
                )

                executed += 1

    print("=" * 72)
    print(f"Executed configurations: {executed}")
    print(f"Skipped existing: {skipped}")

    if arguments.dry_run:
        print("Physical-core batch matrix: DRY RUN COMPLETE")
    else:
        print("Physical-core batch matrix: COMPLETE")


if __name__ == "__main__":
    main()
