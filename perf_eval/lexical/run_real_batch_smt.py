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
    / "perf_eval/results/raw/lexical_real_batch_smt_raw.csv"
)

SELECTED_REPLICATION_FACTORS = [10, 100, 1000]
SELECTED_BATCH_SIZES = [8, 32, 64]
SELECTED_METHODS = [
    "repeated_openmp",
    "native_batch_openmp",
]


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
                "Existing batch SMT CSV has no header"
            )

        missing = required_columns.difference(
            reader.fieldnames
        )

        if missing:
            raise ValueError(
                "Existing batch SMT CSV uses an incompatible schema. "
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
    thread_count: int,
    openmp_config: dict,
) -> dict[str, str]:
    environment = os.environ.copy()

    environment.pop("PYTHONPATH", None)

    environment["OMP_NUM_THREADS"] = str(
        thread_count
    )
    environment["OMP_DYNAMIC"] = str(
        openmp_config.get("dynamic", "FALSE")
    )
    environment["OMP_PROC_BIND"] = str(
        openmp_config.get("proc_bind", "close")
    )

    # SMT workers must occupy distinct logical hardware threads.
    environment["OMP_PLACES"] = "threads"

    environment["OMP_WAIT_POLICY"] = str(
        openmp_config.get("wait_policy", "active")
    )
    environment["OPENBLAS_NUM_THREADS"] = "1"

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-corpus lexical batched-search SMT matrix."
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
            "Delete the existing batch SMT CSV before starting."
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
    configured_batch_sizes = {
        int(value)
        for value in config["candidate_batch_sizes"]
    }

    for factor in SELECTED_REPLICATION_FACTORS:
        if factor not in configured_replication_factors:
            raise ValueError(
                "Selected SMT replication factor "
                f"{factor} is absent from the configuration"
            )

    for batch_size in SELECTED_BATCH_SIZES:
        if batch_size not in configured_batch_sizes:
            raise ValueError(
                "Selected SMT batch size "
                f"{batch_size} is absent from the configuration"
            )

    thread_counts = [
        int(value)
        for value in config["smt_thread_counts"]
    ]

    if thread_counts != [8, 10, 12]:
        raise ValueError(
            "Expected smt_thread_counts to be exactly [8, 10, 12]"
        )

    openmp_config = config.get("openmp", {})

    if arguments.fresh and output_path.exists():
        output_path.unlink()

    completed = completed_configurations(
        output_path
    )

    total_configurations = (
        len(SELECTED_REPLICATION_FACTORS)
        * len(SELECTED_BATCH_SIZES)
        * len(SELECTED_METHODS)
        * len(thread_counts)
    )

    executed = 0
    skipped = 0

    print("Real lexical batched-search SMT matrix")
    print("=" * 72)
    print(
        "Replication factors: "
        f"{SELECTED_REPLICATION_FACTORS}"
    )
    print(
        f"Batch sizes: {SELECTED_BATCH_SIZES}"
    )
    print(f"Methods: {SELECTED_METHODS}")
    print(f"SMT threads: {thread_counts}")
    print("OMP_PLACES: threads")
    print(f"Configurations: {total_configurations}")
    print(f"Output: {output_path}")
    print("=" * 72)

    for replication_factor in (
        SELECTED_REPLICATION_FACTORS
    ):
        for batch_size in SELECTED_BATCH_SIZES:
            for method in SELECTED_METHODS:
                for thread_count in thread_counts:
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
        print("Batched-search SMT matrix: DRY RUN COMPLETE")
    else:
        print("Batched-search SMT matrix: COMPLETE")


if __name__ == "__main__":
    main()
