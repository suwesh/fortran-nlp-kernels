from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "perf_eval"
    / "config"
    / "lexical_cpu_matrix.local.yaml"
)

DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "perf_eval"
    / "results"
    / "raw"
    / "lexical_real_single_smt_raw.csv"
)

# These correspond to:
#
# 3,720 documents
# 37,200 documents
# 372,000 documents
SELECTED_REPLICATION_FACTORS = [
    10,
    100,
    1000,
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()

    if not path.is_absolute():
        path = REPOSITORY_ROOT / path

    return path.resolve()


def load_config(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping"
        )

    return config


def load_candidate_ids(
    candidate_path: Path,
) -> list[str]:
    candidate_ids: list[str] = []

    with candidate_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            candidate = json.loads(line)

            candidate_ids.append(
                candidate["candidate_id"]
            )

    if not candidate_ids:
        raise ValueError(
            "Candidate file contains no candidates"
        )

    return candidate_ids


def completed_configurations(
    output_path: Path,
) -> set[tuple[str, float, int, str, int]]:
    """
    Return configurations already represented in the SMT CSV.

    Key:
        candidate_id
        corpus_fraction
        replication_factor
        backend
        thread_count
    """

    if not output_path.exists():
        return set()

    completed: set[
        tuple[str, float, int, str, int]
    ] = set()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "candidate_id",
            "corpus_fraction",
            "replication_factor",
            "backend",
            "thread_count",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Existing SMT CSV has no header"
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Existing SMT CSV uses an incompatible "
                "schema. Missing columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row in reader:
            completed.add(
                (
                    row["candidate_id"],
                    float(row["corpus_fraction"]),
                    int(row["replication_factor"]),
                    row["backend"],
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

    # Ensure that the immutable PyPI installation is used.
    environment.pop(
        "PYTHONPATH",
        None,
    )

    environment["OMP_NUM_THREADS"] = str(
        thread_count
    )

    environment["OMP_DYNAMIC"] = str(
        openmp_config.get(
            "dynamic",
            "FALSE",
        )
    )

    environment["OMP_PROC_BIND"] = str(
        openmp_config.get(
            "proc_bind",
            "close",
        )
    )

    # Use logical hardware threads as distinct places for the
    # SMT experiment.
    environment["OMP_PLACES"] = "threads"

    environment["OMP_WAIT_POLICY"] = str(
        openmp_config.get(
            "wait_policy",
            "active",
        )
    )

    environment["OPENBLAS_NUM_THREADS"] = "1"

    return environment


def run_configuration(
    *,
    config_path: Path,
    output_path: Path,
    candidate_id: str,
    replication_factor: int,
    thread_count: int,
    openmp_config: dict,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "perf_eval.lexical.benchmark_real_single",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--backend",
        "openmp",
        "--candidate-id",
        candidate_id,
        "--corpus-fraction",
        "1.0",
        "--replication-factor",
        str(replication_factor),
    ]

    environment = build_environment(
        thread_count=thread_count,
        openmp_config=openmp_config,
    )

    print(
        f"{candidate_id} | "
        "fraction=1.00 | "
        f"rf={replication_factor} | "
        "openmp | "
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
            "Run the real-corpus lexical SMT sensitivity "
            "matrix using OpenMP-8 and OpenMP-12."
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
            "Delete the existing SMT CSV before starting."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print configurations without running them."
        ),
    )

    arguments = parser.parse_args()

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    output_path = (
        arguments.output
        .expanduser()
        .resolve()
    )

    config = load_config(config_path)

    candidate_path = resolve_path(
        config["candidates"]["input_path"]
    )

    candidate_ids = load_candidate_ids(
        candidate_path
    )

    configured_replication_factors = {
        int(value)
        for value in config[
            "replication_factors"
        ]
    }

    for replication_factor in (
        SELECTED_REPLICATION_FACTORS
    ):
        if (
            replication_factor
            not in configured_replication_factors
        ):
            raise ValueError(
                "Selected SMT replication factor "
                f"{replication_factor} is not present "
                "in the configuration"
            )

    thread_counts = [
            int(value) for value in config[
            "smt_thread_counts"
        ]
    ]

    if thread_counts != [8, 10, 12]:
        raise ValueError(
            "Expected smt_thread_counts to contain "
            "exactly [8, 12]"
        )

    openmp_config = config.get(
        "openmp",
        {},
    )

    if arguments.fresh and output_path.exists():
        output_path.unlink()

    completed = completed_configurations(
        output_path
    )

    total_configurations = (
        len(candidate_ids)
        * len(SELECTED_REPLICATION_FACTORS)
        * len(thread_counts)
    )

    executed = 0
    skipped = 0

    print("Real lexical SMT sensitivity matrix")
    print("=" * 72)

    print(
        f"Candidates: {len(candidate_ids)}"
    )

    print(
        "Replication factors: "
        f"{SELECTED_REPLICATION_FACTORS}"
    )

    print(
        f"SMT threads: {thread_counts}"
    )

    print(
        "OMP_PLACES: threads"
    )

    print(
        f"Configurations: {total_configurations}"
    )

    print(
        f"Output: {output_path}"
    )

    print("=" * 72)

    for candidate_id in candidate_ids:
        for replication_factor in (
            SELECTED_REPLICATION_FACTORS
        ):
            for thread_count in thread_counts:
                configuration_key = (
                    candidate_id,
                    1.0,
                    replication_factor,
                    "openmp",
                    thread_count,
                )

                if configuration_key in completed:
                    skipped += 1
                    continue

                run_configuration(
                    config_path=config_path,
                    output_path=output_path,
                    candidate_id=candidate_id,
                    replication_factor=(
                        replication_factor
                    ),
                    thread_count=thread_count,
                    openmp_config=openmp_config,
                    dry_run=arguments.dry_run,
                )

                executed += 1

    print("=" * 72)

    print(
        f"Executed configurations: {executed}"
    )

    print(
        f"Skipped existing: {skipped}"
    )

    print(
        "SMT sensitivity matrix: COMPLETE"
    )


if __name__ == "__main__":
    main()