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
DEFAULT_CONFIG = REPOSITORY_ROOT / "perf_eval/config/lexical_cpu_matrix.local.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "perf_eval/results/raw/lexical_real_single_raw.csv"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def load_candidate_ids(candidate_path: Path) -> list[str]:
    candidate_ids = []
    with candidate_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                candidate_ids.append(json.loads(line)["candidate_id"])
    if not candidate_ids:
        raise ValueError("Candidate file contains no candidates")
    return candidate_ids


def build_workload_scales(
    corpus_fractions: list[float],
    replication_factors: list[int],
) -> list[tuple[float, int]]:
    # Fractions below 1.0 are measured only once, without replication.
    fractional_scales = [
        (fraction, 1)
        for fraction in corpus_fractions
        if fraction < 1.0
    ]
    # Full corpus is measured at every configured replication factor.
    replicated_scales = [
        (1.0, replication_factor)
        for replication_factor in replication_factors
    ]
    scales = fractional_scales + replicated_scales
    if len(scales) != len(set(scales)):
        raise ValueError("Duplicate workload scales were configured")
    return scales


def completed_configurations(
    output_path: Path,
) -> set[tuple[str, float, int, str, int]]:
    if not output_path.exists():
        return set()

    completed = set()
    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "candidate_id",
            "corpus_fraction",
            "replication_factor",
            "backend",
            "thread_count",
        }
        if reader.fieldnames is None:
            return set()
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Existing CSV has the old schema. Run with --fresh. "
                f"Missing: {', '.join(sorted(missing))}"
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
    backend: str,
    thread_count: int,
    openmp_config: dict,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["OMP_NUM_THREADS"] = str(thread_count)
    environment["OMP_DYNAMIC"] = str(openmp_config.get("dynamic", "FALSE"))
    environment["OPENBLAS_NUM_THREADS"] = "1"

    if backend == "openmp":
        environment["OMP_PROC_BIND"] = str(openmp_config.get("proc_bind", "close"))
        environment["OMP_PLACES"] = str(openmp_config.get("places", "cores"))
        environment["OMP_WAIT_POLICY"] = str(openmp_config.get("wait_policy", "active"))
    else:
        environment.pop("OMP_PROC_BIND", None)
        environment.pop("OMP_PLACES", None)
        environment.pop("OMP_WAIT_POLICY", None)

    return environment


def run_configuration(
    *,
    config_path: Path,
    output_path: Path,
    candidate_id: str,
    corpus_fraction: float,
    replication_factor: int,
    backend: str,
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
        backend,
        "--candidate-id",
        candidate_id,
        "--corpus-fraction",
        str(corpus_fraction),
        "--replication-factor",
        str(replication_factor),
    ]

    print(
        f"{candidate_id} | fraction={corpus_fraction:.2f} | "
        f"rf={replication_factor} | {backend} | threads={thread_count}",
        flush=True,
    )

    if dry_run:
        return

    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=build_environment(
            backend=backend,
            thread_count=thread_count,
            openmp_config=openmp_config,
        ),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    config_path = arguments.config.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    config = load_config(config_path)

    candidate_ids = load_candidate_ids(
        resolve_path(config["candidates"]["input_path"])
    )
    corpus_fractions = [float(value) for value in config["corpus_fractions"]]
    replication_factors = [int(value) for value in config["replication_factors"]]
    thread_counts = [int(value) for value in config["physical_core_thread_counts"]]

    if any(not 0.0 < fraction <= 1.0 for fraction in corpus_fractions):
        raise ValueError("Every corpus fraction must be in (0, 1]")
    if any(factor < 1 for factor in replication_factors):
        raise ValueError("Every replication factor must be at least 1")

    workload_scales = build_workload_scales(
        corpus_fractions,
        replication_factors,
    )
    openmp_config = config.get("openmp", {})

    if arguments.fresh and output_path.exists():
        output_path.unlink()

    completed = completed_configurations(output_path)
    total = len(candidate_ids) * len(workload_scales) * (1 + len(thread_counts))
    executed = 0
    skipped = 0

    print("Real lexical physical-core matrix")
    print("=" * 72)
    print(f"Candidates: {len(candidate_ids)}")
    print(f"Workload scales: {workload_scales}")
    print(f"OpenMP threads: {thread_counts}")
    print(f"Configurations: {total}")
    print(f"Output: {output_path}")
    print("=" * 72)

    for candidate_id in candidate_ids:
        for corpus_fraction, replication_factor in workload_scales:
            variants = [("serial", 1)] + [
                ("openmp", thread_count)
                for thread_count in thread_counts
            ]

            for backend, thread_count in variants:
                key = (
                    candidate_id,
                    corpus_fraction,
                    replication_factor,
                    backend,
                    thread_count,
                )
                if key in completed:
                    skipped += 1
                    continue

                run_configuration(
                    config_path=config_path,
                    output_path=output_path,
                    candidate_id=candidate_id,
                    corpus_fraction=corpus_fraction,
                    replication_factor=replication_factor,
                    backend=backend,
                    thread_count=thread_count,
                    openmp_config=openmp_config,
                    dry_run=arguments.dry_run,
                )
                executed += 1

    print("=" * 72)
    print(f"Executed configurations: {executed}")
    print(f"Skipped existing: {skipped}")
    print("Physical-core matrix: COMPLETE")


if __name__ == "__main__":
    main()
