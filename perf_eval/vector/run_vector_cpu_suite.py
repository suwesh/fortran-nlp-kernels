from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPOSITORY_ROOT / "perf_eval/results/logs"
DEFAULT_STATUS_PATH = DEFAULT_LOG_DIR / "vector_cpu_suite_status.json"
DEFAULT_DATABASE_PATH = (
    REPOSITORY_ROOT / "perf_eval/data/packed/vector_real_database.npz"
)

BENCHMARK_STAGES = [
    {
        "name": "single_physical",
        "module": "perf_eval.vector.run_real_single_matrix",
        "analysis_module": "perf_eval.analysis.analyze_vector_real_single",
        "expected_configurations": 1_728,
        "raw_csv": REPOSITORY_ROOT
        / "perf_eval/results/raw/vector_real_single_raw.csv",
    },
    {
        "name": "single_smt",
        "module": "perf_eval.vector.run_real_single_smt",
        "analysis_module": "perf_eval.analysis.analyze_vector_real_single_smt",
        "expected_configurations": 432,
        "raw_csv": REPOSITORY_ROOT
        / "perf_eval/results/raw/vector_real_single_smt_raw.csv",
    },
    {
        "name": "batch_physical",
        "module": "perf_eval.vector.run_real_batch_matrix",
        "analysis_module": "perf_eval.analysis.analyze_vector_real_batch",
        "expected_configurations": 2_184,
        "raw_csv": REPOSITORY_ROOT
        / "perf_eval/results/raw/vector_real_batch_raw.csv",
    },
    {
        "name": "batch_smt",
        "module": "perf_eval.vector.run_real_batch_smt",
        "analysis_module": "perf_eval.analysis.analyze_vector_real_batch_smt",
        "expected_configurations": 486,
        "raw_csv": REPOSITORY_ROOT
        / "perf_eval/results/raw/vector_real_batch_smt_raw.csv",
    },
]

REQUIRED_MODULES = [
    "perf_eval.vector.benchmark_real_single",
    "perf_eval.vector.benchmark_real_batch",
    "perf_eval.vector.run_real_single_matrix",
    "perf_eval.vector.run_real_single_smt",
    "perf_eval.vector.run_real_batch_matrix",
    "perf_eval.vector.run_real_batch_smt",
    "perf_eval.analysis.analyze_vector_real_single",
    "perf_eval.analysis.analyze_vector_real_single_smt",
    "perf_eval.analysis.analyze_vector_real_batch",
    "perf_eval.analysis.analyze_vector_real_batch_smt",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and analyze the complete exact-vector CPU performance suite "
            "sequentially, with configurable cooldowns between matrices."
        )
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=900,
        help="Cooldown between completed benchmark/analysis pairs (default: 900).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS_PATH,
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run every matrix launcher with --dry-run, validate analysis "
            "module availability, skip analyzers, and skip cooldowns."
        ),
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Run benchmark matrices without generating analysis artifacts.",
    )
    return parser.parse_args()


def module_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def preflight(database_path: Path) -> None:
    missing_modules = [
        module for module in REQUIRED_MODULES if not module_exists(module)
    ]
    if missing_modules:
        raise RuntimeError(
            "Missing required Python modules: " + ", ".join(missing_modules)
        )

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Packed real-vector database does not exist: {database_path}"
        )

    if database_path.stat().st_size <= 0:
        raise ValueError(f"Packed database is empty: {database_path}")


def initial_status(
    *,
    dry_run: bool,
    cooldown_seconds: int,
    database_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": "vector_cpu",
        "state": "starting",
        "dry_run": dry_run,
        "python_executable": sys.executable,
        "repository_root": str(REPOSITORY_ROOT),
        "database_path": str(database_path),
        "cooldown_seconds": cooldown_seconds,
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "finished_at_utc": None,
        "current_stage": None,
        "total_expected_configurations": sum(
            stage["expected_configurations"] for stage in BENCHMARK_STAGES
        ),
        "stages": [],
    }


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at_utc"] = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(status, file, indent=2)
        file.write("\n")
    temporary.replace(path)


def print_and_log(message: str, log_file) -> None:
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def run_command(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
) -> tuple[int, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()

    with log_path.open("a", encoding="utf-8") as log_file:
        print_and_log("=" * 88, log_file)
        print_and_log(f"Started: {started}", log_file)
        print_and_log("Command: " + " ".join(command), log_file)
        print_and_log("=" * 88, log_file)

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip("\n")
            print_and_log(text, log_file)

        return_code = process.wait()
        finished = utc_now()
        print_and_log("=" * 88, log_file)
        print_and_log(f"Finished: {finished}", log_file)
        print_and_log(f"Return code: {return_code}", log_file)

    return return_code, started, finished


def stage_record(
    *,
    name: str,
    kind: str,
    module: str,
    expected_configurations: int | None,
    raw_csv: Path | None,
    log_path: Path,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "module": module,
        "state": "pending",
        "expected_configurations": expected_configurations,
        "raw_csv": str(raw_csv) if raw_csv is not None else None,
        "log_path": str(log_path),
        "started_at_utc": None,
        "finished_at_utc": None,
        "return_code": None,
        "raw_csv_exists_after_stage": None,
        "raw_csv_size_bytes_after_stage": None,
    }


def cooldown(
    *,
    seconds: int,
    status: dict[str, Any],
    status_path: Path,
    after_stage: str,
) -> None:
    if seconds <= 0:
        return

    status["state"] = "cooldown"
    status["current_stage"] = f"cooldown_after_{after_stage}"
    status["cooldown_started_at_utc"] = utc_now()
    status["cooldown_until_epoch"] = int(time.time()) + seconds
    write_status(status_path, status)

    print(
        f"\nCooldown after {after_stage}: {seconds} seconds "
        f"({seconds / 60:.1f} minutes)",
        flush=True,
    )

    remaining = seconds
    while remaining > 0:
        sleep_for = min(60, remaining)
        time.sleep(sleep_for)
        remaining -= sleep_for
        print(f"Cooldown remaining: {remaining} seconds", flush=True)

    status.pop("cooldown_started_at_utc", None)
    status.pop("cooldown_until_epoch", None)
    write_status(status_path, status)


def main() -> None:
    arguments = parse_arguments()

    if arguments.cooldown_seconds < 0:
        raise ValueError("cooldown-seconds must not be negative")

    log_dir = arguments.log_dir.expanduser().resolve()
    status_path = arguments.status.expanduser().resolve()
    database_path = arguments.database.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    preflight(database_path)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    status = initial_status(
        dry_run=arguments.dry_run,
        cooldown_seconds=arguments.cooldown_seconds,
        database_path=database_path,
    )
    status["session_id"] = session_id
    status["state"] = "running"
    write_status(status_path, status)

    print("Vector CPU performance suite", flush=True)
    print("=" * 88, flush=True)
    print(f"Session: {session_id}", flush=True)
    print(f"Repository: {REPOSITORY_ROOT}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Database: {database_path}", flush=True)
    print(f"Dry run: {arguments.dry_run}", flush=True)
    print(f"Skip analysis: {arguments.skip_analysis}", flush=True)
    print(f"Cooldown: {arguments.cooldown_seconds} seconds", flush=True)
    print("Expected configurations: 4,830", flush=True)
    print("=" * 88, flush=True)

    try:
        for stage_index, benchmark in enumerate(BENCHMARK_STAGES):
            benchmark_name = str(benchmark["name"])
            benchmark_module = str(benchmark["module"])
            analysis_module = str(benchmark["analysis_module"])
            expected = int(benchmark["expected_configurations"])
            raw_csv = Path(benchmark["raw_csv"])

            benchmark_log = (
                log_dir / f"{session_id}_{benchmark_name}_benchmark.log"
            )
            benchmark_record = stage_record(
                name=benchmark_name,
                kind="benchmark",
                module=benchmark_module,
                expected_configurations=expected,
                raw_csv=raw_csv,
                log_path=benchmark_log,
            )
            status["stages"].append(benchmark_record)
            status["current_stage"] = benchmark_name
            benchmark_record["state"] = "running"
            write_status(status_path, status)

            command = [sys.executable, "-m", benchmark_module]
            if arguments.dry_run:
                command.append("--dry-run")

            return_code, started, finished = run_command(
                command=command,
                cwd=REPOSITORY_ROOT,
                log_path=benchmark_log,
            )
            benchmark_record["started_at_utc"] = started
            benchmark_record["finished_at_utc"] = finished
            benchmark_record["return_code"] = return_code
            benchmark_record["state"] = (
                "complete" if return_code == 0 else "failed"
            )
            benchmark_record["raw_csv_exists_after_stage"] = raw_csv.exists()
            benchmark_record["raw_csv_size_bytes_after_stage"] = (
                raw_csv.stat().st_size if raw_csv.exists() else None
            )
            write_status(status_path, status)

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

            if not arguments.dry_run and not raw_csv.is_file():
                raise RuntimeError(
                    f"Benchmark completed but expected raw CSV is absent: {raw_csv}"
                )

            if arguments.dry_run:
                print(
                    f"Analysis module validated but not executed in dry run: "
                    f"{analysis_module}",
                    flush=True,
                )
            elif not arguments.skip_analysis:
                analysis_name = f"analyze_{benchmark_name}"
                analysis_log = (
                    log_dir / f"{session_id}_{benchmark_name}_analysis.log"
                )
                analysis_record = stage_record(
                    name=analysis_name,
                    kind="analysis",
                    module=analysis_module,
                    expected_configurations=expected,
                    raw_csv=raw_csv,
                    log_path=analysis_log,
                )
                status["stages"].append(analysis_record)
                status["current_stage"] = analysis_name
                analysis_record["state"] = "running"
                write_status(status_path, status)

                analysis_command = [sys.executable, "-m", analysis_module]
                return_code, started, finished = run_command(
                    command=analysis_command,
                    cwd=REPOSITORY_ROOT,
                    log_path=analysis_log,
                )
                analysis_record["started_at_utc"] = started
                analysis_record["finished_at_utc"] = finished
                analysis_record["return_code"] = return_code
                analysis_record["state"] = (
                    "complete" if return_code == 0 else "failed"
                )
                analysis_record["raw_csv_exists_after_stage"] = raw_csv.exists()
                analysis_record["raw_csv_size_bytes_after_stage"] = (
                    raw_csv.stat().st_size if raw_csv.exists() else None
                )
                write_status(status_path, status)

                if return_code != 0:
                    raise subprocess.CalledProcessError(
                        return_code,
                        analysis_command,
                    )

            is_last_stage = stage_index == len(BENCHMARK_STAGES) - 1
            if not arguments.dry_run and not is_last_stage:
                cooldown(
                    seconds=arguments.cooldown_seconds,
                    status=status,
                    status_path=status_path,
                    after_stage=benchmark_name,
                )

        status["state"] = "complete"
        status["current_stage"] = None
        status["finished_at_utc"] = utc_now()
        write_status(status_path, status)

        print("=" * 88, flush=True)
        print("Vector CPU performance suite: COMPLETE", flush=True)
        print(f"Status: {status_path}", flush=True)
        print(f"Stage logs: {log_dir}", flush=True)

    except BaseException as error:
        status["state"] = "failed"
        status["finished_at_utc"] = utc_now()
        status["error_type"] = type(error).__name__
        status["error_message"] = str(error)
        write_status(status_path, status)
        print("=" * 88, flush=True)
        print("Vector CPU performance suite: FAILED", flush=True)
        print(f"Error: {type(error).__name__}: {error}", flush=True)
        print(f"Status: {status_path}", flush=True)
        raise


if __name__ == "__main__":
    main()
