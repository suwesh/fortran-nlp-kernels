from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_RAW = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/lexical_real_single_raw.csv"
)
DEFAULT_SMT_RAW = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/lexical_real_single_smt_raw.csv"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "perf_eval/results"

CONFIG_KEYS = [
    "candidate_id",
    "candidate_source",
    "candidate_category",
    "candidate_shingles",
    "corpus_fraction",
    "replication_factor",
    "document_count",
    "corpus_shingles",
    "backend",
    "thread_count",
    "omp_dynamic",
    "omp_proc_bind",
    "omp_places",
    "omp_wait_policy",
    "warmup_runs",
    "measured_runs",
    "estimated_input_bytes",
    "estimated_output_bytes",
]

BASELINE_KEYS = [
    "candidate_id",
    "corpus_fraction",
    "replication_factor",
    "document_count",
    "corpus_shingles",
]

SELECTED_REPLICATION_FACTORS = [10, 100, 1000]
EXPECTED_CANDIDATES = 8
EXPECTED_SMT_THREADS = [8, 10, 12]
EXPECTED_SMT_CONFIGURATIONS = (
    EXPECTED_CANDIDATES
    * len(SELECTED_REPLICATION_FACTORS)
    * len(EXPECTED_SMT_THREADS)
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze lexical SMT sensitivity by combining the existing "
            "OpenMP-6 physical-core baseline with OpenMP-8/10/12 results."
        )
    )
    parser.add_argument(
        "--physical-raw",
        type=Path,
        default=DEFAULT_PHYSICAL_RAW,
    )
    parser.add_argument(
        "--smt-raw",
        type=Path,
        default=DEFAULT_SMT_RAW,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    return parser.parse_args()


def percentile_25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def percentile_75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def percentile_95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def validate_schema(data: pd.DataFrame, label: str) -> None:
    required = set(CONFIG_KEYS) | {
        "package_version",
        "run_index",
        "elapsed_ns",
        "documents_per_second",
        "corpus_shingles_per_second",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{label} CSV is missing columns: "
            + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError(f"{label} CSV contains no measurements")

    numeric_columns = [
        "candidate_shingles",
        "corpus_fraction",
        "replication_factor",
        "document_count",
        "corpus_shingles",
        "thread_count",
        "warmup_runs",
        "measured_runs",
        "run_index",
        "elapsed_ns",
        "documents_per_second",
        "corpus_shingles_per_second",
        "estimated_input_bytes",
        "estimated_output_bytes",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if (data["elapsed_ns"] <= 0).any():
        raise ValueError(f"{label} elapsed_ns must be positive")


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(CONFIG_KEYS, dropna=False, as_index=False)
        .agg(
            repetition_count=("elapsed_ns", "size"),
            latency_p25_ns=("elapsed_ns", percentile_25),
            latency_median_ns=("elapsed_ns", "median"),
            latency_mean_ns=("elapsed_ns", "mean"),
            latency_p75_ns=("elapsed_ns", percentile_75),
            latency_p95_ns=("elapsed_ns", percentile_95),
            latency_std_ns=("elapsed_ns", "std"),
            documents_per_second_median=(
                "documents_per_second",
                "median",
            ),
            corpus_shingles_per_second_median=(
                "corpus_shingles_per_second",
                "median",
            ),
        )
    )
    summary["latency_std_ns"] = summary["latency_std_ns"].fillna(0.0)
    summary["latency_cv"] = np.divide(
        summary["latency_std_ns"],
        summary["latency_mean_ns"],
        out=np.zeros(len(summary), dtype=np.float64),
        where=summary["latency_mean_ns"].to_numpy() != 0,
    )
    for suffix in ("p25", "median", "mean", "p75", "p95", "std"):
        summary[f"latency_{suffix}_ms"] = (
            summary[f"latency_{suffix}_ns"] / 1_000_000.0
        )
    return summary


def select_baselines(physical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = physical[
        (physical["corpus_fraction"] == 1.0)
        & physical["replication_factor"].isin(SELECTED_REPLICATION_FACTORS)
    ].copy()

    serial = selected[
        (selected["backend"] == "serial")
        & (selected["thread_count"] == 1)
    ].copy()

    openmp6 = selected[
        (selected["backend"] == "openmp")
        & (selected["thread_count"] == 6)
    ].copy()

    if serial.empty or openmp6.empty:
        raise ValueError(
            "Physical-core CSV does not contain all selected serial and "
            "OpenMP-6 baselines"
        )

    return serial, openmp6


def add_performance_metrics(
    combined: pd.DataFrame,
    serial: pd.DataFrame,
    openmp6: pd.DataFrame,
) -> pd.DataFrame:
    serial_lookup = serial[
        BASELINE_KEYS + ["latency_median_ns"]
    ].rename(
        columns={"latency_median_ns": "serial_latency_median_ns"}
    )
    six_lookup = openmp6[
        BASELINE_KEYS + ["latency_median_ns"]
    ].rename(
        columns={"latency_median_ns": "openmp6_latency_median_ns"}
    )

    result = combined.merge(
        serial_lookup,
        on=BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    ).merge(
        six_lookup,
        on=BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    )

    if result[[
        "serial_latency_median_ns",
        "openmp6_latency_median_ns",
    ]].isna().any().any():
        raise ValueError("At least one SMT row lacks a matching baseline")

    result["speedup_vs_serial"] = (
        result["serial_latency_median_ns"]
        / result["latency_median_ns"]
    )
    result["parallel_efficiency_percent"] = (
        result["speedup_vs_serial"]
        / result["thread_count"]
        * 100.0
    )
    result["gain_vs_openmp6"] = (
        result["openmp6_latency_median_ns"]
        / result["latency_median_ns"]
    )
    result["latency_change_vs_openmp6_percent"] = (
        (result["latency_median_ns"] / result["openmp6_latency_median_ns"] - 1.0)
        * 100.0
    )
    return result


def aggregate_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "replication_factor",
        "document_count",
        "corpus_shingles",
        "thread_count",
    ]
    return (
        summary.groupby(grouping, as_index=False)
        .agg(
            candidate_count=("candidate_id", "nunique"),
            latency_median_across_candidates_ms=(
                "latency_median_ms",
                "median",
            ),
            latency_p25_across_candidates_ms=(
                "latency_median_ms",
                percentile_25,
            ),
            latency_p75_across_candidates_ms=(
                "latency_median_ms",
                percentile_75,
            ),
            latency_p95_median_across_candidates_ms=(
                "latency_p95_ms",
                "median",
            ),
            latency_cv_median_across_candidates=(
                "latency_cv",
                "median",
            ),
            speedup_median_across_candidates=(
                "speedup_vs_serial",
                "median",
            ),
            efficiency_median_percent=(
                "parallel_efficiency_percent",
                "median",
            ),
            gain_vs_openmp6_median=(
                "gain_vs_openmp6",
                "median",
            ),
            gain_vs_openmp6_p25=(
                "gain_vs_openmp6",
                percentile_25,
            ),
            gain_vs_openmp6_p75=(
                "gain_vs_openmp6",
                percentile_75,
            ),
            corpus_shingles_per_second_median_across_candidates=(
                "corpus_shingles_per_second_median",
                "median",
            ),
        )
        .sort_values(["document_count", "thread_count"])
    )


def verify_smt_completeness(
    raw_smt: pd.DataFrame,
    smt_summary: pd.DataFrame,
) -> dict:
    duplicate_rows = int(
        raw_smt.duplicated(CONFIG_KEYS + ["run_index"]).sum()
    )
    repetition_mismatches = smt_summary[
        smt_summary["repetition_count"] != smt_summary["measured_runs"]
    ]
    thread_counts = sorted(
        int(value) for value in raw_smt["thread_count"].unique()
    )
    replication_factors = sorted(
        int(value) for value in raw_smt["replication_factor"].unique()
    )
    unique_configurations = len(smt_summary)

    return {
        "raw_measurement_rows": int(len(raw_smt)),
        "unique_smt_configurations": unique_configurations,
        "expected_smt_configurations": EXPECTED_SMT_CONFIGURATIONS,
        "is_complete": unique_configurations == EXPECTED_SMT_CONFIGURATIONS,
        "candidate_count": int(raw_smt["candidate_id"].nunique()),
        "replication_factors": replication_factors,
        "smt_thread_counts": thread_counts,
        "duplicate_run_rows": duplicate_rows,
        "repetition_mismatch_configurations": int(len(repetition_mismatches)),
        "package_versions": sorted(
            raw_smt["package_version"].astype(str).unique().tolist()
        ),
    }


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_latency(aggregated: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(9, 6))
    for threads, group in aggregated.groupby("thread_count"):
        group = group.sort_values("document_count")
        plt.plot(
            group["document_count"],
            group["latency_median_across_candidates_ms"],
            marker="o",
            label=f"OpenMP-{int(threads)}",
        )
        plt.fill_between(
            group["document_count"],
            group["latency_p25_across_candidates_ms"],
            group["latency_p75_across_candidates_ms"],
            alpha=0.15,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Document count")
    plt.ylabel("Median latency across candidates (ms)")
    plt.title("SMT sensitivity: exact Jaccard latency")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save_figure(output)


def plot_speedup(aggregated: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(9, 6))
    for documents, group in aggregated.groupby("document_count"):
        group = group.sort_values("thread_count")
        plt.plot(
            group["thread_count"],
            group["speedup_median_across_candidates"],
            marker="o",
            label=f"{int(documents):,} documents",
        )
    plt.xlabel("OpenMP thread count")
    plt.ylabel("Speedup versus dedicated serial backend")
    plt.title("SMT sensitivity: speedup versus serial")
    plt.xticks([6, 8, 10, 12])
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(output)


def plot_efficiency(aggregated: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(9, 6))
    for documents, group in aggregated.groupby("document_count"):
        group = group.sort_values("thread_count")
        plt.plot(
            group["thread_count"],
            group["efficiency_median_percent"],
            marker="o",
            label=f"{int(documents):,} documents",
        )
    plt.xlabel("OpenMP thread count")
    plt.ylabel("Parallel efficiency (%)")
    plt.title("SMT sensitivity: parallel efficiency")
    plt.xticks([6, 8, 10, 12])
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(output)


def plot_gain_vs_six(aggregated: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(9, 6))
    for documents, group in aggregated.groupby("document_count"):
        group = group.sort_values("thread_count")
        plt.plot(
            group["thread_count"],
            group["gain_vs_openmp6_median"],
            marker="o",
            label=f"{int(documents):,} documents",
        )
        plt.fill_between(
            group["thread_count"],
            group["gain_vs_openmp6_p25"],
            group["gain_vs_openmp6_p75"],
            alpha=0.15,
        )
    plt.axhline(1.0, linestyle="--", label="No gain over OpenMP-6")
    plt.xlabel("OpenMP thread count")
    plt.ylabel("Performance gain versus OpenMP-6")
    plt.title("Incremental SMT benefit over physical-core execution")
    plt.xticks([6, 8, 10, 12])
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(output)


def plot_variability(aggregated: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(9, 6))
    for documents, group in aggregated.groupby("document_count"):
        group = group.sort_values("thread_count")
        plt.plot(
            group["thread_count"],
            group["latency_cv_median_across_candidates"] * 100.0,
            marker="o",
            label=f"{int(documents):,} documents",
        )
    plt.xlabel("OpenMP thread count")
    plt.ylabel("Median latency coefficient of variation (%)")
    plt.title("SMT sensitivity: timing variability")
    plt.xticks([6, 8, 10, 12])
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(output)


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    headers = list(data.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.3f}")
            elif isinstance(value, (int, np.integer)):
                rendered.append(f"{int(value):,}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines) + "\n"


def create_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "document_count",
        "corpus_shingles",
        "thread_count",
        "latency_median_across_candidates_ms",
        "latency_p95_median_across_candidates_ms",
        "speedup_median_across_candidates",
        "efficiency_median_percent",
        "gain_vs_openmp6_median",
        "latency_cv_median_across_candidates",
        "corpus_shingles_per_second_median_across_candidates",
    ]
    table = aggregated[columns].rename(
        columns={
            "document_count": "Documents",
            "corpus_shingles": "Corpus shingles",
            "thread_count": "Threads",
            "latency_median_across_candidates_ms": "Median latency (ms)",
            "latency_p95_median_across_candidates_ms": "Median P95 (ms)",
            "speedup_median_across_candidates": "Speedup vs serial",
            "efficiency_median_percent": "Efficiency (%)",
            "gain_vs_openmp6_median": "Gain vs OpenMP-6",
            "latency_cv_median_across_candidates": "Median CV",
            "corpus_shingles_per_second_median_across_candidates": (
                "Corpus shingles/s"
            ),
        }
    )
    return table.sort_values(["Documents", "Threads"])


def main() -> None:
    arguments = parse_arguments()
    physical_path = arguments.physical_raw.expanduser().resolve()
    smt_path = arguments.smt_raw.expanduser().resolve()
    results_root = arguments.results_root.expanduser().resolve()

    processed_dir = results_root / "processed"
    tables_dir = results_root / "tables"
    figures_dir = results_root / "figures"
    for directory in (processed_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    physical_raw = pd.read_csv(physical_path, engine="python")
    smt_raw = pd.read_csv(smt_path, engine="python")
    validate_schema(physical_raw, "Physical-core")
    validate_schema(smt_raw, "SMT")

    physical_summary = summarize(physical_raw)
    smt_summary = summarize(smt_raw)
    serial, openmp6 = select_baselines(physical_summary)

    combined = pd.concat(
        [openmp6, smt_summary],
        ignore_index=True,
    )
    combined = add_performance_metrics(combined, serial, openmp6)
    combined = combined.sort_values(
        ["document_count", "candidate_id", "thread_count"]
    )
    aggregated = aggregate_candidates(combined)
    completeness = verify_smt_completeness(smt_raw, smt_summary)

    if not completeness["is_complete"]:
        raise RuntimeError(
            "SMT benchmark is incomplete: "
            f"{completeness['unique_smt_configurations']}/"
            f"{completeness['expected_smt_configurations']} configurations"
        )
    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate SMT run rows were detected")
    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError(
            "At least one SMT configuration has an incorrect repetition count"
        )

    combined.to_csv(
        processed_dir / "lexical_real_single_smt_configuration_summary.csv",
        index=False,
    )
    aggregated.to_csv(
        processed_dir / "lexical_real_single_smt_aggregated_summary.csv",
        index=False,
    )
    with (
        processed_dir / "lexical_real_single_smt_completeness_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    table = create_table(aggregated)
    table.to_csv(
        tables_dir / "lexical_real_single_smt_table.csv",
        index=False,
    )
    with (
        tables_dir / "lexical_real_single_smt_table.md"
    ).open("w", encoding="utf-8") as file:
        file.write("# Lexical SMT Sensitivity\n\n")
        file.write(
            "Values are medians across eight production-derived candidates. "
            "OpenMP-6 is reused from the physical-core experiment.\n\n"
        )
        file.write(dataframe_to_markdown(table))

    plot_latency(
        aggregated,
        figures_dir / "lexical_smt_latency.png",
    )
    plot_speedup(
        aggregated,
        figures_dir / "lexical_smt_speedup.png",
    )
    plot_efficiency(
        aggregated,
        figures_dir / "lexical_smt_efficiency.png",
    )
    plot_gain_vs_six(
        aggregated,
        figures_dir / "lexical_smt_gain_vs_openmp6.png",
    )
    plot_variability(
        aggregated,
        figures_dir / "lexical_smt_variability.png",
    )

    print("Lexical SMT analysis")
    print("=" * 72)
    print(
        "SMT configurations: "
        f"{completeness['unique_smt_configurations']}/"
        f"{completeness['expected_smt_configurations']}"
    )
    print(
        f"Raw SMT rows: {completeness['raw_measurement_rows']:,}"
    )
    print(f"Candidates: {completeness['candidate_count']}")
    print(
        f"Replication factors: {completeness['replication_factors']}"
    )
    print(f"SMT threads: {completeness['smt_thread_counts']}")
    print("Combined comparison threads: [6, 8, 10, 12]")
    print(f"Processed data: {processed_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print("=" * 72)
    print("Lexical SMT analysis: PASS")


if __name__ == "__main__":
    main()
