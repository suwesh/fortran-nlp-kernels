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
    / "perf_eval/results/raw/lexical_real_batch_raw.csv"
)
DEFAULT_SMT_RAW = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/lexical_real_batch_smt_raw.csv"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "perf_eval/results"

SELECTED_REPLICATION_FACTORS = [10, 100, 1000]
SELECTED_BATCH_SIZES = [8, 32, 64]
SELECTED_METHODS = [
    "repeated_openmp",
    "native_batch_openmp",
]
SMT_THREADS = [8, 10, 12]
COMPARISON_THREADS = [6, 8, 10, 12]
EXPECTED_SMT_CONFIGURATIONS = 54

CONFIG_KEYS = [
    "method",
    "batch_size",
    "unique_base_candidate_count",
    "minimum_candidate_shingles",
    "median_candidate_shingles",
    "mean_candidate_shingles",
    "maximum_candidate_shingles",
    "replication_factor",
    "document_count",
    "corpus_shingles",
    "candidate_document_comparisons",
    "thread_count",
    "omp_dynamic",
    "omp_proc_bind",
    "omp_places",
    "omp_wait_policy",
    "warmup_runs",
    "measured_runs",
    "estimated_input_bytes",
    "estimated_peak_output_bytes",
    "logical_output_bytes",
]

WORKLOAD_KEYS = [
    "method",
    "batch_size",
    "replication_factor",
    "document_count",
    "corpus_shingles",
    "candidate_document_comparisons",
]

SERIAL_BASELINE_KEYS = [
    "batch_size",
    "replication_factor",
    "document_count",
    "corpus_shingles",
    "candidate_document_comparisons",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze batched lexical SMT sensitivity by combining the "
            "existing OpenMP-6 physical-core results with OpenMP-8/10/12."
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
        "candidates_per_second",
        "comparisons_per_second",
        "corpus_shingle_visits_per_second",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{label} CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError(f"{label} CSV contains no measurements")

    numeric_columns = [
        "batch_size",
        "replication_factor",
        "document_count",
        "corpus_shingles",
        "candidate_document_comparisons",
        "thread_count",
        "warmup_runs",
        "measured_runs",
        "run_index",
        "elapsed_ns",
        "candidates_per_second",
        "comparisons_per_second",
        "corpus_shingle_visits_per_second",
        "estimated_input_bytes",
        "estimated_peak_output_bytes",
        "logical_output_bytes",
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
            candidates_per_second_median=(
                "candidates_per_second",
                "median",
            ),
            comparisons_per_second_median=(
                "comparisons_per_second",
                "median",
            ),
            corpus_shingle_visits_per_second_median=(
                "corpus_shingle_visits_per_second",
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

    summary["estimated_input_mib"] = (
        summary["estimated_input_bytes"] / (1024.0**2)
    )
    summary["estimated_peak_output_mib"] = (
        summary["estimated_peak_output_bytes"] / (1024.0**2)
    )
    summary["estimated_peak_total_mib"] = (
        summary["estimated_input_mib"]
        + summary["estimated_peak_output_mib"]
    )

    return summary


def select_physical_baselines(
    physical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = physical[
        physical["replication_factor"].isin(SELECTED_REPLICATION_FACTORS)
        & physical["batch_size"].isin(SELECTED_BATCH_SIZES)
    ].copy()

    serial = selected[
        (selected["method"] == "repeated_serial")
        & (selected["thread_count"] == 1)
    ].copy()

    openmp6 = selected[
        selected["method"].isin(SELECTED_METHODS)
        & (selected["thread_count"] == 6)
    ].copy()

    expected_serial = (
        len(SELECTED_REPLICATION_FACTORS)
        * len(SELECTED_BATCH_SIZES)
    )
    expected_six = expected_serial * len(SELECTED_METHODS)

    if len(serial) != expected_serial:
        raise ValueError(
            "Physical CSV lacks selected repeated-serial baselines"
        )
    if len(openmp6) != expected_six:
        raise ValueError(
            "Physical CSV lacks selected OpenMP-6 method baselines"
        )

    return serial, openmp6


def add_metrics(
    combined: pd.DataFrame,
    serial: pd.DataFrame,
    openmp6: pd.DataFrame,
) -> pd.DataFrame:
    serial_lookup = serial[
        SERIAL_BASELINE_KEYS + ["latency_median_ns"]
    ].rename(
        columns={"latency_median_ns": "repeated_serial_latency_ns"}
    )

    six_lookup = openmp6[
        WORKLOAD_KEYS + ["latency_median_ns"]
    ].rename(
        columns={"latency_median_ns": "openmp6_latency_ns"}
    )

    result = combined.merge(
        serial_lookup,
        on=SERIAL_BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    ).merge(
        six_lookup,
        on=WORKLOAD_KEYS,
        how="left",
        validate="many_to_one",
    )

    if result[[
        "repeated_serial_latency_ns",
        "openmp6_latency_ns",
    ]].isna().any().any():
        raise ValueError("At least one SMT row lacks a matching baseline")

    result["speedup_vs_repeated_serial"] = (
        result["repeated_serial_latency_ns"]
        / result["latency_median_ns"]
    )
    result["parallel_efficiency_percent"] = (
        result["speedup_vs_repeated_serial"]
        / result["thread_count"]
        * 100.0
    )
    result["gain_vs_openmp6"] = (
        result["openmp6_latency_ns"]
        / result["latency_median_ns"]
    )
    result["latency_change_vs_openmp6_percent"] = (
        (result["latency_median_ns"] / result["openmp6_latency_ns"] - 1.0)
        * 100.0
    )

    return result


def verify_completeness(
    smt_raw: pd.DataFrame,
    smt_summary: pd.DataFrame,
) -> dict:
    duplicate_rows = int(
        smt_raw.duplicated(CONFIG_KEYS + ["run_index"]).sum()
    )
    mismatches = smt_summary[
        smt_summary["repetition_count"] != smt_summary["measured_runs"]
    ]

    actual_variants = set(
        smt_raw[["method", "thread_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_variants = {
        (method, thread)
        for method in SELECTED_METHODS
        for thread in SMT_THREADS
    }

    unique_configurations = len(smt_summary)

    return {
        "raw_measurement_rows": int(len(smt_raw)),
        "unique_smt_configurations": unique_configurations,
        "expected_smt_configurations": EXPECTED_SMT_CONFIGURATIONS,
        "is_complete": unique_configurations == EXPECTED_SMT_CONFIGURATIONS,
        "replication_factors": sorted(
            int(value) for value in smt_raw["replication_factor"].unique()
        ),
        "batch_sizes": sorted(
            int(value) for value in smt_raw["batch_size"].unique()
        ),
        "methods": sorted(smt_raw["method"].unique().tolist()),
        "smt_thread_counts": sorted(
            int(value) for value in smt_raw["thread_count"].unique()
        ),
        "execution_variants_match": actual_variants == expected_variants,
        "duplicate_run_rows": duplicate_rows,
        "repetition_mismatch_configurations": int(len(mismatches)),
        "package_versions": sorted(
            smt_raw["package_version"].astype(str).unique().tolist()
        ),
    }


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def method_label(method: str) -> str:
    if method == "repeated_openmp":
        return "Repeated OpenMP"
    return "Native batch OpenMP"


def plot_latency(combined: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)

    groups = combined.groupby(["document_count", "batch_size"])
    for axis, ((documents, batch_size), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("thread_count")
            axis.plot(
                method_group["thread_count"],
                method_group["latency_median_ms"],
                marker="o",
                label=method_label(method),
            )
        axis.set_title(
            f"{int(documents):,} docs, batch {int(batch_size)}"
        )
        axis.set_xlabel("OpenMP thread count")
        axis.set_ylabel("Median batch latency (ms)")
        axis.set_xticks(COMPARISON_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Batched-search SMT sensitivity: latency")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_gain_vs_six(combined: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)

    groups = combined.groupby(["document_count", "batch_size"])
    for axis, ((documents, batch_size), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("thread_count")
            axis.plot(
                method_group["thread_count"],
                method_group["gain_vs_openmp6"],
                marker="o",
                label=method_label(method),
            )
        axis.axhline(1.0, linestyle="--", label="No gain over 6")
        axis.set_title(
            f"{int(documents):,} docs, batch {int(batch_size)}"
        )
        axis.set_xlabel("OpenMP thread count")
        axis.set_ylabel("Gain versus OpenMP-6")
        axis.set_xticks(COMPARISON_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Incremental SMT benefit for batched Jaccard search")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_speedup(combined: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)

    groups = combined.groupby(["document_count", "batch_size"])
    for axis, ((documents, batch_size), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("thread_count")
            axis.plot(
                method_group["thread_count"],
                method_group["speedup_vs_repeated_serial"],
                marker="o",
                label=method_label(method),
            )
        axis.set_title(
            f"{int(documents):,} docs, batch {int(batch_size)}"
        )
        axis.set_xlabel("OpenMP thread count")
        axis.set_ylabel("Speedup vs repeated serial")
        axis.set_xticks(COMPARISON_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Batched-search SMT speedup versus serial")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_efficiency(combined: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)

    groups = combined.groupby(["document_count", "batch_size"])
    for axis, ((documents, batch_size), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("thread_count")
            axis.plot(
                method_group["thread_count"],
                method_group["parallel_efficiency_percent"],
                marker="o",
                label=method_label(method),
            )
        axis.set_title(
            f"{int(documents):,} docs, batch {int(batch_size)}"
        )
        axis.set_xlabel("OpenMP thread count")
        axis.set_ylabel("Parallel efficiency (%)")
        axis.set_xticks(COMPARISON_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Batched-search SMT parallel efficiency")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_variability(combined: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)

    groups = combined.groupby(["document_count", "batch_size"])
    for axis, ((documents, batch_size), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("thread_count")
            axis.plot(
                method_group["thread_count"],
                method_group["latency_cv"] * 100.0,
                marker="o",
                label=method_label(method),
            )
        axis.set_title(
            f"{int(documents):,} docs, batch {int(batch_size)}"
        )
        axis.set_xlabel("OpenMP thread count")
        axis.set_ylabel("Latency CV (%)")
        axis.set_xticks(COMPARISON_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Batched-search SMT timing variability")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_table(combined: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "batch_size",
        "replication_factor",
        "document_count",
        "corpus_shingles",
        "candidate_document_comparisons",
        "thread_count",
        "repetition_count",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_cv",
        "comparisons_per_second_median",
        "speedup_vs_repeated_serial",
        "parallel_efficiency_percent",
        "gain_vs_openmp6",
        "latency_change_vs_openmp6_percent",
        "estimated_peak_total_mib",
    ]
    return combined[columns].sort_values(
        ["document_count", "batch_size", "method", "thread_count"]
    )


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
    validate_schema(physical_raw, "Physical batch")
    validate_schema(smt_raw, "Batch SMT")

    physical_summary = summarize(physical_raw)
    smt_summary = summarize(smt_raw)
    serial, openmp6 = select_physical_baselines(physical_summary)

    combined = pd.concat([openmp6, smt_summary], ignore_index=True)
    combined = add_metrics(combined, serial, openmp6)
    combined = combined.sort_values(
        ["document_count", "batch_size", "method", "thread_count"]
    )

    completeness = verify_completeness(smt_raw, smt_summary)
    if not completeness["is_complete"]:
        raise RuntimeError(
            "Batch SMT benchmark is incomplete: "
            f"{completeness['unique_smt_configurations']}/"
            f"{completeness['expected_smt_configurations']} configurations"
        )
    if not completeness["execution_variants_match"]:
        raise RuntimeError("Batch SMT execution variants do not match design")
    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate batch SMT run rows were detected")
    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError(
            "At least one batch SMT configuration has incorrect repetitions"
        )

    combined.to_csv(
        processed_dir / "lexical_real_batch_smt_configuration_summary.csv",
        index=False,
    )
    with (
        processed_dir / "lexical_real_batch_smt_completeness_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    table = create_table(combined)
    table.to_csv(
        tables_dir / "lexical_real_batch_smt_table.csv",
        index=False,
    )
    with (
        tables_dir / "lexical_real_batch_smt_table.md"
    ).open("w", encoding="utf-8") as file:
        file.write("# Lexical Batched-Search SMT Sensitivity\n\n")
        file.write(
            "OpenMP-6 is reused from the physical-core batch experiment; "
            "OpenMP-8/10/12 come from the dedicated SMT matrix.\n\n"
        )
        file.write(dataframe_to_markdown(table))

    plot_latency(
        combined,
        figures_dir / "lexical_batch_smt_latency.png",
    )
    plot_gain_vs_six(
        combined,
        figures_dir / "lexical_batch_smt_gain_vs_openmp6.png",
    )
    plot_speedup(
        combined,
        figures_dir / "lexical_batch_smt_speedup.png",
    )
    plot_efficiency(
        combined,
        figures_dir / "lexical_batch_smt_efficiency.png",
    )
    plot_variability(
        combined,
        figures_dir / "lexical_batch_smt_variability.png",
    )

    print("Lexical batched-search SMT analysis")
    print("=" * 72)
    print(
        "SMT configurations: "
        f"{completeness['unique_smt_configurations']}/"
        f"{completeness['expected_smt_configurations']}"
    )
    print(f"Raw SMT rows: {completeness['raw_measurement_rows']:,}")
    print(f"Replication factors: {completeness['replication_factors']}")
    print(f"Batch sizes: {completeness['batch_sizes']}")
    print(f"Methods: {completeness['methods']}")
    print(f"SMT threads: {completeness['smt_thread_counts']}")
    print("Combined comparison threads: [6, 8, 10, 12]")
    print(f"Processed data: {processed_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print("=" * 72)
    print("Lexical batched-search SMT analysis: PASS")


if __name__ == "__main__":
    main()
