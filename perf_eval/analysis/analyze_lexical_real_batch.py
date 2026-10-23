from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/lexical_real_batch_raw.csv"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "perf_eval/results"

EXPECTED_REPLICATION_FACTORS = [1, 10, 100, 1000]
EXPECTED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
EXPECTED_OPENMP_THREADS = [1, 2, 4, 6]
EXPECTED_CONFIGURATIONS = 252

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
    "batch_size",
    "replication_factor",
    "document_count",
    "corpus_shingles",
    "candidate_document_comparisons",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze real-corpus lexical batch benchmarks and generate "
            "processed summaries, tables, and figures."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow exploratory outputs from an incomplete matrix.",
    )
    return parser.parse_args()


def percentile_25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def percentile_75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def percentile_95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def validate_raw_schema(data: pd.DataFrame) -> None:
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
            "Raw batch CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError("Raw batch CSV contains no measurements")

    numeric_columns = [
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
        raise ValueError("elapsed_ns must be positive")


def summarize_configurations(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(CONFIG_KEYS, dropna=False, as_index=False)
        .agg(
            repetition_count=("elapsed_ns", "size"),
            latency_min_ns=("elapsed_ns", "min"),
            latency_p25_ns=("elapsed_ns", percentile_25),
            latency_median_ns=("elapsed_ns", "median"),
            latency_mean_ns=("elapsed_ns", "mean"),
            latency_p75_ns=("elapsed_ns", percentile_75),
            latency_p95_ns=("elapsed_ns", percentile_95),
            latency_max_ns=("elapsed_ns", "max"),
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

    for suffix in (
        "min",
        "p25",
        "median",
        "mean",
        "p75",
        "p95",
        "max",
        "std",
    ):
        summary[f"latency_{suffix}_ms"] = (
            summary[f"latency_{suffix}_ns"] / 1_000_000.0
        )

    summary["estimated_input_mib"] = (
        summary["estimated_input_bytes"] / (1024.0**2)
    )
    summary["estimated_peak_output_mib"] = (
        summary["estimated_peak_output_bytes"] / (1024.0**2)
    )
    summary["logical_output_mib"] = (
        summary["logical_output_bytes"] / (1024.0**2)
    )
    summary["estimated_peak_total_mib"] = (
        summary["estimated_input_mib"]
        + summary["estimated_peak_output_mib"]
    )

    return add_comparative_metrics(summary)


def add_comparative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    serial = summary[
        (summary["method"] == "repeated_serial")
        & (summary["thread_count"] == 1)
    ][WORKLOAD_KEYS + ["latency_median_ns"]].rename(
        columns={"latency_median_ns": "repeated_serial_latency_ns"}
    )

    repeated_openmp = summary[
        summary["method"] == "repeated_openmp"
    ][WORKLOAD_KEYS + ["thread_count", "latency_median_ns"]].rename(
        columns={"latency_median_ns": "repeated_openmp_latency_ns"}
    )

    result = summary.merge(
        serial,
        on=WORKLOAD_KEYS,
        how="left",
        validate="many_to_one",
    )

    if result["repeated_serial_latency_ns"].isna().any():
        raise ValueError("At least one configuration lacks repeated-serial baseline")

    result["speedup_vs_repeated_serial"] = (
        result["repeated_serial_latency_ns"]
        / result["latency_median_ns"]
    )
    result["parallel_efficiency_percent"] = np.where(
        result["method"].isin(["repeated_openmp", "native_batch_openmp"]),
        result["speedup_vs_repeated_serial"] / result["thread_count"] * 100.0,
        100.0,
    )

    result = result.merge(
        repeated_openmp,
        on=WORKLOAD_KEYS + ["thread_count"],
        how="left",
        validate="many_to_one",
    )

    result["native_batch_gain_vs_repeated_openmp"] = np.where(
        result["method"] == "native_batch_openmp",
        result["repeated_openmp_latency_ns"] / result["latency_median_ns"],
        np.nan,
    )
    result["native_batch_latency_change_percent"] = np.where(
        result["method"] == "native_batch_openmp",
        (result["latency_median_ns"] / result["repeated_openmp_latency_ns"] - 1.0)
        * 100.0,
        np.nan,
    )

    return result


def verify_completeness(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict:
    duplicate_rows = int(
        raw.duplicated(CONFIG_KEYS + ["run_index"]).sum()
    )
    repetition_mismatches = summary[
        summary["repetition_count"] != summary["measured_runs"]
    ]

    methods = sorted(raw["method"].astype(str).unique().tolist())
    batch_sizes = sorted(int(value) for value in raw["batch_size"].unique())
    replication_factors = sorted(
        int(value) for value in raw["replication_factor"].unique()
    )
    thread_counts = sorted(int(value) for value in raw["thread_count"].unique())
    unique_configurations = len(summary)

    expected_variants = {
        ("repeated_serial", 1),
        *(('repeated_openmp', thread) for thread in EXPECTED_OPENMP_THREADS),
        *(('native_batch_openmp', thread) for thread in EXPECTED_OPENMP_THREADS),
    }
    actual_variants = set(
        raw[["method", "thread_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    return {
        "raw_measurement_rows": int(len(raw)),
        "unique_configurations": unique_configurations,
        "expected_configurations": EXPECTED_CONFIGURATIONS,
        "is_complete": unique_configurations == EXPECTED_CONFIGURATIONS,
        "methods": methods,
        "batch_sizes": batch_sizes,
        "replication_factors": replication_factors,
        "thread_counts": thread_counts,
        "execution_variant_count": len(actual_variants),
        "execution_variants_match": actual_variants == expected_variants,
        "duplicate_run_rows": duplicate_rows,
        "repetition_mismatch_configurations": int(len(repetition_mismatches)),
        "package_versions": sorted(
            raw["package_version"].astype(str).unique().tolist()
        ),
    }


def method_label(method: str, thread_count: int) -> str:
    if method == "repeated_serial":
        return "Repeated serial"
    if method == "repeated_openmp":
        return f"Repeated OpenMP-{int(thread_count)}"
    return f"Native batch OpenMP-{int(thread_count)}"


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def selected_method_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        ((summary["method"] == "repeated_serial") & (summary["thread_count"] == 1))
        | ((summary["method"] == "repeated_openmp") & (summary["thread_count"] == 6))
        | ((summary["method"] == "native_batch_openmp") & (summary["thread_count"] == 6))
    ].copy()


def plot_latency_vs_batch(summary: pd.DataFrame, output: Path) -> None:
    selected = selected_method_rows(summary)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    for axis, (documents, group) in zip(
        axes.ravel(),
        selected.groupby("document_count"),
        strict=False,
    ):
        for (method, threads), method_group in group.groupby(
            ["method", "thread_count"]
        ):
            method_group = method_group.sort_values("batch_size")
            axis.plot(
                method_group["batch_size"],
                method_group["latency_median_ms"],
                marker="o",
                label=method_label(method, threads),
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"{int(documents):,} documents")
        axis.set_xlabel("Candidate batch size")
        axis.set_ylabel("Median batch latency (ms)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.suptitle("Batched Jaccard latency versus candidate batch size")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_comparison_throughput(summary: pd.DataFrame, output: Path) -> None:
    selected = selected_method_rows(summary)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    for axis, (documents, group) in zip(
        axes.ravel(),
        selected.groupby("document_count"),
        strict=False,
    ):
        for (method, threads), method_group in group.groupby(
            ["method", "thread_count"]
        ):
            method_group = method_group.sort_values("batch_size")
            axis.plot(
                method_group["batch_size"],
                method_group["comparisons_per_second_median"],
                marker="o",
                label=method_label(method, threads),
            )
        axis.set_xscale("log", base=2)
        axis.set_title(f"{int(documents):,} documents")
        axis.set_xlabel("Candidate batch size")
        axis.set_ylabel("Candidate-document comparisons/s")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.suptitle("Batched Jaccard throughput")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_native_gain(summary: pd.DataFrame, output: Path) -> None:
    native = summary[
        (summary["method"] == "native_batch_openmp")
        & summary["thread_count"].isin(EXPECTED_OPENMP_THREADS)
    ].copy()

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for axis, (documents, group) in zip(
        axes.ravel(),
        native.groupby("document_count"),
        strict=False,
    ):
        for threads, thread_group in group.groupby("thread_count"):
            thread_group = thread_group.sort_values("batch_size")
            axis.plot(
                thread_group["batch_size"],
                thread_group["native_batch_gain_vs_repeated_openmp"],
                marker="o",
                label=f"{int(threads)} threads",
            )
        axis.axhline(1.0, linestyle="--", label="No gain")
        axis.set_xscale("log", base=2)
        axis.set_title(f"{int(documents):,} documents")
        axis.set_xlabel("Candidate batch size")
        axis.set_ylabel("Native batch gain vs repeated OpenMP")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.suptitle("Native collapsed-loop benefit at matching thread counts")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_speedup_vs_threads(summary: pd.DataFrame, output: Path) -> None:
    selected_batches = [1, 8, 64]
    selected = summary[
        summary["batch_size"].isin(selected_batches)
        & summary["method"].isin(["repeated_openmp", "native_batch_openmp"])
    ].copy()

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for axis, (documents, group) in zip(
        axes.ravel(),
        selected.groupby("document_count"),
        strict=False,
    ):
        for (method, batch_size), method_group in group.groupby(
            ["method", "batch_size"]
        ):
            method_group = method_group.sort_values("thread_count")
            label = (
                ("Repeated" if method == "repeated_openmp" else "Native batch")
                + f", batch={int(batch_size)}"
            )
            axis.plot(
                method_group["thread_count"],
                method_group["speedup_vs_repeated_serial"],
                marker="o",
                label=label,
            )
        axis.set_title(f"{int(documents):,} documents")
        axis.set_xlabel("OpenMP thread count")
        axis.set_ylabel("Speedup vs repeated serial")
        axis.set_xticks(EXPECTED_OPENMP_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Physical-core scaling for repeated and native batched search")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_memory(summary: pd.DataFrame, output: Path) -> None:
    native = summary[
        (summary["method"] == "native_batch_openmp")
        & (summary["thread_count"] == 6)
    ].copy()

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for axis, (documents, group) in zip(
        axes.ravel(),
        native.groupby("document_count"),
        strict=False,
    ):
        group = group.sort_values("batch_size")
        axis.plot(
            group["batch_size"],
            group["estimated_input_mib"],
            marker="o",
            label="Packed inputs",
        )
        axis.plot(
            group["batch_size"],
            group["estimated_peak_output_mib"],
            marker="o",
            label="Batch outputs",
        )
        axis.plot(
            group["batch_size"],
            group["estimated_peak_total_mib"],
            marker="o",
            label="Estimated total",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"{int(documents):,} documents")
        axis.set_xlabel("Candidate batch size")
        axis.set_ylabel("Estimated native memory (MiB)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.suptitle("Native batched-search memory growth")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (documents, corpus_shingles, batch_size), group in summary.groupby(
        ["document_count", "corpus_shingles", "batch_size"]
    ):
        serial = group[
            (group["method"] == "repeated_serial")
            & (group["thread_count"] == 1)
        ].iloc[0]
        repeated6 = group[
            (group["method"] == "repeated_openmp")
            & (group["thread_count"] == 6)
        ].iloc[0]
        native6 = group[
            (group["method"] == "native_batch_openmp")
            & (group["thread_count"] == 6)
        ].iloc[0]

        variants = {
            "Repeated serial": float(serial["latency_median_ms"]),
            "Repeated OpenMP-6": float(repeated6["latency_median_ms"]),
            "Native batch OpenMP-6": float(native6["latency_median_ms"]),
        }

        rows.append(
            {
                "Documents": int(documents),
                "Corpus shingles": int(corpus_shingles),
                "Batch size": int(batch_size),
                "Repeated serial (ms)": variants["Repeated serial"],
                "Repeated OpenMP-6 (ms)": variants["Repeated OpenMP-6"],
                "Native batch OpenMP-6 (ms)": variants["Native batch OpenMP-6"],
                "Native gain vs repeated OpenMP-6": float(
                    native6["native_batch_gain_vs_repeated_openmp"]
                ),
                "Native speedup vs repeated serial": float(
                    native6["speedup_vs_repeated_serial"]
                ),
                "Best method": min(variants, key=variants.get),
                "Native estimated memory (MiB)": float(
                    native6["estimated_peak_total_mib"]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["Documents", "Batch size"])


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
    raw_path = arguments.raw.expanduser().resolve()
    results_root = arguments.results_root.expanduser().resolve()

    processed_dir = results_root / "processed"
    tables_dir = results_root / "tables"
    figures_dir = results_root / "figures"
    for directory in (processed_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path, engine="python")
    validate_raw_schema(raw)
    summary = summarize_configurations(raw)
    completeness = verify_completeness(raw, summary)

    if not completeness["is_complete"] and not arguments.allow_incomplete:
        raise RuntimeError(
            "Batch benchmark is incomplete: "
            f"{completeness['unique_configurations']}/"
            f"{completeness['expected_configurations']} configurations"
        )
    if not completeness["execution_variants_match"]:
        raise RuntimeError("Batch execution variants do not match the design")
    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate raw batch run rows were detected")
    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError(
            "At least one batch configuration has an incorrect repetition count"
        )

    summary.to_csv(
        processed_dir / "lexical_real_batch_configuration_summary.csv",
        index=False,
    )
    with (
        processed_dir / "lexical_real_batch_completeness_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    compact_table = create_compact_table(summary)
    compact_table.to_csv(
        tables_dir / "lexical_real_batch_table.csv",
        index=False,
    )
    with (tables_dir / "lexical_real_batch_table.md").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("# Lexical Batched-Search Benchmark\n\n")
        file.write(
            "The native batched implementation uses one collapsed OpenMP "
            "candidate-document workload.\n\n"
        )
        file.write(dataframe_to_markdown(compact_table))

    detailed_columns = [
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
        "candidates_per_second_median",
        "comparisons_per_second_median",
        "corpus_shingle_visits_per_second_median",
        "speedup_vs_repeated_serial",
        "parallel_efficiency_percent",
        "native_batch_gain_vs_repeated_openmp",
        "estimated_input_mib",
        "estimated_peak_output_mib",
        "estimated_peak_total_mib",
    ]
    summary[detailed_columns].to_csv(
        tables_dir / "lexical_real_batch_detailed_table.csv",
        index=False,
    )

    plot_latency_vs_batch(
        summary,
        figures_dir / "lexical_batch_latency_vs_batch_size.png",
    )
    plot_comparison_throughput(
        summary,
        figures_dir / "lexical_batch_comparisons_per_second.png",
    )
    plot_native_gain(
        summary,
        figures_dir / "lexical_batch_native_gain.png",
    )
    plot_speedup_vs_threads(
        summary,
        figures_dir / "lexical_batch_speedup_vs_threads.png",
    )
    plot_memory(
        summary,
        figures_dir / "lexical_batch_memory.png",
    )

    print("Lexical batch benchmark analysis")
    print("=" * 72)
    print(
        "Configurations: "
        f"{completeness['unique_configurations']}/"
        f"{completeness['expected_configurations']}"
    )
    print(f"Raw measurement rows: {completeness['raw_measurement_rows']:,}")
    print(f"Methods: {completeness['methods']}")
    print(f"Batch sizes: {completeness['batch_sizes']}")
    print(f"Replication factors: {completeness['replication_factors']}")
    print(f"Thread counts: {completeness['thread_counts']}")
    print(f"Processed data: {processed_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print("=" * 72)
    print("Lexical batch benchmark analysis: PASS")


if __name__ == "__main__":
    main()
