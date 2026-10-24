from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_RAW = (
    REPOSITORY_ROOT / "perf_eval/results/raw/vector_real_batch_raw.csv"
)
DEFAULT_SMT_RAW = (
    REPOSITORY_ROOT / "perf_eval/results/raw/vector_real_batch_smt_raw.csv"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "perf_eval/results"

EXPECTED_SMT_CONFIGURATIONS = 486
EXPECTED_METRICS = ["ip", "l2"]
EXPECTED_REPLICATION_FACTORS = [10, 100, 250]
EXPECTED_BATCH_SIZES = [8, 32, 64]
EXPECTED_K_VALUES = [1, 10, 50]
EXPECTED_METHODS = ["repeated_openmp", "repeated_sgemv", "native_sgemm"]
EXPECTED_SMT_THREADS = [8, 10, 12]
COMPARISON_THREADS = [6, 8, 10, 12]

CONFIG_KEYS = [
    "metric", "method", "query_batch_size", "dimension", "k",
    "base_vector_count", "replication_factor", "vector_count",
    "query_vector_comparisons", "thread_count", "openblas_thread_count",
    "omp_dynamic", "omp_proc_bind", "omp_places", "omp_wait_policy",
    "warmup_runs", "measured_runs", "floating_point_operations_estimate",
    "estimated_input_bytes", "estimated_output_bytes",
]

WORKLOAD_KEYS = [
    "metric", "query_batch_size", "dimension", "k", "base_vector_count",
    "replication_factor", "vector_count", "query_vector_comparisons",
]
METHOD_KEYS = WORKLOAD_KEYS + ["method"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze vector batch SMT sensitivity by combining physical-core "
            "six-thread baselines with 8/10/12-thread measurements."
        )
    )
    parser.add_argument("--physical-raw", type=Path, default=DEFAULT_PHYSICAL_RAW)
    parser.add_argument("--smt-raw", type=Path, default=DEFAULT_SMT_RAW)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def p25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def p75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def p95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def validate_schema(data: pd.DataFrame, label: str) -> None:
    required = set(CONFIG_KEYS) | {
        "package_version", "run_index", "elapsed_ns", "queries_per_second",
        "comparisons_per_second", "estimated_gflops",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{label} CSV is missing columns: " + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError(f"{label} CSV contains no rows")

    numeric = [
        "query_batch_size", "dimension", "k", "base_vector_count",
        "replication_factor", "vector_count", "query_vector_comparisons",
        "thread_count", "openblas_thread_count", "warmup_runs",
        "measured_runs", "run_index", "elapsed_ns", "queries_per_second",
        "comparisons_per_second", "floating_point_operations_estimate",
        "estimated_gflops", "estimated_input_bytes", "estimated_output_bytes",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (data["elapsed_ns"] <= 0).any():
        raise ValueError(f"{label} elapsed_ns must be positive")


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(CONFIG_KEYS, dropna=False, as_index=False)
        .agg(
            repetition_count=("elapsed_ns", "size"),
            latency_p25_ns=("elapsed_ns", p25),
            latency_median_ns=("elapsed_ns", "median"),
            latency_mean_ns=("elapsed_ns", "mean"),
            latency_p75_ns=("elapsed_ns", p75),
            latency_p95_ns=("elapsed_ns", p95),
            latency_std_ns=("elapsed_ns", "std"),
            queries_per_second_median=("queries_per_second", "median"),
            comparisons_per_second_median=("comparisons_per_second", "median"),
            estimated_gflops_median=("estimated_gflops", "median"),
        )
    )
    summary["latency_std_ns"] = summary["latency_std_ns"].fillna(0.0)
    summary["latency_cv"] = np.divide(
        summary["latency_std_ns"], summary["latency_mean_ns"],
        out=np.zeros(len(summary), dtype=np.float64),
        where=summary["latency_mean_ns"].to_numpy() != 0,
    )
    for suffix in ("p25", "median", "mean", "p75", "p95", "std"):
        summary[f"latency_{suffix}_ms"] = (
            summary[f"latency_{suffix}_ns"] / 1_000_000.0
        )
    summary["estimated_total_mib"] = (
        summary["estimated_input_bytes"] + summary["estimated_output_bytes"]
    ) / (1024.0 ** 2)
    return summary


def select_physical_baselines(
    physical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = physical[
        physical["metric"].isin(EXPECTED_METRICS)
        & physical["replication_factor"].isin(EXPECTED_REPLICATION_FACTORS)
        & physical["query_batch_size"].isin(EXPECTED_BATCH_SIZES)
        & physical["k"].isin(EXPECTED_K_VALUES)
    ].copy()

    serial = selected[
        (selected["method"] == "repeated_serial")
        & (selected["thread_count"] == 1)
        & (selected["openblas_thread_count"] == 1)
    ].copy()

    six = selected[
        (
            (selected["method"] == "repeated_openmp")
            & (selected["thread_count"] == 6)
            & (selected["openblas_thread_count"] == 1)
        )
        | (
            selected["method"].isin(["repeated_sgemv", "native_sgemm"])
            & (selected["thread_count"] == 1)
            & (selected["openblas_thread_count"] == 6)
        )
    ].copy()

    workloads = (
        len(EXPECTED_METRICS) * len(EXPECTED_REPLICATION_FACTORS)
        * len(EXPECTED_BATCH_SIZES) * len(EXPECTED_K_VALUES)
    )
    if len(serial) != workloads:
        raise ValueError("Physical batch CSV lacks repeated-serial baselines")
    if len(six) != workloads * len(EXPECTED_METHODS):
        raise ValueError("Physical batch CSV lacks all six-thread baselines")
    return serial, six


def add_metrics(
    combined: pd.DataFrame,
    serial: pd.DataFrame,
    six: pd.DataFrame,
) -> pd.DataFrame:
    serial_lookup = serial[
        WORKLOAD_KEYS + ["latency_median_ns"]
    ].rename(columns={"latency_median_ns": "serial_latency_median_ns"})
    six_lookup = six[
        METHOD_KEYS + ["latency_median_ns"]
    ].rename(columns={"latency_median_ns": "six_thread_latency_median_ns"})

    result = combined.merge(
        serial_lookup, on=WORKLOAD_KEYS, how="left", validate="many_to_one"
    ).merge(
        six_lookup, on=METHOD_KEYS, how="left", validate="many_to_one"
    )
    required = ["serial_latency_median_ns", "six_thread_latency_median_ns"]
    if result[required].isna().any().any():
        raise ValueError("At least one SMT row lacks a matching baseline")

    result["effective_thread_count"] = np.where(
        result["method"] == "repeated_openmp",
        result["thread_count"],
        result["openblas_thread_count"],
    )
    result["speedup_vs_repeated_serial"] = (
        result["serial_latency_median_ns"] / result["latency_median_ns"]
    )
    result["parallel_efficiency_percent"] = (
        result["speedup_vs_repeated_serial"]
        / result["effective_thread_count"] * 100.0
    )
    result["gain_vs_six_threads"] = (
        result["six_thread_latency_median_ns"] / result["latency_median_ns"]
    )
    result["latency_change_vs_six_percent"] = (
        result["latency_median_ns"] / result["six_thread_latency_median_ns"]
        - 1.0
    ) * 100.0

    sgemv = result[result["method"] == "repeated_sgemv"][
        WORKLOAD_KEYS + ["effective_thread_count", "latency_median_ns"]
    ].rename(columns={"latency_median_ns": "repeated_sgemv_latency_ns"})
    result = result.merge(
        sgemv,
        on=WORKLOAD_KEYS + ["effective_thread_count"],
        how="left",
        validate="many_to_one",
    )
    result["native_sgemm_gain_vs_repeated_sgemv"] = np.where(
        result["method"] == "native_sgemm",
        result["repeated_sgemv_latency_ns"] / result["latency_median_ns"],
        np.nan,
    )
    return result


def verify_completeness(raw: pd.DataFrame, summary: pd.DataFrame) -> dict:
    duplicate_rows = int(raw.duplicated(CONFIG_KEYS + ["run_index"]).sum())
    mismatches = summary[
        summary["repetition_count"] != summary["measured_runs"]
    ]
    variants = set(
        raw[["method", "thread_count", "openblas_thread_count"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    expected_variants = {
        *(("repeated_openmp", thread, 1) for thread in EXPECTED_SMT_THREADS),
        *(("repeated_sgemv", 1, thread) for thread in EXPECTED_SMT_THREADS),
        *(("native_sgemm", 1, thread) for thread in EXPECTED_SMT_THREADS),
    }
    return {
        "raw_measurement_rows": int(len(raw)),
        "unique_smt_configurations": int(len(summary)),
        "expected_smt_configurations": EXPECTED_SMT_CONFIGURATIONS,
        "is_complete": len(summary) == EXPECTED_SMT_CONFIGURATIONS,
        "metrics": sorted(raw["metric"].astype(str).unique().tolist()),
        "batch_sizes": sorted(int(x) for x in raw["query_batch_size"].unique()),
        "replication_factors": sorted(
            int(x) for x in raw["replication_factor"].unique()
        ),
        "k_values": sorted(int(x) for x in raw["k"].unique()),
        "execution_variants_match": variants == expected_variants,
        "duplicate_run_rows": duplicate_rows,
        "repetition_mismatch_configurations": int(len(mismatches)),
        "package_versions": sorted(
            raw["package_version"].astype(str).unique().tolist()
        ),
    }


def method_label(method: str) -> str:
    return {
        "repeated_openmp": "Repeated OpenMP",
        "repeated_sgemv": "Repeated SGEMV",
        "native_sgemm": "Native SGEMM",
    }[method]


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def matrix_figure(
    data: pd.DataFrame,
    value: str,
    ylabel: str,
    title: str,
    output: Path,
    reference_one: bool = False,
) -> None:
    selected = data[data["k"] == 10]
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)
    groups = selected.groupby(["vector_count", "query_batch_size"])
    for axis, ((vectors, batch), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("effective_thread_count")
            axis.plot(
                method_group["effective_thread_count"], method_group[value],
                marker="o", label=method_label(method),
            )
        if reference_one:
            axis.axhline(1.0, linestyle="--", label="No gain over 6")
        axis.set_title(f"{int(vectors):,} vectors, batch {int(batch)}")
        axis.set_xlabel("Backend threads")
        axis.set_ylabel(ylabel)
        axis.set_xticks(COMPARISON_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_sgemm_gain(data: pd.DataFrame, output: Path) -> None:
    selected = data[(data["k"] == 10) & (data["method"] == "native_sgemm")]
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)
    for axis, ((vectors, batch), group) in zip(
        axes.ravel(), selected.groupby(["vector_count", "query_batch_size"]),
        strict=False,
    ):
        for metric, metric_group in group.groupby("metric"):
            metric_group = metric_group.sort_values("effective_thread_count")
            axis.plot(
                metric_group["effective_thread_count"],
                metric_group["native_sgemm_gain_vs_repeated_sgemv"],
                marker="o", label=metric.upper(),
            )
        axis.axhline(1.0, linestyle="--", label="No gain")
        axis.set_title(f"{int(vectors):,} vectors, batch {int(batch)}")
        axis.set_xlabel("OpenBLAS threads")
        axis.set_ylabel("SGEMM gain vs repeated SGEMV")
        axis.set_xticks(COMPARISON_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Native SGEMM benefit under SMT, k=10")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_table(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric", "method", "query_batch_size", "replication_factor",
        "vector_count", "k", "effective_thread_count", "repetition_count",
        "latency_median_ms", "latency_p95_ms", "latency_cv",
        "queries_per_second_median", "comparisons_per_second_median",
        "estimated_gflops_median", "speedup_vs_repeated_serial",
        "parallel_efficiency_percent", "gain_vs_six_threads",
        "native_sgemm_gain_vs_repeated_sgemv", "estimated_total_mib",
    ]
    return data[columns].sort_values(
        ["metric", "k", "vector_count", "query_batch_size", "method",
         "effective_thread_count"]
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
    args = parse_arguments()
    physical_path = args.physical_raw.expanduser().resolve()
    smt_path = args.smt_raw.expanduser().resolve()
    root = args.results_root.expanduser().resolve()
    processed = root / "processed"
    tables = root / "tables"
    figures = root / "figures"
    for directory in (processed, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    physical_raw = pd.read_csv(physical_path, engine="python")
    smt_raw = pd.read_csv(smt_path, engine="python")
    validate_schema(physical_raw, "Physical batch")
    validate_schema(smt_raw, "Batch SMT")

    physical_summary = summarize(physical_raw)
    smt_summary = summarize(smt_raw)
    serial, six = select_physical_baselines(physical_summary)
    combined = pd.concat([six, smt_summary], ignore_index=True)
    combined = add_metrics(combined, serial, six)
    combined = combined.sort_values(
        ["metric", "k", "vector_count", "query_batch_size", "method",
         "effective_thread_count"]
    )
    completeness = verify_completeness(smt_raw, smt_summary)

    if not completeness["is_complete"]:
        raise RuntimeError(
            "Vector batch SMT benchmark is incomplete: "
            f"{completeness['unique_smt_configurations']}/"
            f"{completeness['expected_smt_configurations']} configurations"
        )
    if not completeness["execution_variants_match"]:
        raise RuntimeError("SMT execution variants do not match matrix design")
    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate SMT run rows were detected")
    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError("At least one SMT repetition count is incorrect")

    combined.to_csv(
        processed / "vector_real_batch_smt_configuration_summary.csv",
        index=False,
    )
    with (processed / "vector_real_batch_smt_completeness_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    table = create_table(combined)
    table.to_csv(tables / "vector_real_batch_smt_table.csv", index=False)
    with (tables / "vector_real_batch_smt_table.md").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("# Vector Batch SMT Sensitivity\n\n")
        file.write(
            "Six-thread baselines are reused from the physical-core batch "
            "matrix. Timings include scoring, serial O(Nk) top-k, binding, "
            "and output creation.\n\n"
        )
        file.write(dataframe_to_markdown(table))

    for metric in EXPECTED_METRICS:
        metric_data = combined[combined["metric"] == metric]
        matrix_figure(
            metric_data, "latency_median_ms", "Median batch latency (ms)",
            f"{metric.upper()} vector batch SMT latency, k=10",
            figures / f"vector_batch_smt_{metric}_latency.png",
        )
        matrix_figure(
            metric_data, "gain_vs_six_threads", "Gain versus six threads",
            f"{metric.upper()} incremental SMT gain, k=10",
            figures / f"vector_batch_smt_{metric}_gain_vs_six.png",
            reference_one=True,
        )
        matrix_figure(
            metric_data, "parallel_efficiency_percent", "Parallel efficiency (%)",
            f"{metric.upper()} vector batch SMT efficiency, k=10",
            figures / f"vector_batch_smt_{metric}_efficiency.png",
        )
        matrix_figure(
            metric_data, "latency_cv", "Latency coefficient of variation",
            f"{metric.upper()} vector batch SMT variability, k=10",
            figures / f"vector_batch_smt_{metric}_variability.png",
        )

    plot_sgemm_gain(combined, figures / "vector_batch_smt_sgemm_gain.png")

    print("Vector batch SMT analysis")
    print("=" * 72)
    print(
        f"SMT configurations: {completeness['unique_smt_configurations']}/"
        f"{completeness['expected_smt_configurations']}"
    )
    print(f"Raw SMT rows: {completeness['raw_measurement_rows']:,}")
    print(f"Metrics: {completeness['metrics']}")
    print(f"Batch sizes: {completeness['batch_sizes']}")
    print(f"Replication factors: {completeness['replication_factors']}")
    print(f"k values: {completeness['k_values']}")
    print("Combined comparison threads: [6, 8, 10, 12]")
    print(f"Processed data: {processed}")
    print(f"Tables: {tables}")
    print(f"Figures: {figures}")
    print("=" * 72)
    print("Vector batch SMT analysis: PASS")


if __name__ == "__main__":
    main()
