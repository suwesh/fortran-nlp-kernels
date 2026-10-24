from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPOSITORY_ROOT / "perf_eval/results/raw/vector_real_batch_raw.csv"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "perf_eval/results"

EXPECTED_CONFIGURATIONS = 2_184
EXPECTED_METRICS = ["ip", "l2"]
EXPECTED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
EXPECTED_REPLICATION_FACTORS = [1, 10, 100, 250]
EXPECTED_K_VALUES = [1, 10, 50]
EXPECTED_THREADS = [1, 2, 4, 6]
EXPECTED_VARIANTS = {
    ("repeated_serial", 1, 1),
    *(("repeated_openmp", thread, 1) for thread in EXPECTED_THREADS),
    *(("repeated_sgemv", 1, thread) for thread in EXPECTED_THREADS),
    *(("native_sgemm", 1, thread) for thread in EXPECTED_THREADS),
}

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the real-vector physical-core batch matrix."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def p25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def p75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def p95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def validate_schema(data: pd.DataFrame) -> None:
    required = set(CONFIG_KEYS) | {
        "package_version", "run_index", "elapsed_ns", "queries_per_second",
        "comparisons_per_second", "estimated_gflops",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Raw vector batch CSV is missing columns: "
            + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError("Raw vector batch CSV contains no rows")

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
        raise ValueError("elapsed_ns must be positive")


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
    return add_metrics(summary)


def add_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    serial = summary[
        (summary["method"] == "repeated_serial")
        & (summary["thread_count"] == 1)
        & (summary["openblas_thread_count"] == 1)
    ][WORKLOAD_KEYS + ["latency_median_ns"]].rename(
        columns={"latency_median_ns": "serial_latency_median_ns"}
    )
    result = summary.merge(
        serial, on=WORKLOAD_KEYS, how="left", validate="many_to_one"
    )
    if result["serial_latency_median_ns"].isna().any():
        raise ValueError("At least one workload lacks repeated-serial baseline")

    result["effective_thread_count"] = np.where(
        result["method"] == "repeated_openmp",
        result["thread_count"],
        result["openblas_thread_count"],
    )
    result.loc[result["method"] == "repeated_serial", "effective_thread_count"] = 1
    result["speedup_vs_repeated_serial"] = (
        result["serial_latency_median_ns"] / result["latency_median_ns"]
    )
    result["parallel_efficiency_percent"] = np.where(
        result["method"] == "repeated_serial",
        100.0,
        result["speedup_vs_repeated_serial"]
        / result["effective_thread_count"] * 100.0,
    )

    sgemv = summary[summary["method"] == "repeated_sgemv"][
        WORKLOAD_KEYS + ["openblas_thread_count", "latency_median_ns"]
    ].rename(columns={"latency_median_ns": "repeated_sgemv_latency_ns"})
    result = result.merge(
        sgemv,
        on=WORKLOAD_KEYS + ["openblas_thread_count"],
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
    return {
        "raw_measurement_rows": int(len(raw)),
        "unique_configurations": int(len(summary)),
        "expected_configurations": EXPECTED_CONFIGURATIONS,
        "is_complete": len(summary) == EXPECTED_CONFIGURATIONS,
        "metrics": sorted(raw["metric"].astype(str).unique().tolist()),
        "batch_sizes": sorted(int(x) for x in raw["query_batch_size"].unique()),
        "replication_factors": sorted(
            int(x) for x in raw["replication_factor"].unique()
        ),
        "k_values": sorted(int(x) for x in raw["k"].unique()),
        "execution_variant_count": len(variants),
        "execution_variants_match": variants == EXPECTED_VARIANTS,
        "duplicate_run_rows": duplicate_rows,
        "repetition_mismatch_configurations": int(len(mismatches)),
        "package_versions": sorted(
            raw["package_version"].astype(str).unique().tolist()
        ),
    }


def method_label(method: str, threads: int) -> str:
    names = {
        "repeated_serial": "Repeated serial",
        "repeated_openmp": "Repeated OpenMP",
        "repeated_sgemv": "Repeated SGEMV",
        "native_sgemm": "Native SGEMM",
    }
    if method == "repeated_serial":
        return names[method]
    return f"{names[method]}-{int(threads)}"


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def selected_primary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["k"] == 10)
        & (
            (summary["method"] == "repeated_serial")
            | ((summary["method"] == "repeated_openmp") & (summary["thread_count"] == 6))
            | ((summary["method"] == "repeated_sgemv") & (summary["openblas_thread_count"] == 6))
            | ((summary["method"] == "native_sgemm") & (summary["openblas_thread_count"] == 6))
        )
    ].copy()


def four_panel_by_scale(
    data: pd.DataFrame,
    value: str,
    ylabel: str,
    title: str,
    output: Path,
    reference_one: bool = False,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for axis, (vector_count, group) in zip(
        axes.ravel(), data.groupby("vector_count"), strict=False
    ):
        for (method, threads), method_group in group.groupby(
            ["method", "effective_thread_count"]
        ):
            method_group = method_group.sort_values("query_batch_size")
            axis.plot(
                method_group["query_batch_size"], method_group[value],
                marker="o", label=method_label(method, threads),
            )
        if reference_one:
            axis.axhline(1.0, linestyle="--", label="Break-even")
        axis.set_xscale("log", base=2)
        axis.set_title(f"{int(vector_count):,} vectors")
        axis.set_xlabel("Query batch size")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_latency(summary: pd.DataFrame, figures: Path) -> None:
    for metric in EXPECTED_METRICS:
        data = selected_primary(summary)
        data = data[data["metric"] == metric]
        four_panel_by_scale(
            data, "latency_median_ms", "Median batch latency (ms)",
            f"{metric.upper()} batch latency, k=10",
            figures / f"vector_batch_{metric}_latency.png",
        )


def plot_throughput(summary: pd.DataFrame, figures: Path) -> None:
    for metric in EXPECTED_METRICS:
        data = selected_primary(summary)
        data = data[data["metric"] == metric]
        four_panel_by_scale(
            data, "estimated_gflops_median", "Estimated complete-search GFLOP/s",
            f"{metric.upper()} batch throughput, k=10",
            figures / f"vector_batch_{metric}_gflops.png",
        )


def plot_sgemm_gain(summary: pd.DataFrame, output: Path) -> None:
    data = summary[
        (summary["method"] == "native_sgemm")
        & (summary["k"] == 10)
    ].copy()
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
    for axis, ((metric, vector_count), group) in zip(
        axes.ravel(), data.groupby(["metric", "vector_count"]), strict=False
    ):
        for threads, thread_group in group.groupby("effective_thread_count"):
            thread_group = thread_group.sort_values("query_batch_size")
            axis.plot(
                thread_group["query_batch_size"],
                thread_group["native_sgemm_gain_vs_repeated_sgemv"],
                marker="o", label=f"BLAS-{int(threads)}",
            )
        axis.axhline(1.0, linestyle="--", label="No gain")
        axis.set_xscale("log", base=2)
        axis.set_title(f"{metric.upper()}, {int(vector_count):,} vectors")
        axis.set_xlabel("Query batch size")
        axis.set_ylabel("SGEMM gain vs repeated SGEMV")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Native SGEMM benefit at matching OpenBLAS thread counts")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_k_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    data = summary[
        summary["vector_count"].isin([4_750, 47_500, 118_750])
        & summary["query_batch_size"].isin([1, 8, 64])
        & (
            (summary["method"] == "repeated_serial")
            | ((summary["method"] == "repeated_openmp") & (summary["thread_count"] == 6))
            | ((summary["method"] == "repeated_sgemv") & (summary["openblas_thread_count"] == 6))
            | ((summary["method"] == "native_sgemm") & (summary["openblas_thread_count"] == 6))
        )
    ].copy()
    # Compact overview: median across selected database and batch scales.
    reduced = (
        data.groupby(["metric", "method", "effective_thread_count", "k"], as_index=False)
        ["latency_median_ms"].median()
    )
    plt.figure(figsize=(10, 6))
    for (metric, method, threads), group in reduced.groupby(
        ["metric", "method", "effective_thread_count"]
    ):
        group = group.sort_values("k")
        plt.plot(
            group["k"], group["latency_median_ms"], marker="o",
            label=f"{metric.upper()} {method_label(method, threads)}",
        )
    plt.xlabel("k")
    plt.ylabel("Median latency across selected workloads (ms)")
    plt.title("Vector batch complete-search top-k sensitivity")
    plt.xticks(EXPECTED_K_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    save_figure(output)


def plot_scaling(summary: pd.DataFrame, output: Path) -> None:
    data = summary[
        (summary["k"] == 10)
        & (summary["query_batch_size"].isin([8, 64]))
        & (summary["vector_count"].isin([4_750, 47_500, 118_750]))
        & (summary["method"] != "repeated_serial")
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    for axis, metric in zip(axes.ravel(), EXPECTED_METRICS, strict=False):
        metric_data = data[data["metric"] == metric]
        for (method, batch, vectors), group in metric_data.groupby(
            ["method", "query_batch_size", "vector_count"]
        ):
            group = group.sort_values("effective_thread_count")
            axis.plot(
                group["effective_thread_count"], group["speedup_vs_repeated_serial"],
                marker="o",
                label=f"{method}, b={int(batch)}, n={int(vectors):,}",
            )
        axis.set_title(metric.upper())
        axis.set_xlabel("Backend threads")
        axis.set_ylabel("Speedup versus repeated serial")
        axis.set_xticks(EXPECTED_THREADS)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=6)
    figure.suptitle("Vector batch physical-core scaling, k=10")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_memory(summary: pd.DataFrame, output: Path) -> None:
    memory = (
        summary.groupby(["vector_count", "query_batch_size"], as_index=False)
        ["estimated_total_mib"].median()
    )
    plt.figure(figsize=(10, 6))
    for batch, group in memory.groupby("query_batch_size"):
        group = group.sort_values("vector_count")
        plt.plot(
            group["vector_count"], group["estimated_total_mib"],
            marker="o", label=f"batch {int(batch)}",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Database vectors")
    plt.ylabel("Estimated native input and top-k output memory (MiB)")
    plt.title("Vector batch memory growth")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    save_figure(output)


def create_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary[summary["k"] == 10]
    rows = []
    for (metric, vectors, batch), group in data.groupby(
        ["metric", "vector_count", "query_batch_size"]
    ):
        chosen = {}
        selectors = {
            "Repeated serial": ("repeated_serial", 1),
            "Repeated OpenMP-6": ("repeated_openmp", 6),
            "Repeated SGEMV-6": ("repeated_sgemv", 6),
            "Native SGEMM-6": ("native_sgemm", 6),
        }
        for label, (method, threads) in selectors.items():
            row = group[
                (group["method"] == method)
                & (group["effective_thread_count"] == threads)
            ].iloc[0]
            chosen[label] = row
        latencies = {
            label: float(row["latency_median_ms"])
            for label, row in chosen.items()
        }
        native = chosen["Native SGEMM-6"]
        rows.append({
            "Metric": metric.upper(),
            "Vectors": int(vectors),
            "Query batch": int(batch),
            "Repeated serial (ms)": latencies["Repeated serial"],
            "Repeated OpenMP-6 (ms)": latencies["Repeated OpenMP-6"],
            "Repeated SGEMV-6 (ms)": latencies["Repeated SGEMV-6"],
            "Native SGEMM-6 (ms)": latencies["Native SGEMM-6"],
            "SGEMM gain vs SGEMV-6": float(
                native["native_sgemm_gain_vs_repeated_sgemv"]
            ),
            "Best method": min(latencies, key=latencies.get),
            "Estimated memory (MiB)": float(native["estimated_total_mib"]),
        })
    return pd.DataFrame(rows).sort_values(["Metric", "Vectors", "Query batch"])


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
    raw_path = args.raw.expanduser().resolve()
    root = args.results_root.expanduser().resolve()
    processed = root / "processed"
    tables = root / "tables"
    figures = root / "figures"
    for directory in (processed, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path, engine="python")
    validate_schema(raw)
    summary = summarize(raw)
    completeness = verify_completeness(raw, summary)

    if not completeness["is_complete"] and not args.allow_incomplete:
        raise RuntimeError(
            "Vector batch benchmark is incomplete: "
            f"{completeness['unique_configurations']}/"
            f"{completeness['expected_configurations']} configurations"
        )
    if not completeness["execution_variants_match"]:
        raise RuntimeError("Execution variants do not match matrix design")
    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate raw run rows were detected")
    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError("At least one repetition count is incorrect")

    summary.to_csv(
        processed / "vector_real_batch_configuration_summary.csv", index=False
    )
    with (processed / "vector_real_batch_completeness_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    compact = create_compact_table(summary)
    compact.to_csv(tables / "vector_real_batch_table.csv", index=False)
    with (tables / "vector_real_batch_table.md").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("# Vector Batch CPU Benchmark\n\n")
        file.write(
            "Timings measure complete native search: scoring, shared serial "
            "O(Nk) top-k, binding, and output creation.\n\n"
        )
        file.write(dataframe_to_markdown(compact))

    detailed_columns = [
        "metric", "method", "query_batch_size", "vector_count",
        "replication_factor", "dimension", "k", "thread_count",
        "openblas_thread_count", "effective_thread_count", "repetition_count",
        "latency_median_ms", "latency_p95_ms", "latency_cv",
        "queries_per_second_median", "comparisons_per_second_median",
        "estimated_gflops_median", "speedup_vs_repeated_serial",
        "parallel_efficiency_percent", "native_sgemm_gain_vs_repeated_sgemv",
        "estimated_total_mib",
    ]
    summary[detailed_columns].to_csv(
        tables / "vector_real_batch_detailed_table.csv", index=False
    )

    plot_latency(summary, figures)
    plot_throughput(summary, figures)
    plot_sgemm_gain(summary, figures / "vector_batch_sgemm_gain.png")
    plot_k_sensitivity(summary, figures / "vector_batch_k_sensitivity.png")
    plot_scaling(summary, figures / "vector_batch_scaling.png")
    plot_memory(summary, figures / "vector_batch_memory.png")

    print("Vector batch benchmark analysis")
    print("=" * 72)
    print(
        f"Configurations: {completeness['unique_configurations']}/"
        f"{completeness['expected_configurations']}"
    )
    print(f"Raw rows: {completeness['raw_measurement_rows']:,}")
    print(f"Metrics: {completeness['metrics']}")
    print(f"Batch sizes: {completeness['batch_sizes']}")
    print(f"Replication factors: {completeness['replication_factors']}")
    print(f"k values: {completeness['k_values']}")
    print(f"Processed data: {processed}")
    print(f"Tables: {tables}")
    print(f"Figures: {figures}")
    print("=" * 72)
    print("Vector batch benchmark analysis: PASS")


if __name__ == "__main__":
    main()
