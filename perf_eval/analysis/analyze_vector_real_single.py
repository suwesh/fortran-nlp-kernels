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
    / "perf_eval/results/raw/vector_real_single_raw.csv"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "perf_eval/results"

EXPECTED_CONFIGURATIONS = 1_728
EXPECTED_METRICS = ["ip", "l2"]
EXPECTED_QUERY_INDICES = [0, 67, 208, 474]
EXPECTED_REPLICATION_FACTORS = [1, 2, 5, 10, 25, 50, 100, 250]
EXPECTED_K_VALUES = [1, 10, 50]
EXPECTED_VARIANTS = {
    ("serial", 1, 1),
    ("openmp", 1, 1),
    ("openmp", 2, 1),
    ("openmp", 4, 1),
    ("openmp", 6, 1),
    ("blas", 1, 1),
    ("blas", 1, 2),
    ("blas", 1, 4),
    ("blas", 1, 6),
}

CONFIG_KEYS = [
    "metric",
    "backend",
    "query_index",
    "dimension",
    "k",
    "base_vector_count",
    "replication_factor",
    "vector_count",
    "thread_count",
    "openblas_thread_count",
    "omp_dynamic",
    "omp_proc_bind",
    "omp_places",
    "omp_wait_policy",
    "warmup_runs",
    "measured_runs",
    "floating_point_operations_estimate",
    "estimated_input_bytes",
    "estimated_output_bytes",
]

BASELINE_KEYS = [
    "metric",
    "query_index",
    "dimension",
    "k",
    "base_vector_count",
    "replication_factor",
    "vector_count",
]

AGGREGATE_KEYS = [
    "metric",
    "dimension",
    "k",
    "base_vector_count",
    "replication_factor",
    "vector_count",
    "backend",
    "thread_count",
    "openblas_thread_count",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the real-vector single-query physical-core matrix and "
            "generate completeness reports, processed tables, and figures."
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
        help="Generate exploratory outputs from an incomplete matrix.",
    )
    return parser.parse_args()


def percentile_25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def percentile_75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def percentile_95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def validate_schema(data: pd.DataFrame) -> None:
    required = set(CONFIG_KEYS) | {
        "package_version",
        "run_index",
        "elapsed_ns",
        "queries_per_second",
        "vectors_scored_per_second",
        "estimated_gflops",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Raw vector CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError("Raw vector CSV contains no measurements")

    numeric_columns = [
        "query_index",
        "dimension",
        "k",
        "base_vector_count",
        "replication_factor",
        "vector_count",
        "thread_count",
        "openblas_thread_count",
        "warmup_runs",
        "measured_runs",
        "run_index",
        "elapsed_ns",
        "queries_per_second",
        "vectors_scored_per_second",
        "floating_point_operations_estimate",
        "estimated_gflops",
        "estimated_input_bytes",
        "estimated_output_bytes",
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
            queries_per_second_median=("queries_per_second", "median"),
            vectors_scored_per_second_median=(
                "vectors_scored_per_second",
                "median",
            ),
            estimated_gflops_median=("estimated_gflops", "median"),
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
    summary["estimated_output_mib"] = (
        summary["estimated_output_bytes"] / (1024.0**2)
    )
    summary["estimated_total_mib"] = (
        summary["estimated_input_mib"]
        + summary["estimated_output_mib"]
    )

    return add_comparative_metrics(summary)


def add_comparative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    serial = summary[
        (summary["backend"] == "serial")
        & (summary["thread_count"] == 1)
        & (summary["openblas_thread_count"] == 1)
    ][BASELINE_KEYS + ["latency_median_ns"]].rename(
        columns={"latency_median_ns": "serial_latency_median_ns"}
    )

    if serial.duplicated(BASELINE_KEYS).any():
        raise ValueError("Multiple serial baselines exist for one workload")

    result = summary.merge(
        serial,
        on=BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    )

    if result["serial_latency_median_ns"].isna().any():
        raise ValueError("At least one configuration lacks a serial baseline")

    result["speedup_vs_serial"] = (
        result["serial_latency_median_ns"]
        / result["latency_median_ns"]
    )
    result["effective_thread_count"] = np.where(
        result["backend"] == "blas",
        result["openblas_thread_count"],
        result["thread_count"],
    )
    result["parallel_efficiency_percent"] = np.where(
        result["backend"] == "serial",
        100.0,
        result["speedup_vs_serial"]
        / result["effective_thread_count"]
        * 100.0,
    )

    return result


def aggregate_queries(summary: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        summary.groupby(AGGREGATE_KEYS, as_index=False)
        .agg(
            query_count=("query_index", "nunique"),
            latency_median_across_queries_ms=(
                "latency_median_ms",
                "median",
            ),
            latency_p25_across_queries_ms=(
                "latency_median_ms",
                percentile_25,
            ),
            latency_p75_across_queries_ms=(
                "latency_median_ms",
                percentile_75,
            ),
            latency_p95_median_across_queries_ms=(
                "latency_p95_ms",
                "median",
            ),
            latency_cv_median_across_queries=(
                "latency_cv",
                "median",
            ),
            vectors_scored_per_second_median_across_queries=(
                "vectors_scored_per_second_median",
                "median",
            ),
            gflops_median_across_queries=(
                "estimated_gflops_median",
                "median",
            ),
            speedup_median_across_queries=(
                "speedup_vs_serial",
                "median",
            ),
            speedup_p25_across_queries=(
                "speedup_vs_serial",
                percentile_25,
            ),
            speedup_p75_across_queries=(
                "speedup_vs_serial",
                percentile_75,
            ),
            efficiency_median_percent=(
                "parallel_efficiency_percent",
                "median",
            ),
            estimated_total_mib=("estimated_total_mib", "median"),
        )
    )
    aggregated["effective_thread_count"] = np.where(
        aggregated["backend"] == "blas",
        aggregated["openblas_thread_count"],
        aggregated["thread_count"],
    )
    return aggregated.sort_values(
        ["metric", "k", "vector_count", "backend", "effective_thread_count"]
    )


def verify_completeness(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict:
    unique_configurations = len(summary)
    duplicate_rows = int(
        raw.duplicated(CONFIG_KEYS + ["run_index"]).sum()
    )
    mismatches = summary[
        summary["repetition_count"] != summary["measured_runs"]
    ]

    actual_variants = set(
        raw[["backend", "thread_count", "openblas_thread_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    return {
        "raw_measurement_rows": int(len(raw)),
        "unique_configurations": unique_configurations,
        "expected_configurations": EXPECTED_CONFIGURATIONS,
        "is_complete": unique_configurations == EXPECTED_CONFIGURATIONS,
        "metrics": sorted(raw["metric"].astype(str).unique().tolist()),
        "query_indices": sorted(
            int(value) for value in raw["query_index"].unique()
        ),
        "replication_factors": sorted(
            int(value) for value in raw["replication_factor"].unique()
        ),
        "k_values": sorted(int(value) for value in raw["k"].unique()),
        "execution_variant_count": len(actual_variants),
        "execution_variants_match": actual_variants == EXPECTED_VARIANTS,
        "duplicate_run_rows": duplicate_rows,
        "repetition_mismatch_configurations": int(len(mismatches)),
        "package_versions": sorted(
            raw["package_version"].astype(str).unique().tolist()
        ),
    }


def backend_label(
    backend: str,
    thread_count: int,
    openblas_thread_count: int,
) -> str:
    if backend == "serial":
        return "Fortran serial"
    if backend == "openmp":
        return f"Fortran OpenMP-{int(thread_count)}"
    return f"OpenBLAS SGEMV-{int(openblas_thread_count)}"


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def selected_k10(aggregated: pd.DataFrame) -> pd.DataFrame:
    return aggregated[aggregated["k"] == 10].copy()


def plot_latency(aggregated: pd.DataFrame, output: Path) -> None:
    selected = selected_k10(aggregated)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)

    for axis, metric in zip(axes.ravel(), EXPECTED_METRICS, strict=False):
        metric_data = selected[selected["metric"] == metric]
        for (backend, omp, blas), group in metric_data.groupby(
            ["backend", "thread_count", "openblas_thread_count"]
        ):
            group = group.sort_values("vector_count")
            axis.plot(
                group["vector_count"],
                group["latency_median_across_queries_ms"],
                marker="o",
                label=backend_label(backend, omp, blas),
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(metric.upper())
        axis.set_xlabel("Database vectors")
        axis.set_ylabel("Median complete-search latency (ms)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Vector single-query latency, k=10")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_speedup(aggregated: pd.DataFrame, output: Path) -> None:
    selected = selected_k10(aggregated)
    selected = selected[selected["backend"] != "serial"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)

    for axis, metric in zip(axes.ravel(), EXPECTED_METRICS, strict=False):
        metric_data = selected[selected["metric"] == metric]
        for (backend, omp, blas), group in metric_data.groupby(
            ["backend", "thread_count", "openblas_thread_count"]
        ):
            group = group.sort_values("vector_count")
            axis.plot(
                group["vector_count"],
                group["speedup_median_across_queries"],
                marker="o",
                label=backend_label(backend, omp, blas),
            )
        axis.axhline(1.0, linestyle="--", label="Break-even")
        axis.set_xscale("log")
        axis.set_title(metric.upper())
        axis.set_xlabel("Database vectors")
        axis.set_ylabel("Speedup versus Fortran serial")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Vector single-query backend speedup, k=10")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_gflops(aggregated: pd.DataFrame, output: Path) -> None:
    selected = selected_k10(aggregated)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)

    for axis, metric in zip(axes.ravel(), EXPECTED_METRICS, strict=False):
        metric_data = selected[selected["metric"] == metric]
        for (backend, omp, blas), group in metric_data.groupby(
            ["backend", "thread_count", "openblas_thread_count"]
        ):
            group = group.sort_values("vector_count")
            axis.plot(
                group["vector_count"],
                group["gflops_median_across_queries"],
                marker="o",
                label=backend_label(backend, omp, blas),
            )
        axis.set_xscale("log")
        axis.set_title(metric.upper())
        axis.set_xlabel("Database vectors")
        axis.set_ylabel("Estimated complete-search GFLOP/s")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Vector single-query estimated throughput, k=10")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_k_sensitivity(aggregated: pd.DataFrame, output: Path) -> None:
    selected_counts = [4_750, 47_500, 118_750]
    selected = aggregated[
        aggregated["vector_count"].isin(selected_counts)
        & (
            ((aggregated["backend"] == "serial"))
            | (
                (aggregated["backend"] == "openmp")
                & (aggregated["thread_count"] == 6)
            )
            | (
                (aggregated["backend"] == "blas")
                & (aggregated["openblas_thread_count"] == 6)
            )
        )
    ].copy()

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    groups = selected.groupby(["metric", "vector_count"])
    for axis, ((metric, vector_count), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for (backend, omp, blas), backend_group in group.groupby(
            ["backend", "thread_count", "openblas_thread_count"]
        ):
            backend_group = backend_group.sort_values("k")
            axis.plot(
                backend_group["k"],
                backend_group["latency_median_across_queries_ms"],
                marker="o",
                label=backend_label(backend, omp, blas),
            )
        axis.set_title(f"{metric.upper()}, {int(vector_count):,} vectors")
        axis.set_xlabel("k")
        axis.set_ylabel("Median latency (ms)")
        axis.set_xticks(EXPECTED_K_VALUES)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Top-k sensitivity in complete vector search")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_efficiency(aggregated: pd.DataFrame, output: Path) -> None:
    selected_counts = [4_750, 47_500, 118_750]
    selected = aggregated[
        (aggregated["k"] == 10)
        & aggregated["vector_count"].isin(selected_counts)
        & aggregated["backend"].isin(["openmp", "blas"])
    ].copy()

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    for axis, metric in zip(axes.ravel(), EXPECTED_METRICS, strict=False):
        metric_data = selected[selected["metric"] == metric]
        for (backend, vector_count), group in metric_data.groupby(
            ["backend", "vector_count"]
        ):
            group = group.sort_values("effective_thread_count")
            label = (
                ("OpenMP" if backend == "openmp" else "SGEMV")
                + f", {int(vector_count):,} vectors"
            )
            axis.plot(
                group["effective_thread_count"],
                group["efficiency_median_percent"],
                marker="o",
                label=label,
            )
        axis.set_title(metric.upper())
        axis.set_xlabel("Backend threads")
        axis.set_ylabel("Parallel efficiency (%)")
        axis.set_xticks([1, 2, 4, 6])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle("Vector physical-core efficiency, k=10")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_memory(aggregated: pd.DataFrame, output: Path) -> None:
    memory = (
        aggregated.groupby(
            ["replication_factor", "vector_count"],
            as_index=False,
        )["estimated_total_mib"]
        .median()
        .sort_values("vector_count")
    )
    plt.figure(figsize=(9, 6))
    plt.plot(
        memory["vector_count"],
        memory["estimated_total_mib"],
        marker="o",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Database vectors")
    plt.ylabel("Estimated native input and output memory (MiB)")
    plt.title("Vector single-query memory growth")
    plt.grid(True, which="both", alpha=0.3)
    save_figure(output)


def create_compact_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    selected = aggregated[aggregated["k"] == 10].copy()
    rows = []

    for (metric, vector_count), group in selected.groupby(
        ["metric", "vector_count"]
    ):
        serial = group[group["backend"] == "serial"].iloc[0]
        openmp6 = group[
            (group["backend"] == "openmp")
            & (group["thread_count"] == 6)
        ].iloc[0]
        blas6 = group[
            (group["backend"] == "blas")
            & (group["openblas_thread_count"] == 6)
        ].iloc[0]

        variants = {
            "Fortran serial": float(
                serial["latency_median_across_queries_ms"]
            ),
            "Fortran OpenMP-6": float(
                openmp6["latency_median_across_queries_ms"]
            ),
            "OpenBLAS SGEMV-6": float(
                blas6["latency_median_across_queries_ms"]
            ),
        }

        rows.append(
            {
                "Metric": metric.upper(),
                "Vectors": int(vector_count),
                "Serial median (ms)": variants["Fortran serial"],
                "OpenMP-6 median (ms)": variants["Fortran OpenMP-6"],
                "SGEMV-6 median (ms)": variants["OpenBLAS SGEMV-6"],
                "OpenMP-6 speedup": float(
                    openmp6["speedup_median_across_queries"]
                ),
                "SGEMV-6 speedup": float(
                    blas6["speedup_median_across_queries"]
                ),
                "Best backend": min(variants, key=variants.get),
                "Estimated memory (MiB)": float(
                    serial["estimated_total_mib"]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["Metric", "Vectors"])


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
    validate_schema(raw)
    summary = summarize_configurations(raw)
    aggregated = aggregate_queries(summary)
    completeness = verify_completeness(raw, summary)

    if not completeness["is_complete"] and not arguments.allow_incomplete:
        raise RuntimeError(
            "Vector single-query benchmark is incomplete: "
            f"{completeness['unique_configurations']}/"
            f"{completeness['expected_configurations']} configurations"
        )
    if not completeness["execution_variants_match"]:
        raise RuntimeError("Execution variants do not match the matrix design")
    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate raw run rows were detected")
    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError(
            "At least one configuration has an incorrect repetition count"
        )

    summary.to_csv(
        processed_dir / "vector_real_single_configuration_summary.csv",
        index=False,
    )
    aggregated.to_csv(
        processed_dir / "vector_real_single_aggregated_summary.csv",
        index=False,
    )
    with (
        processed_dir / "vector_real_single_completeness_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    compact = create_compact_table(aggregated)
    compact.to_csv(
        tables_dir / "vector_real_single_table.csv",
        index=False,
    )
    with (tables_dir / "vector_real_single_table.md").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("# Vector Single-Query CPU Benchmark\n\n")
        file.write(
            "Values are medians across four production-derived queries at "
            "k=10. Timings include native scoring, exact O(Nk) top-k, "
            "binding, and output creation.\n\n"
        )
        file.write(dataframe_to_markdown(compact))

    detailed_columns = [
        "metric",
        "backend",
        "query_index",
        "vector_count",
        "replication_factor",
        "dimension",
        "k",
        "thread_count",
        "openblas_thread_count",
        "repetition_count",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_cv",
        "vectors_scored_per_second_median",
        "estimated_gflops_median",
        "speedup_vs_serial",
        "parallel_efficiency_percent",
        "estimated_total_mib",
    ]
    summary[detailed_columns].to_csv(
        tables_dir / "vector_real_single_detailed_table.csv",
        index=False,
    )

    plot_latency(
        aggregated,
        figures_dir / "vector_single_latency.png",
    )
    plot_speedup(
        aggregated,
        figures_dir / "vector_single_speedup.png",
    )
    plot_gflops(
        aggregated,
        figures_dir / "vector_single_gflops.png",
    )
    plot_k_sensitivity(
        aggregated,
        figures_dir / "vector_single_k_sensitivity.png",
    )
    plot_efficiency(
        aggregated,
        figures_dir / "vector_single_efficiency.png",
    )
    plot_memory(
        aggregated,
        figures_dir / "vector_single_memory.png",
    )

    print("Vector single-query benchmark analysis")
    print("=" * 72)
    print(
        "Configurations: "
        f"{completeness['unique_configurations']}/"
        f"{completeness['expected_configurations']}"
    )
    print(f"Raw rows: {completeness['raw_measurement_rows']:,}")
    print(f"Metrics: {completeness['metrics']}")
    print(f"Queries: {completeness['query_indices']}")
    print(f"Replication factors: {completeness['replication_factors']}")
    print(f"k values: {completeness['k_values']}")
    print(f"Processed data: {processed_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print("=" * 72)
    print("Vector single-query benchmark analysis: PASS")


if __name__ == "__main__":
    main()
