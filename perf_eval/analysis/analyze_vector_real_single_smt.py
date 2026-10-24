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
    / "perf_eval/results/raw/vector_real_single_raw.csv"
)
DEFAULT_SMT_RAW = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/vector_real_single_smt_raw.csv"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "perf_eval/results"

EXPECTED_SMT_CONFIGURATIONS = 432
EXPECTED_METRICS = ["ip", "l2"]
EXPECTED_QUERIES = [0, 67, 208, 474]
EXPECTED_REPLICATION_FACTORS = [10, 100, 250]
EXPECTED_K_VALUES = [1, 10, 50]
EXPECTED_SMT_THREADS = [8, 10, 12]
COMPARISON_THREADS = [6, 8, 10, 12]
BACKENDS = ["openmp", "blas"]

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

METHOD_BASELINE_KEYS = BASELINE_KEYS + ["backend"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze vector single-query SMT sensitivity by combining "
            "physical-core six-thread baselines with 8/10/12-thread runs."
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


def p25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def p75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def p95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def validate_schema(data: pd.DataFrame, label: str) -> None:
    required = set(CONFIG_KEYS) | {
        "package_version",
        "run_index",
        "elapsed_ns",
        "vectors_scored_per_second",
        "estimated_gflops",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{label} CSV is missing columns: "
            + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError(f"{label} CSV contains no rows")

    numeric = [
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
        "vectors_scored_per_second",
        "estimated_gflops",
        "estimated_input_bytes",
        "estimated_output_bytes",
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
    for suffix in ("p25", "median", "mean", "p75", "p95", "std"):
        summary[f"latency_{suffix}_ms"] = (
            summary[f"latency_{suffix}_ns"] / 1_000_000.0
        )
    summary["estimated_total_mib"] = (
        summary["estimated_input_bytes"]
        + summary["estimated_output_bytes"]
    ) / (1024.0**2)
    return summary


def select_physical_baselines(
    physical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = physical[
        physical["replication_factor"].isin(EXPECTED_REPLICATION_FACTORS)
        & physical["metric"].isin(EXPECTED_METRICS)
        & physical["query_index"].isin(EXPECTED_QUERIES)
        & physical["k"].isin(EXPECTED_K_VALUES)
    ].copy()

    serial = selected[
        (selected["backend"] == "serial")
        & (selected["thread_count"] == 1)
        & (selected["openblas_thread_count"] == 1)
    ].copy()

    six = selected[
        (
            (selected["backend"] == "openmp")
            & (selected["thread_count"] == 6)
            & (selected["openblas_thread_count"] == 1)
        )
        | (
            (selected["backend"] == "blas")
            & (selected["thread_count"] == 1)
            & (selected["openblas_thread_count"] == 6)
        )
    ].copy()

    expected_workloads = (
        len(EXPECTED_METRICS)
        * len(EXPECTED_QUERIES)
        * len(EXPECTED_REPLICATION_FACTORS)
        * len(EXPECTED_K_VALUES)
    )
    if len(serial) != expected_workloads:
        raise ValueError("Physical CSV lacks required serial baselines")
    if len(six) != expected_workloads * len(BACKENDS):
        raise ValueError("Physical CSV lacks required six-thread baselines")

    return serial, six


def add_metrics(
    combined: pd.DataFrame,
    serial: pd.DataFrame,
    six: pd.DataFrame,
) -> pd.DataFrame:
    serial_lookup = serial[
        BASELINE_KEYS + ["latency_median_ns"]
    ].rename(
        columns={"latency_median_ns": "serial_latency_median_ns"}
    )
    six_lookup = six[
        METHOD_BASELINE_KEYS + ["latency_median_ns"]
    ].rename(
        columns={"latency_median_ns": "six_thread_latency_median_ns"}
    )

    result = combined.merge(
        serial_lookup,
        on=BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    ).merge(
        six_lookup,
        on=METHOD_BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    )

    required = ["serial_latency_median_ns", "six_thread_latency_median_ns"]
    if result[required].isna().any().any():
        raise ValueError("At least one SMT row lacks a matching baseline")

    result["effective_thread_count"] = np.where(
        result["backend"] == "blas",
        result["openblas_thread_count"],
        result["thread_count"],
    )
    result["speedup_vs_serial"] = (
        result["serial_latency_median_ns"]
        / result["latency_median_ns"]
    )
    result["parallel_efficiency_percent"] = (
        result["speedup_vs_serial"]
        / result["effective_thread_count"]
        * 100.0
    )
    result["gain_vs_six_threads"] = (
        result["six_thread_latency_median_ns"]
        / result["latency_median_ns"]
    )
    result["latency_change_vs_six_percent"] = (
        result["latency_median_ns"]
        / result["six_thread_latency_median_ns"]
        - 1.0
    ) * 100.0
    return result


def aggregate_queries(combined: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "metric",
        "dimension",
        "k",
        "replication_factor",
        "vector_count",
        "backend",
        "thread_count",
        "openblas_thread_count",
        "effective_thread_count",
    ]
    return (
        combined.groupby(keys, as_index=False)
        .agg(
            query_count=("query_index", "nunique"),
            latency_median_across_queries_ms=(
                "latency_median_ms",
                "median",
            ),
            latency_p25_across_queries_ms=(
                "latency_median_ms",
                p25,
            ),
            latency_p75_across_queries_ms=(
                "latency_median_ms",
                p75,
            ),
            latency_p95_median_across_queries_ms=(
                "latency_p95_ms",
                "median",
            ),
            latency_cv_median_across_queries=("latency_cv", "median"),
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
            efficiency_median_percent=(
                "parallel_efficiency_percent",
                "median",
            ),
            gain_vs_six_median=("gain_vs_six_threads", "median"),
            gain_vs_six_p25=("gain_vs_six_threads", p25),
            gain_vs_six_p75=("gain_vs_six_threads", p75),
            estimated_total_mib=("estimated_total_mib", "median"),
        )
        .sort_values(
            ["metric", "k", "vector_count", "backend", "effective_thread_count"]
        )
    )


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
        smt_raw[["backend", "thread_count", "openblas_thread_count"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_variants = {
        *(('openmp', thread, 1) for thread in EXPECTED_SMT_THREADS),
        *(('blas', 1, thread) for thread in EXPECTED_SMT_THREADS),
    }

    return {
        "raw_measurement_rows": int(len(smt_raw)),
        "unique_smt_configurations": int(len(smt_summary)),
        "expected_smt_configurations": EXPECTED_SMT_CONFIGURATIONS,
        "is_complete": len(smt_summary) == EXPECTED_SMT_CONFIGURATIONS,
        "metrics": sorted(smt_raw["metric"].astype(str).unique().tolist()),
        "query_indices": sorted(
            int(value) for value in smt_raw["query_index"].unique()
        ),
        "replication_factors": sorted(
            int(value) for value in smt_raw["replication_factor"].unique()
        ),
        "k_values": sorted(int(value) for value in smt_raw["k"].unique()),
        "execution_variants_match": actual_variants == expected_variants,
        "duplicate_run_rows": duplicate_rows,
        "repetition_mismatch_configurations": int(len(mismatches)),
        "package_versions": sorted(
            smt_raw["package_version"].astype(str).unique().tolist()
        ),
    }


def backend_label(backend: str) -> str:
    return "Fortran OpenMP" if backend == "openmp" else "OpenBLAS SGEMV"


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def subplot_by_metric_and_scale(
    aggregated: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    output: Path,
    reference_one: bool = False,
) -> None:
    selected = aggregated[aggregated["k"] == 10].copy()
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    groups = selected.groupby(["metric", "vector_count"])

    for axis, ((metric, vector_count), group) in zip(
        axes.ravel(), groups, strict=False
    ):
        for backend, backend_group in group.groupby("backend"):
            backend_group = backend_group.sort_values("effective_thread_count")
            axis.plot(
                backend_group["effective_thread_count"],
                backend_group[value_column],
                marker="o",
                label=backend_label(backend),
            )
        if reference_one:
            axis.axhline(1.0, linestyle="--", label="No gain over 6")
        axis.set_title(f"{metric.upper()}, {int(vector_count):,} vectors")
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


def create_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "k",
        "replication_factor",
        "vector_count",
        "backend",
        "effective_thread_count",
        "latency_median_across_queries_ms",
        "latency_p95_median_across_queries_ms",
        "latency_cv_median_across_queries",
        "vectors_scored_per_second_median_across_queries",
        "gflops_median_across_queries",
        "speedup_median_across_queries",
        "efficiency_median_percent",
        "gain_vs_six_median",
        "estimated_total_mib",
    ]
    return aggregated[columns].sort_values(
        ["metric", "k", "vector_count", "backend", "effective_thread_count"]
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
    validate_schema(physical_raw, "Physical-core")
    validate_schema(smt_raw, "SMT")

    physical_summary = summarize(physical_raw)
    smt_summary = summarize(smt_raw)
    serial, six = select_physical_baselines(physical_summary)

    combined = pd.concat([six, smt_summary], ignore_index=True)
    combined = add_metrics(combined, serial, six)
    combined = combined.sort_values(
        ["metric", "k", "vector_count", "backend", "effective_thread_count"]
    )
    aggregated = aggregate_queries(combined)
    completeness = verify_completeness(smt_raw, smt_summary)

    if not completeness["is_complete"]:
        raise RuntimeError(
            "Vector single-query SMT benchmark is incomplete: "
            f"{completeness['unique_smt_configurations']}/"
            f"{completeness['expected_smt_configurations']} configurations"
        )
    if not completeness["execution_variants_match"]:
        raise RuntimeError("SMT execution variants do not match the design")
    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate SMT run rows were detected")
    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError("At least one SMT repetition count is incorrect")

    combined.to_csv(
        processed_dir / "vector_real_single_smt_configuration_summary.csv",
        index=False,
    )
    aggregated.to_csv(
        processed_dir / "vector_real_single_smt_aggregated_summary.csv",
        index=False,
    )
    with (
        processed_dir / "vector_real_single_smt_completeness_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    table = create_table(aggregated)
    table.to_csv(
        tables_dir / "vector_real_single_smt_table.csv",
        index=False,
    )
    with (tables_dir / "vector_real_single_smt_table.md").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("# Vector Single-Query SMT Sensitivity\n\n")
        file.write(
            "OpenMP/OpenBLAS six-thread baselines are reused from the "
            "physical-core matrix. Timings include scoring, serial O(Nk) "
            "top-k, binding, and output creation.\n\n"
        )
        file.write(dataframe_to_markdown(table))

    subplot_by_metric_and_scale(
        aggregated,
        value_column="latency_median_across_queries_ms",
        ylabel="Median complete-search latency (ms)",
        title="Vector single-query SMT latency, k=10",
        output=figures_dir / "vector_single_smt_latency.png",
    )
    subplot_by_metric_and_scale(
        aggregated,
        value_column="gain_vs_six_median",
        ylabel="Gain versus six threads",
        title="Incremental SMT gain over physical-core execution, k=10",
        output=figures_dir / "vector_single_smt_gain_vs_six.png",
        reference_one=True,
    )
    subplot_by_metric_and_scale(
        aggregated,
        value_column="speedup_median_across_queries",
        ylabel="Speedup versus Fortran serial",
        title="Vector single-query SMT speedup, k=10",
        output=figures_dir / "vector_single_smt_speedup.png",
    )
    subplot_by_metric_and_scale(
        aggregated,
        value_column="efficiency_median_percent",
        ylabel="Parallel efficiency (%)",
        title="Vector single-query SMT efficiency, k=10",
        output=figures_dir / "vector_single_smt_efficiency.png",
    )
    subplot_by_metric_and_scale(
        aggregated,
        value_column="latency_cv_median_across_queries",
        ylabel="Median latency coefficient of variation",
        title="Vector single-query SMT variability, k=10",
        output=figures_dir / "vector_single_smt_variability.png",
    )

    print("Vector single-query SMT analysis")
    print("=" * 72)
    print(
        "SMT configurations: "
        f"{completeness['unique_smt_configurations']}/"
        f"{completeness['expected_smt_configurations']}"
    )
    print(f"Raw SMT rows: {completeness['raw_measurement_rows']:,}")
    print(f"Metrics: {completeness['metrics']}")
    print(f"Queries: {completeness['query_indices']}")
    print(f"Replication factors: {completeness['replication_factors']}")
    print(f"k values: {completeness['k_values']}")
    print("Combined comparison threads: [6, 8, 10, 12]")
    print(f"Processed data: {processed_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print("=" * 72)
    print("Vector single-query SMT analysis: PASS")


if __name__ == "__main__":
    main()
