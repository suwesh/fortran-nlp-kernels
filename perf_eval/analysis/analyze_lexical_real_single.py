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
    / "perf_eval/results/raw/lexical_real_single_raw.csv"
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
    "minimum_document_shingles",
    "median_document_shingles",
    "mean_document_shingles",
    "maximum_document_shingles",
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

WORKLOAD_KEYS = [
    "corpus_fraction",
    "replication_factor",
    "document_count",
    "corpus_shingles",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the real-corpus Fortran lexical single-query "
            "benchmark and generate processed tables and figures."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--expected-configurations",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate partial outputs even if fewer configurations exist.",
    )
    return parser.parse_args()


def validate_raw_schema(data: pd.DataFrame) -> None:
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
            "Raw CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if data.empty:
        raise ValueError("Raw CSV contains no measurements")

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
        raise ValueError("elapsed_ns must be positive")


def percentile_25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def percentile_75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def percentile_95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


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

    summary["estimated_total_bytes"] = (
        summary["estimated_input_bytes"]
        + summary["estimated_output_bytes"]
    )
    summary["estimated_input_mib"] = (
        summary["estimated_input_bytes"] / (1024.0**2)
    )
    summary["estimated_output_mib"] = (
        summary["estimated_output_bytes"] / (1024.0**2)
    )
    summary["estimated_total_mib"] = (
        summary["estimated_total_bytes"] / (1024.0**2)
    )

    return add_speedup_and_efficiency(summary)


def add_speedup_and_efficiency(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    serial = summary[
        (summary["backend"] == "serial")
        & (summary["thread_count"] == 1)
    ][BASELINE_KEYS + ["latency_median_ns"]].copy()

    serial = serial.rename(
        columns={"latency_median_ns": "serial_latency_median_ns"}
    )

    if serial.duplicated(BASELINE_KEYS).any():
        raise ValueError("Multiple serial baselines exist for one workload")

    merged = summary.merge(
        serial,
        on=BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    )

    if merged["serial_latency_median_ns"].isna().any():
        raise ValueError("At least one configuration lacks a serial baseline")

    merged["speedup_vs_serial"] = (
        merged["serial_latency_median_ns"]
        / merged["latency_median_ns"]
    )
    merged["parallel_efficiency"] = np.where(
        merged["backend"] == "openmp",
        merged["speedup_vs_serial"] / merged["thread_count"],
        1.0,
    )
    merged["parallel_efficiency_percent"] = (
        merged["parallel_efficiency"] * 100.0
    )

    return merged


def aggregate_across_candidates(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    grouping = WORKLOAD_KEYS + ["backend", "thread_count"]
    aggregated = (
        summary.groupby(grouping, as_index=False)
        .agg(
            candidate_count=("candidate_id", "nunique"),
            candidate_shingles_min=("candidate_shingles", "min"),
            candidate_shingles_median=("candidate_shingles", "median"),
            candidate_shingles_max=("candidate_shingles", "max"),
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
            documents_per_second_median_across_candidates=(
                "documents_per_second_median",
                "median",
            ),
            corpus_shingles_per_second_median_across_candidates=(
                "corpus_shingles_per_second_median",
                "median",
            ),
            speedup_median_across_candidates=(
                "speedup_vs_serial",
                "median",
            ),
            speedup_p25_across_candidates=(
                "speedup_vs_serial",
                percentile_25,
            ),
            speedup_p75_across_candidates=(
                "speedup_vs_serial",
                percentile_75,
            ),
            efficiency_median_percent=(
                "parallel_efficiency_percent",
                "median",
            ),
            efficiency_p25_percent=(
                "parallel_efficiency_percent",
                percentile_25,
            ),
            efficiency_p75_percent=(
                "parallel_efficiency_percent",
                percentile_75,
            ),
            estimated_input_mib=("estimated_input_mib", "median"),
            estimated_output_mib=("estimated_output_mib", "median"),
            estimated_total_mib=("estimated_total_mib", "median"),
        )
    )
    return aggregated.sort_values(
        ["document_count", "backend", "thread_count"]
    )


def configuration_label(backend: str, thread_count: int) -> str:
    if backend == "serial":
        return "Fortran serial"
    return f"OpenMP-{int(thread_count)}"


def verify_completeness(
    data: pd.DataFrame,
    configuration_summary: pd.DataFrame,
    expected_configurations: int,
) -> dict:
    unique_configurations = len(configuration_summary)
    duplicate_run_rows = int(
        data.duplicated(CONFIG_KEYS + ["run_index"]).sum()
    )
    repetition_mismatches = configuration_summary[
        configuration_summary["repetition_count"]
        != configuration_summary["measured_runs"]
    ]

    candidate_count = int(data["candidate_id"].nunique())
    workload_count = int(
        data[WORKLOAD_KEYS].drop_duplicates().shape[0]
    )
    variant_count = int(
        data[["backend", "thread_count"]]
        .drop_duplicates()
        .shape[0]
    )

    return {
        "raw_measurement_rows": int(len(data)),
        "unique_configurations": unique_configurations,
        "expected_configurations": expected_configurations,
        "is_complete": unique_configurations == expected_configurations,
        "candidate_count": candidate_count,
        "workload_scale_count": workload_count,
        "execution_variant_count": variant_count,
        "duplicate_run_rows": duplicate_run_rows,
        "repetition_mismatch_configurations": int(
            len(repetition_mismatches)
        ),
        "package_versions": sorted(
            data["package_version"].astype(str).unique().tolist()
        ),
    }


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_latency(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    for (backend, threads), group in aggregated.groupby(
        ["backend", "thread_count"]
    ):
        group = group.sort_values("document_count")
        label = configuration_label(backend, threads)
        x = group["document_count"].to_numpy()
        y = group["latency_median_across_candidates_ms"].to_numpy()
        lower = group["latency_p25_across_candidates_ms"].to_numpy()
        upper = group["latency_p75_across_candidates_ms"].to_numpy()
        plt.plot(x, y, marker="o", label=label)
        plt.fill_between(x, lower, upper, alpha=0.15)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Document count")
    plt.ylabel("Median latency across candidates (ms)")
    plt.title("Exact Jaccard latency versus corpus size")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save_figure(output_path)


def plot_speedup(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    openmp = aggregated[aggregated["backend"] == "openmp"]

    for threads, group in openmp.groupby("thread_count"):
        group = group.sort_values("document_count")
        x = group["document_count"].to_numpy()
        y = group["speedup_median_across_candidates"].to_numpy()
        lower = group["speedup_p25_across_candidates"].to_numpy()
        upper = group["speedup_p75_across_candidates"].to_numpy()
        plt.plot(x, y, marker="o", label=f"OpenMP-{int(threads)}")
        plt.fill_between(x, lower, upper, alpha=0.15)

    plt.axhline(1.0, linestyle="--", label="Serial crossover")
    plt.xscale("log")
    plt.xlabel("Document count")
    plt.ylabel("Speedup versus dedicated serial backend")
    plt.title("OpenMP speedup versus corpus size")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save_figure(output_path)


def nearest_workload_sizes(
    available: np.ndarray,
    targets: list[int],
) -> list[int]:
    available = np.asarray(sorted(set(int(value) for value in available)))
    selected = []
    for target in targets:
        index = int(np.argmin(np.abs(available - target)))
        selected.append(int(available[index]))
    return list(dict.fromkeys(selected))


def plot_efficiency(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    openmp = aggregated[aggregated["backend"] == "openmp"]
    selected_sizes = nearest_workload_sizes(
        openmp["document_count"].unique(),
        [372, 3_720, 37_200, 372_000],
    )

    for size in selected_sizes:
        group = openmp[openmp["document_count"] == size].sort_values(
            "thread_count"
        )
        plt.plot(
            group["thread_count"],
            group["efficiency_median_percent"],
            marker="o",
            label=f"{size:,} documents",
        )

    plt.xlabel("OpenMP thread count")
    plt.ylabel("Parallel efficiency (%)")
    plt.title("Physical-core parallel efficiency")
    plt.xticks(sorted(openmp["thread_count"].unique()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(output_path)


def plot_throughput(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    for (backend, threads), group in aggregated.groupby(
        ["backend", "thread_count"]
    ):
        group = group.sort_values("corpus_shingles")
        plt.plot(
            group["corpus_shingles"],
            group[
                "corpus_shingles_per_second_median_across_candidates"
            ],
            marker="o",
            label=configuration_label(backend, threads),
        )

    plt.xscale("log")
    plt.xlabel("Total corpus shingles")
    plt.ylabel("Corpus shingles processed per second")
    plt.title("Exact Jaccard throughput versus corpus work")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save_figure(output_path)


def plot_memory(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    memory = (
        aggregated.groupby(WORKLOAD_KEYS, as_index=False)
        .agg(
            input_mib=("estimated_input_mib", "median"),
            output_mib=("estimated_output_mib", "median"),
            total_mib=("estimated_total_mib", "median"),
        )
        .sort_values("document_count")
    )

    plt.figure(figsize=(9, 6))
    plt.plot(
        memory["document_count"],
        memory["input_mib"],
        marker="o",
        label="Packed inputs",
    )
    plt.plot(
        memory["document_count"],
        memory["output_mib"],
        marker="o",
        label="Native outputs",
    )
    plt.plot(
        memory["document_count"],
        memory["total_mib"],
        marker="o",
        label="Estimated total",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Document count")
    plt.ylabel("Estimated native memory (MiB)")
    plt.title("Estimated native memory growth")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save_figure(output_path)


def plot_candidate_sensitivity(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    available_sizes = summary["document_count"].unique()
    selected_sizes = nearest_workload_sizes(
        available_sizes,
        [372, 3_720, 37_200, 372_000],
    )

    selected = summary[
        summary["document_count"].isin(selected_sizes)
        & (
            ((summary["backend"] == "serial") & (summary["thread_count"] == 1))
            | ((summary["backend"] == "openmp") & (summary["thread_count"] == 6))
        )
    ].copy()

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    for axis, size in zip(axes.ravel(), selected_sizes, strict=False):
        workload = selected[selected["document_count"] == size]
        for (backend, threads), group in workload.groupby(
            ["backend", "thread_count"]
        ):
            group = group.sort_values("candidate_shingles")
            axis.plot(
                group["candidate_shingles"],
                group["latency_median_ms"],
                marker="o",
                label=configuration_label(backend, threads),
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{size:,} documents")
        axis.set_xlabel("Candidate shingles")
        axis.set_ylabel("Median latency (ms)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.suptitle("Candidate-size sensitivity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_readme_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    latency = aggregated.pivot_table(
        index=["document_count", "corpus_shingles"],
        columns=["backend", "thread_count"],
        values="latency_median_across_candidates_ms",
        aggfunc="first",
    )
    speedup = aggregated.pivot_table(
        index=["document_count", "corpus_shingles"],
        columns=["backend", "thread_count"],
        values="speedup_median_across_candidates",
        aggfunc="first",
    )
    memory = aggregated.groupby(
        ["document_count", "corpus_shingles"],
        as_index=True,
    )["estimated_total_mib"].median()

    rows = []
    for index in latency.index:
        serial_ms = float(latency.loc[index, ("serial", 1)])
        omp2_ms = float(latency.loc[index, ("openmp", 2)])
        omp4_ms = float(latency.loc[index, ("openmp", 4)])
        omp6_ms = float(latency.loc[index, ("openmp", 6)])

        variants = {
            "Serial": serial_ms,
            "OpenMP-2": omp2_ms,
            "OpenMP-4": omp4_ms,
            "OpenMP-6": omp6_ms,
        }
        best_backend = min(variants, key=variants.get)
        best_speedup = max(
            1.0,
            float(speedup.loc[index, ("openmp", 2)]),
            float(speedup.loc[index, ("openmp", 4)]),
            float(speedup.loc[index, ("openmp", 6)]),
        )

        rows.append(
            {
                "Documents": int(index[0]),
                "Corpus shingles": int(index[1]),
                "Serial median (ms)": serial_ms,
                "OpenMP-2 median (ms)": omp2_ms,
                "OpenMP-4 median (ms)": omp4_ms,
                "OpenMP-6 median (ms)": omp6_ms,
                "Best backend": best_backend,
                "Best speedup": best_speedup,
                "Estimated memory (MiB)": float(memory.loc[index]),
            }
        )

    return pd.DataFrame(rows).sort_values("Documents")


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

    data = pd.read_csv(raw_path, engine="python")
    validate_raw_schema(data)

    configuration_summary = summarize_configurations(data)
    aggregated_summary = aggregate_across_candidates(configuration_summary)
    completeness = verify_completeness(
        data,
        configuration_summary,
        arguments.expected_configurations,
    )

    if not completeness["is_complete"] and not arguments.allow_incomplete:
        raise RuntimeError(
            "Benchmark is incomplete: "
            f"{completeness['unique_configurations']}/"
            f"{completeness['expected_configurations']} configurations. "
            "Use --allow-incomplete only for exploratory analysis."
        )

    if completeness["duplicate_run_rows"]:
        raise RuntimeError("Duplicate raw run rows were detected")

    if completeness["repetition_mismatch_configurations"]:
        raise RuntimeError(
            "At least one configuration has fewer or more repetitions "
            "than its measured_runs value"
        )

    configuration_summary.to_csv(
        processed_dir / "lexical_real_single_configuration_summary.csv",
        index=False,
    )
    aggregated_summary.to_csv(
        processed_dir / "lexical_real_single_aggregated_summary.csv",
        index=False,
    )

    with (
        processed_dir / "lexical_real_single_completeness_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(completeness, file, indent=2)
        file.write("\n")

    detailed_columns = [
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
        "repetition_count",
        "latency_median_ms",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_p25_ms",
        "latency_p75_ms",
        "latency_p95_ms",
        "latency_cv",
        "documents_per_second_median",
        "corpus_shingles_per_second_median",
        "speedup_vs_serial",
        "parallel_efficiency_percent",
        "estimated_input_mib",
        "estimated_output_mib",
        "estimated_total_mib",
    ]
    configuration_summary[detailed_columns].to_csv(
        tables_dir / "lexical_cpu_detailed_table.csv",
        index=False,
    )

    readme_table = create_readme_table(aggregated_summary)
    readme_table.to_csv(
        tables_dir / "lexical_cpu_readme_table.csv",
        index=False,
    )
    with (tables_dir / "lexical_cpu_readme_table.md").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("# CPU Lexical Benchmark Summary\n\n")
        file.write(
            "Values are medians across the eight real candidates. "
            "Latency is measured through the native package binding.\n\n"
        )
        file.write(dataframe_to_markdown(readme_table))

    plot_latency(
        aggregated_summary,
        figures_dir / "lexical_latency_vs_documents.png",
    )
    plot_speedup(
        aggregated_summary,
        figures_dir / "lexical_speedup_vs_documents.png",
    )
    plot_efficiency(
        aggregated_summary,
        figures_dir / "lexical_efficiency_vs_threads.png",
    )
    plot_throughput(
        aggregated_summary,
        figures_dir / "lexical_throughput_vs_shingles.png",
    )
    plot_memory(
        aggregated_summary,
        figures_dir / "lexical_memory_vs_documents.png",
    )
    plot_candidate_sensitivity(
        configuration_summary,
        figures_dir / "lexical_candidate_size_sensitivity.png",
    )

    print("Lexical benchmark analysis")
    print("=" * 72)
    print(
        "Configurations: "
        f"{completeness['unique_configurations']}/"
        f"{completeness['expected_configurations']}"
    )
    print(f"Raw measurement rows: {completeness['raw_measurement_rows']:,}")
    print(f"Candidates: {completeness['candidate_count']}")
    print(f"Workload scales: {completeness['workload_scale_count']}")
    print(f"Execution variants: {completeness['execution_variant_count']}")
    print(f"Processed data: {processed_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print("=" * 72)
    print("Lexical benchmark analysis: PASS")


if __name__ == "__main__":
    main()
