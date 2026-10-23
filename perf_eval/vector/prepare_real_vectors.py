from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

import fortran_nlp

from perf_eval.common.kb_source_loader import (
    configured_sources,
    load_kb_config,
    load_kb_source,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "perf_eval/config/lexical_cpu_matrix.local.yaml"
)

DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "perf_eval/data/packed/vector_real_database.npz"
)

DEFAULT_STATISTICS_PATH = (
    REPOSITORY_ROOT
    / "perf_eval/metadata/vector_real_database_statistics.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine all configured real embedding sources into one "
            "validated float32 vector database for CPU benchmarking."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Override the packed vector output path. If omitted, the "
            "vector_database.output_path YAML value is used when present."
        ),
    )

    parser.add_argument(
        "--statistics",
        type=Path,
        default=None,
        help=(
            "Override the statistics output path. If omitted, the "
            "vector_database.statistics_path YAML value is used when present."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing packed and statistics files.",
    )

    return parser.parse_args()


def resolve_repository_path(
    configured_path: str | Path,
) -> Path:
    path = Path(configured_path).expanduser()

    if not path.is_absolute():
        path = REPOSITORY_ROOT / path

    return path.resolve()


def resolve_output_paths(
    *,
    config: dict[str, Any],
    output_override: Path | None,
    statistics_override: Path | None,
) -> tuple[Path, Path]:
    vector_config = config.get("vector_database", {})

    output_value: str | Path = (
        output_override
        if output_override is not None
        else vector_config.get(
            "output_path",
            DEFAULT_OUTPUT_PATH,
        )
    )

    statistics_value: str | Path = (
        statistics_override
        if statistics_override is not None
        else vector_config.get(
            "statistics_path",
            DEFAULT_STATISTICS_PATH,
        )
    )

    return (
        resolve_repository_path(output_value),
        resolve_repository_path(statistics_value),
    )


def ensure_output_available(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. "
            "Pass --overwrite to replace it."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return "unknown"


def load_and_validate_sources(
    config_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, Any]],
]:
    source_matrices: list[np.ndarray] = []
    source_labels: list[str] = []
    source_row_ids: list[int] = []
    source_statistics: dict[str, dict[str, Any]] = {}

    expected_dimension: int | None = None

    for source in configured_sources(config_path):
        metadata, embeddings = load_kb_source(
            source,
            config_path,
            mmap_mode="r",
        )

        if embeddings.dtype != np.float32:
            raise ValueError(
                f"Embeddings for source '{source}' must use float32; "
                f"received {embeddings.dtype}"
            )

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings for source '{source}' must be rank two; "
                f"received shape {embeddings.shape}"
            )

        if expected_dimension is None:
            expected_dimension = int(embeddings.shape[1])
        elif embeddings.shape[1] != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for source '{source}': "
                f"expected {expected_dimension}, received {embeddings.shape[1]}"
            )

        matrix = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Embeddings for source '{source}' contain NaN or infinity"
            )

        norms = np.linalg.norm(
            matrix,
            axis=1,
        )

        zero_norm_count = int(
            np.count_nonzero(norms == 0.0)
        )

        if zero_norm_count:
            raise ValueError(
                f"Embeddings for source '{source}' contain "
                f"{zero_norm_count} zero-norm vectors"
            )

        source_matrices.append(matrix)
        source_labels.extend(
            [source] * matrix.shape[0]
        )
        source_row_ids.extend(
            range(matrix.shape[0])
        )

        source_statistics[source] = {
            "metadata_rows": len(metadata),
            "vector_count": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]),
            "dtype": str(matrix.dtype),
            "c_contiguous": bool(matrix.flags.c_contiguous),
            "minimum_norm": float(norms.min()),
            "median_norm": float(np.median(norms)),
            "mean_norm": float(norms.mean()),
            "maximum_norm": float(norms.max()),
            "zero_norm_vectors": zero_norm_count,
        }

    if not source_matrices:
        raise ValueError(
            "No configured embedding sources were loaded"
        )

    database_vectors = np.ascontiguousarray(
        np.concatenate(
            source_matrices,
            axis=0,
        ),
        dtype=np.float32,
    )

    database_sources = np.asarray(
        source_labels,
        dtype=np.str_,
    )

    database_source_row_ids = np.asarray(
        source_row_ids,
        dtype=np.int64,
    )

    if database_sources.size != database_vectors.shape[0]:
        raise AssertionError(
            "Source-label count does not match vector count"
        )

    if database_source_row_ids.size != database_vectors.shape[0]:
        raise AssertionError(
            "Source-row-ID count does not match vector count"
        )

    return (
        database_vectors,
        database_sources,
        database_source_row_ids,
        source_statistics,
    )


def validate_combined_database(
    database_vectors: np.ndarray,
) -> np.ndarray:
    if database_vectors.dtype != np.float32:
        raise AssertionError(
            "Combined database must use float32"
        )

    if database_vectors.ndim != 2:
        raise AssertionError(
            "Combined database must be rank two"
        )

    if not database_vectors.flags.c_contiguous:
        raise AssertionError(
            "Combined database must be C-contiguous"
        )

    if not np.isfinite(database_vectors).all():
        raise AssertionError(
            "Combined database contains NaN or infinity"
        )

    norms = np.linalg.norm(
        database_vectors,
        axis=1,
    )

    if np.any(norms == 0.0):
        raise AssertionError(
            "Combined database contains zero-norm vectors"
        )

    return norms


def calculate_statistics(
    *,
    database_vectors: np.ndarray,
    norms: np.ndarray,
    source_statistics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    vector_count = int(database_vectors.shape[0])
    dimension = int(database_vectors.shape[1])
    database_bytes = int(database_vectors.nbytes)

    # Single-query native scoring produces one float32 score per vector.
    single_score_bytes = vector_count * 4

    # Public exact top-k returns float32 distances/scores and integer IDs.
    # The actual ID dtype should be confirmed during backend validation.
    return {
        "schema_version": 1,
        "package_version": fortran_nlp.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "git_commit": get_git_commit(),
        "vector_count": vector_count,
        "dimension": dimension,
        "dtype": str(database_vectors.dtype),
        "c_contiguous": bool(database_vectors.flags.c_contiguous),
        "finite": bool(np.isfinite(database_vectors).all()),
        "minimum_norm": float(norms.min()),
        "median_norm": float(np.median(norms)),
        "mean_norm": float(norms.mean()),
        "maximum_norm": float(norms.max()),
        "zero_norm_vectors": int(np.count_nonzero(norms == 0.0)),
        "maximum_absolute_norm_error_from_one": float(
            np.max(np.abs(norms - 1.0))
        ),
        "source_statistics": source_statistics,
        "memory_bytes": {
            "database_vectors": database_bytes,
            "single_query_vector": dimension * 4,
            "single_query_scores": single_score_bytes,
            "single_query_native_working_set": (
                database_bytes
                + dimension * 4
                + single_score_bytes
            ),
        },
        "metric_relationships": {
            "unit_normalized": True,
            "cosine_equals_inner_product": True,
            "squared_l2_equals_two_minus_two_inner_product": True,
        },
    }


def write_statistics(
    path: Path,
    statistics: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def print_summary(
    *,
    output_path: Path,
    statistics_path: Path,
    statistics: dict[str, Any],
) -> None:
    memory = statistics["memory_bytes"]

    print("Real vector database preparation")
    print("=" * 72)
    print(
        "fortran-nlp-kernels: "
        f"{statistics['package_version']}"
    )
    print(
        f"Vectors: {statistics['vector_count']:,}"
    )
    print(
        f"Dimension: {statistics['dimension']:,}"
    )
    print(f"dtype: {statistics['dtype']}")
    print(
        "Norm min/median/mean/max: "
        f"{statistics['minimum_norm']:.8f}/"
        f"{statistics['median_norm']:.8f}/"
        f"{statistics['mean_norm']:.8f}/"
        f"{statistics['maximum_norm']:.8f}"
    )
    print(
        "Maximum |norm - 1|: "
        f"{statistics['maximum_absolute_norm_error_from_one']:.8e}"
    )
    print(
        "Database memory: "
        f"{memory['database_vectors'] / (1024 ** 2):.3f} MiB"
    )
    print("-" * 72)
    print("Vectors by source")

    for source, source_data in (
        statistics["source_statistics"].items()
    ):
        print(
            f"{source}: "
            f"{source_data['vector_count']:,} vectors"
        )

    print("-" * 72)
    print(f"Packed database: {output_path}")
    print(f"Statistics: {statistics_path}")
    print("=" * 72)
    print("Real vector database preparation: PASS")


def main() -> None:
    arguments = parse_arguments()

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    config = load_kb_config(config_path)

    output_path, statistics_path = resolve_output_paths(
        config=config,
        output_override=arguments.output,
        statistics_override=arguments.statistics,
    )

    ensure_output_available(
        output_path,
        overwrite=arguments.overwrite,
    )
    ensure_output_available(
        statistics_path,
        overwrite=arguments.overwrite,
    )

    (
        database_vectors,
        database_sources,
        database_source_row_ids,
        source_statistics,
    ) = load_and_validate_sources(config_path)

    norms = validate_combined_database(
        database_vectors
    )

    np.savez_compressed(
        output_path,
        database_vectors=database_vectors,
        database_sources=database_sources,
        database_source_row_ids=database_source_row_ids,
        vector_dimension=np.asarray(
            database_vectors.shape[1],
            dtype=np.int64,
        ),
        vectors_are_unit_normalized=np.asarray(
            True,
            dtype=np.bool_,
        ),
    )

    statistics = calculate_statistics(
        database_vectors=database_vectors,
        norms=norms,
        source_statistics=source_statistics,
    )

    write_statistics(
        statistics_path,
        statistics,
    )

    print_summary(
        output_path=output_path,
        statistics_path=statistics_path,
        statistics=statistics,
    )


if __name__ == "__main__":
    main()
