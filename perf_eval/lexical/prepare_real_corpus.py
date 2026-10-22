from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
from typing import Any
import numpy as np
import fortran_nlp
from fortran_nlp.lexical.preprocessing import (
    pack_corpus,
)
from perf_eval.common.kb_source_loader import (
    configured_sources,
    lexical_builder_name,
    load_kb_config,
    load_kb_metadata,
)
from perf_eval.lexical.corpus_builder import (
    LexicalDocument,
    combine_lexical_documents,
    prepare_lexical_documents,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "lexical_cpu_matrix.local.yaml"
)
REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2]
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the production-derived lexical corpus "
            "for CPU benchmarking."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing packed corpus, manifest, "
            "and statistics files."
        ),
    )
    return parser.parse_args()

def resolve_output_path(
    configured_path: str,
) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()

def ensure_output_is_available(
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

def load_logical_documents(
    config_path: Path,
) -> tuple[
    list[LexicalDocument],
    dict[str, dict[str, int | str]],
]:
    documents_by_source: dict[
        str,
        list[LexicalDocument],
    ] = {}
    source_statistics: dict[
        str,
        dict[str, int | str],
    ] = {}
    for source in configured_sources(config_path):
        metadata = load_kb_metadata(
            source,
            config_path,
        )
        builder_name = lexical_builder_name(
            source,
            config_path,
        )
        documents = prepare_lexical_documents(
            source=source,
            metadata=metadata,
            builder_name=builder_name,
        )
        if not documents:
            raise ValueError(
                f"Source '{source}' produced no "
                "lexical documents"
            )
        documents_by_source[source] = documents
        source_statistics[source] = {
            "metadata_rows": len(metadata),
            "lexical_documents": len(documents),
            "lexical_builder": builder_name,
        }
    combined_documents = combine_lexical_documents(
        documents_by_source
    )
    if not combined_documents:
        raise ValueError(
            "Combined corpus contains no lexical documents"
        )
    return combined_documents, source_statistics

def validate_documents(
    documents: list[LexicalDocument],
) -> None:
    for document_index, document in enumerate(
        documents
    ):
        if not document.text.strip():
            raise ValueError(
                f"Document {document_index} from source "
                f"'{document.source}' contains empty text"
            )
        if not document.document_id:
            raise ValueError(
                f"Document {document_index} from source "
                f"'{document.source}' has an empty ID"
            )

def write_document_manifest(
    path: Path,
    documents: list[LexicalDocument],
) -> None:
    """
    Save the logical corpus for public JaccardIndex benchmarks.
    This file contains source-derived text and must remain outside
    version control unless it has been reviewed and sanitized.
    """
    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for positional_id, document in enumerate(
            documents
        ):
            row = {
                "positional_id": positional_id,
                "source": document.source,
                "document_id": document.document_id,
                "text": document.text,
            }
            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )
            output_file.write("\n")

def validate_packed_corpus(
    *,
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
    document_count: int,
) -> np.ndarray:
    if corpus_shingles.dtype != np.int64:
        raise AssertionError(
            "corpus_shingles must use int64"
        )
    if document_offsets.dtype != np.int64:
        raise AssertionError(
            "document_offsets must use int64"
        )
    if corpus_shingles.ndim != 1:
        raise AssertionError(
            "corpus_shingles must be one-dimensional"
        )
    if document_offsets.ndim != 1:
        raise AssertionError(
            "document_offsets must be one-dimensional"
        )
    if not corpus_shingles.flags.c_contiguous:
        raise AssertionError(
            "corpus_shingles must be C-contiguous"
        )
    if not document_offsets.flags.c_contiguous:
        raise AssertionError(
            "document_offsets must be C-contiguous"
        )
    if document_offsets.size != document_count + 1:
        raise AssertionError(
            "document offset count is inconsistent"
        )
    if document_offsets[0] != 0:
        raise AssertionError(
            "document_offsets must begin with zero"
        )
    if document_offsets[-1] != corpus_shingles.size:
        raise AssertionError(
            "final document offset must equal "
            "corpus shingle count"
        )
    if not np.all(
        document_offsets[1:]
        >= document_offsets[:-1]
    ):
        raise AssertionError(
            "document_offsets must be nondecreasing"
        )
    return np.diff(document_offsets)

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return "unknown"

def calculate_statistics(
    *,
    documents: list[LexicalDocument],
    document_lengths: np.ndarray,
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
    source_statistics: dict[
        str,
        dict[str, int | str],
    ],
    shingle_size: int,
) -> dict[str, Any]:
    source_document_counts: dict[str, int] = {}
    for document in documents:
        source_document_counts[document.source] = (
            source_document_counts.get(
                document.source,
                0,
            )
            + 1
        )
    corpus_bytes = int(corpus_shingles.nbytes)
    offset_bytes = int(document_offsets.nbytes)
    # Each single-candidate native call returns:
    # intersections: int64[document_count]
    # unions:        int64[document_count]
    # scores:        float64[document_count]
    single_query_output_bytes = (
        len(documents) * 24
    )
    return {
        "schema_version": 1,
        "package_version": fortran_nlp.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "git_commit": get_git_commit(),
        "shingle_size": shingle_size,
        "document_count": len(documents),
        "total_corpus_shingles": int(
            corpus_shingles.size
        ),
        "minimum_document_shingles": int(
            np.min(document_lengths)
        ),
        "median_document_shingles": float(
            np.median(document_lengths)
        ),
        "mean_document_shingles": float(
            np.mean(document_lengths)
        ),
        "maximum_document_shingles": int(
            np.max(document_lengths)
        ),
        "document_shingle_standard_deviation": float(
            np.std(document_lengths)
        ),
        "zero_shingle_document_count": int(
            np.count_nonzero(
                document_lengths == 0
            )
        ),
        "source_document_counts": (
            source_document_counts
        ),
        "source_statistics": source_statistics,
        "memory_bytes": {
            "corpus_shingles": corpus_bytes,
            "document_offsets": offset_bytes,
            "packed_input_total": (
                corpus_bytes + offset_bytes
            ),
            "single_query_native_outputs": (
                single_query_output_bytes
            ),
            "single_query_native_working_set": (
                corpus_bytes
                + offset_bytes
                + single_query_output_bytes
            ),
        },
    }

def write_statistics(
    path: Path,
    statistics: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            statistics,
            output_file,
            indent=2,
            ensure_ascii=False,
        )
        output_file.write("\n")

def print_summary(
    *,
    packed_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    statistics: dict[str, Any],
) -> None:
    memory = statistics["memory_bytes"]
    print("Real lexical corpus preparation")
    print("=" * 64)
    print(
        "fortran-nlp-kernels: "
        f"{statistics['package_version']}"
    )
    print(
        f"Documents: {statistics['document_count']}"
    )
    print(
        "Corpus shingles: "
        f"{statistics['total_corpus_shingles']}"
    )
    print(
        "Document shingles min/median/mean/max: "
        f"{statistics['minimum_document_shingles']}/"
        f"{statistics['median_document_shingles']:.2f}/"
        f"{statistics['mean_document_shingles']:.2f}/"
        f"{statistics['maximum_document_shingles']}"
    )
    print(
        "Zero-shingle documents: "
        f"{statistics['zero_shingle_document_count']}"
    )
    print("\nDocuments by source")
    print("-" * 64)
    for source, count in (
        statistics["source_document_counts"].items()
    ):
        print(f"{source}: {count}")
    print("\nPacked memory")
    print("-" * 64)
    print(
        "Corpus shingles: "
        f"{memory['corpus_shingles']:,} bytes"
    )
    print(
        "Document offsets: "
        f"{memory['document_offsets']:,} bytes"
    )
    print(
        "Single-query native outputs: "
        f"{memory['single_query_native_outputs']:,} bytes"
    )
    print(
        "Estimated single-query native working set: "
        f"{memory['single_query_native_working_set']:,} bytes"
    )
    print("\nOutputs")
    print("-" * 64)
    print(f"Packed corpus: {packed_path}")
    print(f"Private manifest: {manifest_path}")
    print(f"Statistics: {statistics_path}")
    print("=" * 64)
    print("Real lexical corpus preparation: PASS")

def main() -> None:
    arguments = parse_arguments()
    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )
    config = load_kb_config(config_path)
    if "packed_corpus" not in config:
        raise ValueError(
            "Configuration is missing 'packed_corpus'"
        )
    packed_config = config["packed_corpus"]
    required_keys = (
        "output_path",
        "document_manifest_path",
        "statistics_path",
        "shingle_size",
    )
    for key in required_keys:
        if key not in packed_config:
            raise ValueError(
                "'packed_corpus' is missing "
                f"required key '{key}'"
            )
    packed_path = resolve_output_path(
        packed_config["output_path"]
    )
    manifest_path = resolve_output_path(
        packed_config[
            "document_manifest_path"
        ]
    )
    statistics_path = resolve_output_path(
        packed_config["statistics_path"]
    )
    for output_path in (
        packed_path,
        manifest_path,
        statistics_path,
    ):
        ensure_output_is_available(
            output_path,
            overwrite=arguments.overwrite,
        )
    shingle_size = int(
        packed_config["shingle_size"]
    )
    if shingle_size < 1:
        raise ValueError(
            "shingle_size must be at least 1"
        )
    (
        documents,
        source_statistics,
    ) = load_logical_documents(config_path)
    validate_documents(documents)
    corpus_texts = [
        document.text
        for document in documents
    ]
    (
        corpus_shingles,
        document_offsets,
    ) = pack_corpus(
        corpus_texts,
        shingle_size=shingle_size,
    )
    document_lengths = validate_packed_corpus(
        corpus_shingles=corpus_shingles,
        document_offsets=document_offsets,
        document_count=len(documents),
    )
    document_sources = np.array(
        [
            document.source
            for document in documents
        ],
        dtype=np.str_,
    )
    document_ids = np.array(
        [
            document.document_id
            for document in documents
        ],
        dtype=np.str_,
    )
    np.savez_compressed(
        packed_path,
        corpus_shingles=corpus_shingles,
        document_offsets=document_offsets,
        document_lengths=document_lengths,
        document_sources=document_sources,
        document_ids=document_ids,
        shingle_size=np.array(
            shingle_size,
            dtype=np.int64,
        ),
    )
    write_document_manifest(
        manifest_path,
        documents,
    )
    statistics = calculate_statistics(
        documents=documents,
        document_lengths=document_lengths,
        corpus_shingles=corpus_shingles,
        document_offsets=document_offsets,
        source_statistics=source_statistics,
        shingle_size=shingle_size,
    )
    write_statistics(
        statistics_path,
        statistics,
    )
    print_summary(
        packed_path=packed_path,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        statistics=statistics,
    )

if __name__ == "__main__":
    main()