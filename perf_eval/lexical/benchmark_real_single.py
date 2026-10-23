from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import yaml

import fortran_nlp
from fortran_nlp import _cpu
from fortran_nlp.lexical.preprocessing import (
    encode_text_shingles,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "perf_eval/config/lexical_cpu_matrix.local.yaml"
)

DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "perf_eval/results/raw/lexical_real_single_raw.csv"
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()

    if not path.is_absolute():
        path = REPOSITORY_ROOT / path

    return path.resolve()

def select_fractional_corpus(
    *,
    base_corpus: np.ndarray,
    base_offsets: np.ndarray,
    document_sources: np.ndarray,
    corpus_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < corpus_fraction <= 1.0:
        raise ValueError(
            "corpus-fraction must be in the interval (0, 1]"
        )
    if corpus_fraction == 1.0:
        return base_corpus, base_offsets
    selected_indices: list[int] = []
    for source in dict.fromkeys(
        document_sources.tolist()
    ):
        source_indices = np.flatnonzero(
            document_sources == source
        )
        selected_count = max(
            1,
            int(round(
                source_indices.size
                * corpus_fraction
            )),
        )
        # Evenly spaced deterministic selection within each source.
        selection_positions = np.linspace(
            0,
            source_indices.size - 1,
            num=selected_count,
            dtype=np.int64,
        )
        selected_indices.extend(
            source_indices[
                selection_positions
            ].tolist()
        )
    selected_indices = sorted(
        set(selected_indices)
    )
    selected_documents = []
    for document_index in selected_indices:
        start = int(base_offsets[document_index])
        end = int(base_offsets[document_index + 1])
        selected_documents.append(
            base_corpus[start:end]
        )
    selected_lengths = np.array(
        [
            document.size
            for document in selected_documents
        ],
        dtype=np.int64,
    )
    selected_offsets = np.empty(
        selected_lengths.size + 1,
        dtype=np.int64,
    )
    selected_offsets[0] = 0
    np.cumsum(
        selected_lengths,
        dtype=np.int64,
        out=selected_offsets[1:],
    )
    selected_corpus = np.ascontiguousarray(
        np.concatenate(selected_documents),
        dtype=np.int64,
    )
    return selected_corpus, selected_offsets

def load_inputs(
    config_path: Path,
    candidate_id: str,
    corpus_fraction: float,
    replication_factor: int,
):
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    packed_path = resolve_path(
        config["packed_corpus"]["output_path"]
    )

    candidate_path = resolve_path(
        config["candidates"]["input_path"]
    )

    with np.load(
        packed_path,
        allow_pickle=False,
    ) as packed:
        base_corpus = np.ascontiguousarray(
            packed["corpus_shingles"],
            dtype=np.int64,
        )

        base_offsets = np.ascontiguousarray(
            packed["document_offsets"],
            dtype=np.int64,
        )

        shingle_size = int(
            packed["shingle_size"]
        )

        document_sources = np.array(
            packed["document_sources"],
            copy=True,
        )

    candidate = None

    with candidate_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            row = json.loads(line)

            if row["candidate_id"] == candidate_id:
                candidate = row
                break

    if candidate is None:
        raise ValueError(
            f"Unknown candidate_id: {candidate_id}"
        )

    candidate_shingles = encode_text_shingles(
        candidate["text"],
        shingle_size=shingle_size,
    )

    base_corpus, base_offsets = select_fractional_corpus(
        base_corpus=base_corpus,
        base_offsets=base_offsets,
        document_sources=document_sources,
        corpus_fraction=corpus_fraction,
    )

    base_lengths = np.diff(base_offsets)

    document_lengths = np.tile(
        base_lengths,
        replication_factor,
    )

    document_offsets = np.empty(
        document_lengths.size + 1,
        dtype=np.int64,
    )

    document_offsets[0] = 0

    np.cumsum(
        document_lengths,
        dtype=np.int64,
        out=document_offsets[1:],
    )

    corpus_shingles = np.ascontiguousarray(
        np.tile(
            base_corpus,
            replication_factor,
        ),
        dtype=np.int64,
    )

    return (
        candidate,
        np.ascontiguousarray(
            candidate_shingles,
            dtype=np.int64,
        ),
        corpus_shingles,
        document_offsets,
        document_lengths,
    )


def run_backend(
    backend: str,
    candidate_shingles: np.ndarray,
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
):
    if backend == "serial":
        return _cpu.jaccard_corpus_serial(
            candidate_shingles,
            corpus_shingles,
            document_offsets,
        )

    return _cpu.jaccard_corpus_openmp(
        candidate_shingles,
        corpus_shingles,
        document_offsets,
    )


def choose_runs(document_count: int) -> int:
    if document_count <= 1_000:
        return 500

    if document_count <= 10_000:
        return 200

    if document_count <= 100_000:
        return 50

    return 20


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=["serial", "openmp"],
        required=True,
    )

    parser.add_argument(
        "--candidate-id",
        required=True,
    )

    parser.add_argument(
        "--corpus-fraction",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--replication-factor",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    arguments = parser.parse_args()

    if arguments.replication_factor < 1:
        raise ValueError(
            "replication-factor must be at least 1"
        )

    (
        candidate,
        candidate_shingles,
        corpus_shingles,
        document_offsets,
        document_lengths,
    ) = load_inputs(
        arguments.config.resolve(),
        arguments.candidate_id,
        arguments.corpus_fraction,
        arguments.replication_factor,
    )

    document_count = int(
        document_lengths.size
    )

    measured_runs = (
        arguments.measured_runs
        if arguments.measured_runs is not None
        else choose_runs(document_count)
    )

    # Verify before timing.
    serial_result = run_backend(
        "serial",
        candidate_shingles,
        corpus_shingles,
        document_offsets,
    )

    actual_result = run_backend(
        arguments.backend,
        candidate_shingles,
        corpus_shingles,
        document_offsets,
    )

    assert np.array_equal(
        serial_result[0],
        actual_result[0],
    )

    assert np.array_equal(
        serial_result[1],
        actual_result[1],
    )

    assert np.allclose(
        serial_result[2],
        actual_result[2],
        rtol=0.0,
        atol=1e-15,
    )

    # Warm up outside the measured region.
    for _ in range(arguments.warmup_runs):
        run_backend(
            arguments.backend,
            candidate_shingles,
            corpus_shingles,
            document_offsets,
        )

    timings = np.empty(
        measured_runs,
        dtype=np.int64,
    )

    for run_index in range(measured_runs):
        start = perf_counter_ns()

        run_backend(
            arguments.backend,
            candidate_shingles,
            corpus_shingles,
            document_offsets,
        )

        timings[run_index] = (
            perf_counter_ns() - start
        )

    output_path = arguments.output.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "package_version",
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
        "run_index",
        "elapsed_ns",
        "documents_per_second",
        "corpus_shingles_per_second",
        "estimated_input_bytes",
        "estimated_output_bytes",
    ]

    file_exists = output_path.exists()

    estimated_input_bytes = (
        candidate_shingles.nbytes
        + corpus_shingles.nbytes
        + document_offsets.nbytes
    )

    estimated_output_bytes = (
        document_count * 24
    )

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        if not file_exists:
            writer.writeheader()

        for run_index, elapsed_ns in enumerate(
            timings
        ):
            elapsed_seconds = (
                int(elapsed_ns)
                / 1_000_000_000
            )

            writer.writerow(
                {
                    "package_version": (
                        fortran_nlp.__version__
                    ),
                    "candidate_id": (
                        candidate["candidate_id"]
                    ),
                    "candidate_source": (
                        candidate["source"]
                    ),
                    "candidate_category": (
                        candidate["category"]
                    ),
                    "candidate_shingles": (
                        candidate_shingles.size
                    ),
                    "corpus_fraction": (
                        arguments.corpus_fraction
                    ),
                    "replication_factor": (
                        arguments.replication_factor
                    ),
                    "document_count": document_count,
                    "corpus_shingles": (
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
                    "backend": arguments.backend,
                    "thread_count": os.environ.get(
                        "OMP_NUM_THREADS",
                        "runtime-default",
                    ),
                    "omp_dynamic": os.environ.get(
                        "OMP_DYNAMIC",
                        "unset",
                    ),
                    "omp_proc_bind": os.environ.get(
                        "OMP_PROC_BIND",
                        "unset",
                    ),
                    "omp_places": os.environ.get(
                        "OMP_PLACES",
                        "unset",
                    ),
                    "omp_wait_policy": os.environ.get(
                        "OMP_WAIT_POLICY",
                        "unset",
                    ),
                    "warmup_runs": (
                        arguments.warmup_runs
                    ),
                    "measured_runs": measured_runs,
                    "run_index": run_index,
                    "elapsed_ns": int(elapsed_ns),
                    "documents_per_second": (
                        document_count
                        / elapsed_seconds
                    ),
                    "corpus_shingles_per_second": (
                        corpus_shingles.size
                        / elapsed_seconds
                    ),
                    "estimated_input_bytes": (
                        estimated_input_bytes
                    ),
                    "estimated_output_bytes": (
                        estimated_output_bytes
                    ),
                }
            )

    print("Real-corpus lexical benchmark")
    print("=" * 64)
    print(f"Candidate: {candidate['candidate_id']}")
    print(
        f"Candidate shingles: "
        f"{candidate_shingles.size:,}"
    )
    print(f"Backend: {arguments.backend}")
    print(
        "Threads: "
        f"{os.environ.get('OMP_NUM_THREADS', 'runtime-default')}"
    )
    print(
        f"Replication factor: "
        f"{arguments.replication_factor}"
    )
    print(f"Documents: {document_count:,}")
    print(
        f"Corpus shingles: "
        f"{corpus_shingles.size:,}"
    )
    print(f"Measured runs: {measured_runs}")
    print("-" * 64)
    print(
        "Median latency: "
        f"{np.median(timings) / 1_000_000:.6f} ms"
    )
    print(
        "P95 latency: "
        f"{np.percentile(timings, 95) / 1_000_000:.6f} ms"
    )
    print(
        "Median documents/s: "
        f"{document_count / (np.median(timings) / 1e9):,.2f}"
    )
    print(
        "Estimated input memory: "
        f"{estimated_input_bytes / (1024 ** 2):,.3f} MiB"
    )
    print(
        "Estimated output memory: "
        f"{estimated_output_bytes / (1024 ** 2):,.3f} MiB"
    )
    print("-" * 64)
    print(f"Raw CSV: {output_path}")


if __name__ == "__main__":
    main()