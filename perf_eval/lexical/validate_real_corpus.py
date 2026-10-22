from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import fortran_nlp
from fortran_nlp import _cpu
from fortran_nlp.lexical.preprocessing import (
    encode_text_shingles,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "perf_eval"
    / "config"
    / "lexical_cpu_matrix.local.yaml"
)

ABSOLUTE_TOLERANCE = 1.0e-15


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate serial, OpenMP, and batched OpenMP "
            "Jaccard scoring on the real lexical corpus."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    return parser.parse_args()


def load_config(
    config_path: Path,
) -> dict:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping"
        )

    return config


def resolve_repository_path(
    configured_path: str,
) -> Path:
    path = Path(configured_path).expanduser()

    if not path.is_absolute():
        path = REPOSITORY_ROOT / path

    return path.resolve()


def load_packed_corpus(
    packed_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    with np.load(
        packed_path,
        allow_pickle=False,
    ) as packed:
        required_arrays = (
            "corpus_shingles",
            "document_offsets",
            "document_lengths",
            "document_sources",
            "document_ids",
            "shingle_size",
        )

        for array_name in required_arrays:
            if array_name not in packed:
                raise ValueError(
                    f"Packed corpus is missing "
                    f"'{array_name}'"
                )

        corpus_shingles = np.ascontiguousarray(
            packed["corpus_shingles"],
            dtype=np.int64,
        )

        document_offsets = np.ascontiguousarray(
            packed["document_offsets"],
            dtype=np.int64,
        )

        document_lengths = np.ascontiguousarray(
            packed["document_lengths"],
            dtype=np.int64,
        )

        document_sources = np.array(
            packed["document_sources"],
            copy=True,
        )

        document_ids = np.array(
            packed["document_ids"],
            copy=True,
        )

        shingle_size = int(
            packed["shingle_size"]
        )

    return (
        corpus_shingles,
        document_offsets,
        document_lengths,
        document_sources,
        document_ids,
        shingle_size,
    )


def validate_packed_corpus(
    *,
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
    document_lengths: np.ndarray,
    document_sources: np.ndarray,
    document_ids: np.ndarray,
) -> None:
    document_count = document_lengths.size

    if corpus_shingles.ndim != 1:
        raise AssertionError(
            "corpus_shingles must be one-dimensional"
        )

    if document_offsets.ndim != 1:
        raise AssertionError(
            "document_offsets must be one-dimensional"
        )

    if corpus_shingles.dtype != np.int64:
        raise AssertionError(
            "corpus_shingles must use int64"
        )

    if document_offsets.dtype != np.int64:
        raise AssertionError(
            "document_offsets must use int64"
        )

    if document_lengths.dtype != np.int64:
        raise AssertionError(
            "document_lengths must use int64"
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

    if not np.array_equal(
        np.diff(document_offsets),
        document_lengths,
    ):
        raise AssertionError(
            "document lengths do not match offsets"
        )

    if document_sources.size != document_count:
        raise AssertionError(
            "document source count is inconsistent"
        )

    if document_ids.size != document_count:
        raise AssertionError(
            "document ID count is inconsistent"
        )

    for document_index in range(document_count):
        start = int(
            document_offsets[document_index]
        )

        end = int(
            document_offsets[document_index + 1]
        )

        document = corpus_shingles[start:end]

        if document.size == 0:
            raise AssertionError(
                f"Document {document_index} has no shingles"
            )

        if (
            document.size > 1
            and not np.all(
                document[1:] > document[:-1]
            )
        ):
            raise AssertionError(
                f"Document {document_index} shingles "
                "are not sorted and unique"
            )


def load_candidates(
    candidate_path: Path,
    *,
    shingle_size: int,
) -> list[dict]:
    candidates = []

    with candidate_path.open(
        "r",
        encoding="utf-8",
    ) as candidate_file:
        for line_number, line in enumerate(
            candidate_file,
            start=1,
        ):
            if not line.strip():
                continue

            candidate = json.loads(line)

            required_fields = (
                "candidate_id",
                "category",
                "source",
                "origin_positional_id",
                "origin_document_id",
                "shingle_count",
                "text",
            )

            for field_name in required_fields:
                if field_name not in candidate:
                    raise ValueError(
                        f"Candidate line {line_number} "
                        f"is missing '{field_name}'"
                    )

            shingles = encode_text_shingles(
                candidate["text"],
                shingle_size=shingle_size,
            )

            if shingles.size == 0:
                raise ValueError(
                    f"Candidate "
                    f"'{candidate['candidate_id']}' "
                    "contains no shingles"
                )

            if (
                int(candidate["shingle_count"])
                != shingles.size
            ):
                raise ValueError(
                    f"Stored shingle count for candidate "
                    f"'{candidate['candidate_id']}' "
                    "does not match encoded count"
                )

            candidate["shingles"] = (
                np.ascontiguousarray(
                    shingles,
                    dtype=np.int64,
                )
            )

            candidates.append(candidate)

    if not candidates:
        raise ValueError(
            "Candidate file contains no candidates"
        )

    return candidates


def assert_result_equivalence(
    *,
    expected: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    actual: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    description: str,
) -> None:
    if not np.array_equal(
        expected[0],
        actual[0],
    ):
        mismatch = np.flatnonzero(
            expected[0] != actual[0]
        )

        raise AssertionError(
            f"Intersection mismatch for {description}; "
            f"first mismatches: {mismatch[:10]}"
        )

    if not np.array_equal(
        expected[1],
        actual[1],
    ):
        mismatch = np.flatnonzero(
            expected[1] != actual[1]
        )

        raise AssertionError(
            f"Union mismatch for {description}; "
            f"first mismatches: {mismatch[:10]}"
        )

    if not np.allclose(
        expected[2],
        actual[2],
        rtol=0.0,
        atol=ABSOLUTE_TOLERANCE,
    ):
        differences = np.abs(
            expected[2] - actual[2]
        )

        mismatch = np.flatnonzero(
            differences > ABSOLUTE_TOLERANCE
        )

        maximum_error = float(
            np.max(differences)
        )

        raise AssertionError(
            f"Score mismatch for {description}; "
            f"maximum error={maximum_error}; "
            f"first mismatches={mismatch[:10]}"
        )


def validate_origin_mapping(
    *,
    candidate: dict,
    document_sources: np.ndarray,
    document_ids: np.ndarray,
    document_count: int,
) -> int:
    origin_index = int(
        candidate["origin_positional_id"]
    )

    if not 0 <= origin_index < document_count:
        raise AssertionError(
            f"Candidate '{candidate['candidate_id']}' "
            "has an invalid origin positional ID"
        )

    actual_source = str(
        document_sources[origin_index]
    )

    actual_document_id = str(
        document_ids[origin_index]
    )

    if actual_source != candidate["source"]:
        raise AssertionError(
            f"Candidate '{candidate['candidate_id']}' "
            "source does not match corpus origin"
        )

    if (
        actual_document_id
        != str(candidate["origin_document_id"])
    ):
        raise AssertionError(
            f"Candidate '{candidate['candidate_id']}' "
            "document ID does not match corpus origin"
        )

    return origin_index


def pack_candidates(
    candidates: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_lengths = np.array(
        [
            candidate["shingles"].size
            for candidate in candidates
        ],
        dtype=np.int64,
    )

    candidate_offsets = np.empty(
        len(candidates) + 1,
        dtype=np.int64,
    )

    candidate_offsets[0] = 0

    np.cumsum(
        candidate_lengths,
        dtype=np.int64,
        out=candidate_offsets[1:],
    )

    candidate_shingles = np.concatenate(
        [
            candidate["shingles"]
            for candidate in candidates
        ]
    )

    return (
        np.ascontiguousarray(
            candidate_shingles,
            dtype=np.int64,
        ),
        np.ascontiguousarray(
            candidate_offsets,
            dtype=np.int64,
        ),
    )


def validate_single_candidates(
    *,
    candidates: list[dict],
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
    document_sources: np.ndarray,
    document_ids: np.ndarray,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:
    document_count = document_offsets.size - 1
    serial_results = []

    for candidate in candidates:
        origin_index = validate_origin_mapping(
            candidate=candidate,
            document_sources=document_sources,
            document_ids=document_ids,
            document_count=document_count,
        )

        serial_result = (
            _cpu.jaccard_corpus_serial(
                candidate["shingles"],
                corpus_shingles,
                document_offsets,
            )
        )

        openmp_result = (
            _cpu.jaccard_corpus_openmp(
                candidate["shingles"],
                corpus_shingles,
                document_offsets,
            )
        )

        assert_result_equivalence(
            expected=serial_result,
            actual=openmp_result,
            description=(
                f"candidate "
                f"'{candidate['candidate_id']}' "
                "serial/OpenMP"
            ),
        )

        if serial_result[0][origin_index] != (
            candidate["shingles"].size
        ):
            raise AssertionError(
                f"Candidate '{candidate['candidate_id']}' "
                "does not fully intersect its origin"
            )

        if serial_result[1][origin_index] != (
            candidate["shingles"].size
        ):
            raise AssertionError(
                f"Candidate '{candidate['candidate_id']}' "
                "has an unexpected origin union count"
            )

        if not np.isclose(
            serial_result[2][origin_index],
            1.0,
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        ):
            raise AssertionError(
                f"Candidate '{candidate['candidate_id']}' "
                "does not score 1.0 against its origin"
            )

        maximum_score = float(
            np.max(serial_result[2])
        )

        if not np.isclose(
            maximum_score,
            1.0,
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        ):
            raise AssertionError(
                f"Candidate '{candidate['candidate_id']}' "
                "has an unexpected maximum score"
            )

        serial_results.append(serial_result)

        print(
            f"{candidate['candidate_id']}: "
            f"shingles={candidate['shingles'].size}, "
            f"origin={origin_index}, "
            "serial/OpenMP=PASS, "
            "self-match=PASS"
        )

    return serial_results


def validate_batch_candidates(
    *,
    candidates: list[dict],
    serial_results: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
) -> None:
    (
        packed_candidates,
        candidate_offsets,
    ) = pack_candidates(candidates)

    (
        batch_intersections,
        batch_unions,
        batch_scores,
    ) = _cpu.jaccard_batch_corpus(
        packed_candidates,
        candidate_offsets,
        corpus_shingles,
        document_offsets,
    )

    document_count = document_offsets.size - 1

    expected_shape = (
        document_count,
        len(candidates),
    )

    if batch_intersections.shape != expected_shape:
        raise AssertionError(
            "Batched intersection matrix has "
            "an incorrect shape"
        )

    if batch_unions.shape != expected_shape:
        raise AssertionError(
            "Batched union matrix has "
            "an incorrect shape"
        )

    if batch_scores.shape != expected_shape:
        raise AssertionError(
            "Batched score matrix has "
            "an incorrect shape"
        )

    for candidate_index, candidate in enumerate(
        candidates
    ):
        serial_result = serial_results[
            candidate_index
        ]

        batch_result = (
            batch_intersections[
                :,
                candidate_index,
            ],
            batch_unions[
                :,
                candidate_index,
            ],
            batch_scores[
                :,
                candidate_index,
            ],
        )

        assert_result_equivalence(
            expected=serial_result,
            actual=batch_result,
            description=(
                f"candidate "
                f"'{candidate['candidate_id']}' "
                "serial/batched"
            ),
        )

    print(
        "Batched OpenMP/repeated serial equivalence: PASS"
    )


def main() -> None:
    arguments = parse_arguments()

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    config = load_config(config_path)

    packed_path = resolve_repository_path(
        config["packed_corpus"]["output_path"]
    )

    candidate_path = resolve_repository_path(
        config["candidates"]["input_path"]
    )

    (
        corpus_shingles,
        document_offsets,
        document_lengths,
        document_sources,
        document_ids,
        shingle_size,
    ) = load_packed_corpus(packed_path)

    validate_packed_corpus(
        corpus_shingles=corpus_shingles,
        document_offsets=document_offsets,
        document_lengths=document_lengths,
        document_sources=document_sources,
        document_ids=document_ids,
    )

    candidates = load_candidates(
        candidate_path,
        shingle_size=shingle_size,
    )

    print("Real lexical corpus validation")
    print("=" * 72)

    print(
        f"fortran-nlp-kernels: "
        f"{fortran_nlp.__version__}"
    )

    print(
        f"Documents: {document_lengths.size}"
    )

    print(
        f"Corpus shingles: {corpus_shingles.size}"
    )

    print(
        f"Candidates: {len(candidates)}"
    )

    print(
        "Candidate shingle range: "
        f"{min(c['shingles'].size for c in candidates)}"
        "–"
        f"{max(c['shingles'].size for c in candidates)}"
    )

    print("-" * 72)

    serial_results = validate_single_candidates(
        candidates=candidates,
        corpus_shingles=corpus_shingles,
        document_offsets=document_offsets,
        document_sources=document_sources,
        document_ids=document_ids,
    )

    print("-" * 72)

    validate_batch_candidates(
        candidates=candidates,
        serial_results=serial_results,
        corpus_shingles=corpus_shingles,
        document_offsets=document_offsets,
    )

    print("=" * 72)

    print(
        "Real lexical corpus validation: PASS"
    )


if __name__ == "__main__":
    main()