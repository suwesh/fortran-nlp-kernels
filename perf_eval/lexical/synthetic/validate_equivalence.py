from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


# Allow execution directly from the repository root:
#
# python perf_eval/lexical/validate_equivalence.py
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from fortran_nlp import _cpu

from generate_workloads import (
    LexicalWorkload,
    generate_lexical_workload,
)


ATOL = 1.0e-15

SEEDS = [
    20261022,
    20261023,
]

DOCUMENT_COUNTS = [
    1,
    7,
    37,
]

SHINGLE_COUNTS = [
    16,
    64,
    256,
    1024,
]

TARGET_OVERLAPS = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
]

LENGTH_MODES = [
    "fixed",
    "variable",
]


@dataclass(frozen=True)
class ReferenceResult:
    intersections: np.ndarray
    unions: np.ndarray
    scores: np.ndarray


def validate_sorted_unique(
    values: np.ndarray,
    *,
    name: str,
) -> None:
    if values.dtype != np.int64:
        raise AssertionError(
            f"{name} must use int64"
        )

    if values.ndim != 1:
        raise AssertionError(
            f"{name} must be one-dimensional"
        )

    if not values.flags.c_contiguous:
        raise AssertionError(
            f"{name} must be C-contiguous"
        )

    if values.size > 1:
        if not np.all(values[1:] > values[:-1]):
            raise AssertionError(
                f"{name} must be sorted and unique"
            )


def validate_packed_workload(
    workload: LexicalWorkload,
) -> None:
    validate_sorted_unique(
        workload.candidate_shingles,
        name="candidate_shingles",
    )

    if workload.corpus_shingles.dtype != np.int64:
        raise AssertionError(
            "corpus_shingles must use int64"
        )

    if workload.document_offsets.dtype != np.int64:
        raise AssertionError(
            "document_offsets must use int64"
        )

    if not workload.corpus_shingles.flags.c_contiguous:
        raise AssertionError(
            "corpus_shingles must be C-contiguous"
        )

    if not workload.document_offsets.flags.c_contiguous:
        raise AssertionError(
            "document_offsets must be C-contiguous"
        )

    if workload.document_offsets[0] != 0:
        raise AssertionError(
            "document_offsets must begin with zero"
        )

    if (
        workload.document_offsets[-1]
        != workload.corpus_shingles.size
    ):
        raise AssertionError(
            "final document offset must equal "
            "corpus shingle count"
        )

    if not np.all(
        workload.document_offsets[1:]
        >= workload.document_offsets[:-1]
    ):
        raise AssertionError(
            "document_offsets must be nondecreasing"
        )

    if (
        workload.document_offsets.size - 1
        != workload.document_count
    ):
        raise AssertionError(
            "document offset count is inconsistent"
        )

    measured_lengths = np.diff(
        workload.document_offsets
    )

    if not np.array_equal(
        measured_lengths,
        workload.document_lengths,
    ):
        raise AssertionError(
            "document lengths do not match offsets"
        )

    for document_index in range(
        workload.document_count
    ):
        start = int(
            workload.document_offsets[
                document_index
            ]
        )

        end = int(
            workload.document_offsets[
                document_index + 1
            ]
        )

        validate_sorted_unique(
            np.ascontiguousarray(
                workload.corpus_shingles[start:end],
                dtype=np.int64,
            ),
            name=f"document_{document_index}",
        )


def python_jaccard_reference(
    candidate_shingles: np.ndarray,
    corpus_shingles: np.ndarray,
    document_offsets: np.ndarray,
) -> ReferenceResult:
    candidate_set = set(
        candidate_shingles.tolist()
    )

    document_count = document_offsets.size - 1

    intersections = np.empty(
        document_count,
        dtype=np.int64,
    )

    unions = np.empty(
        document_count,
        dtype=np.int64,
    )

    scores = np.empty(
        document_count,
        dtype=np.float64,
    )

    for document_index in range(document_count):
        start = int(
            document_offsets[document_index]
        )

        end = int(
            document_offsets[document_index + 1]
        )

        document_set = set(
            corpus_shingles[start:end].tolist()
        )

        intersection_count = len(
            candidate_set.intersection(
                document_set
            )
        )

        union_count = (
            len(candidate_set)
            + len(document_set)
            - intersection_count
        )

        intersections[document_index] = (
            intersection_count
        )

        unions[document_index] = union_count

        if union_count == 0:
            scores[document_index] = 0.0
        else:
            scores[document_index] = (
                intersection_count
                / union_count
            )

    return ReferenceResult(
        intersections=intersections,
        unions=unions,
        scores=scores,
    )


def assert_result_equivalence(
    *,
    expected: ReferenceResult,
    actual: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    implementation_name: str,
    configuration_name: str,
) -> None:
    actual_intersections = actual[0]
    actual_unions = actual[1]
    actual_scores = actual[2]

    if actual_intersections.dtype != np.int64:
        raise AssertionError(
            f"{implementation_name} returned incorrect "
            f"intersection dtype for {configuration_name}"
        )

    if actual_unions.dtype != np.int64:
        raise AssertionError(
            f"{implementation_name} returned incorrect "
            f"union dtype for {configuration_name}"
        )

    if actual_scores.dtype != np.float64:
        raise AssertionError(
            f"{implementation_name} returned incorrect "
            f"score dtype for {configuration_name}"
        )

    if not np.array_equal(
        actual_intersections,
        expected.intersections,
    ):
        mismatch = np.flatnonzero(
            actual_intersections
            != expected.intersections
        )

        raise AssertionError(
            f"{implementation_name} intersection "
            f"mismatch for {configuration_name}; "
            f"first mismatches: {mismatch[:10]}"
        )

    if not np.array_equal(
        actual_unions,
        expected.unions,
    ):
        mismatch = np.flatnonzero(
            actual_unions
            != expected.unions
        )

        raise AssertionError(
            f"{implementation_name} union mismatch "
            f"for {configuration_name}; "
            f"first mismatches: {mismatch[:10]}"
        )

    if not np.allclose(
        actual_scores,
        expected.scores,
        rtol=0.0,
        atol=ATOL,
    ):
        mismatch = np.flatnonzero(
            np.abs(
                actual_scores
                - expected.scores
            )
            > ATOL
        )

        maximum_error = float(
            np.max(
                np.abs(
                    actual_scores
                    - expected.scores
                )
            )
        )

        raise AssertionError(
            f"{implementation_name} score mismatch "
            f"for {configuration_name}; "
            f"maximum error={maximum_error}; "
            f"first mismatches={mismatch[:10]}"
        )


def create_batch_candidates(
    candidate: np.ndarray,
) -> list[np.ndarray]:
    """
    Create several deterministic candidate variants.

    The variants test:

    1. the original candidate;
    2. a smaller subset;
    3. a shifted candidate with different shingles;
    4. a duplicate candidate column.
    """

    subset = np.ascontiguousarray(
        candidate[::2],
        dtype=np.int64,
    )

    shifted = np.ascontiguousarray(
        candidate + 1,
        dtype=np.int64,
    )

    duplicate = np.ascontiguousarray(
        candidate.copy(),
        dtype=np.int64,
    )

    return [
        np.ascontiguousarray(
            candidate,
            dtype=np.int64,
        ),
        subset,
        shifted,
        duplicate,
    ]


def pack_candidates(
    candidates: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError(
            "candidates must not be empty"
        )

    lengths = np.array(
        [
            candidate.size
            for candidate in candidates
        ],
        dtype=np.int64,
    )

    offsets = np.empty(
        len(candidates) + 1,
        dtype=np.int64,
    )

    offsets[0] = 0

    np.cumsum(
        lengths,
        dtype=np.int64,
        out=offsets[1:],
    )

    if offsets[-1] == 0:
        packed = np.empty(
            0,
            dtype=np.int64,
        )
    else:
        packed = np.concatenate(
            candidates
        ).astype(
            np.int64,
            copy=False,
        )

    return (
        np.ascontiguousarray(
            packed,
            dtype=np.int64,
        ),
        np.ascontiguousarray(
            offsets,
            dtype=np.int64,
        ),
    )


def validate_batch_equivalence(
    *,
    workload: LexicalWorkload,
    configuration_name: str,
) -> None:
    candidates = create_batch_candidates(
        workload.candidate_shingles
    )

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
        workload.corpus_shingles,
        workload.document_offsets,
    )

    expected_shape = (
        workload.document_count,
        len(candidates),
    )

    if batch_intersections.shape != expected_shape:
        raise AssertionError(
            "incorrect batched intersection shape "
            f"for {configuration_name}: "
            f"{batch_intersections.shape}"
        )

    if batch_unions.shape != expected_shape:
        raise AssertionError(
            "incorrect batched union shape "
            f"for {configuration_name}: "
            f"{batch_unions.shape}"
        )

    if batch_scores.shape != expected_shape:
        raise AssertionError(
            "incorrect batched score shape "
            f"for {configuration_name}: "
            f"{batch_scores.shape}"
        )

    for candidate_index, candidate in enumerate(
        candidates
    ):
        expected = python_jaccard_reference(
            candidate,
            workload.corpus_shingles,
            workload.document_offsets,
        )

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
            expected=expected,
            actual=batch_result,
            implementation_name=(
                "Fortran batched OpenMP"
            ),
            configuration_name=(
                f"{configuration_name}, "
                f"batch_candidate={candidate_index}"
            ),
        )

        serial_result = (
            _cpu.jaccard_corpus_serial(
                candidate,
                workload.corpus_shingles,
                workload.document_offsets,
            )
        )

        assert_result_equivalence(
            expected=expected,
            actual=serial_result,
            implementation_name=(
                "Fortran repeated serial"
            ),
            configuration_name=(
                f"{configuration_name}, "
                f"batch_candidate={candidate_index}"
            ),
        )

        if not np.array_equal(
            batch_intersections[
                :,
                candidate_index,
            ],
            serial_result[0],
        ):
            raise AssertionError(
                "batch and repeated serial "
                "intersections differ"
            )

        if not np.array_equal(
            batch_unions[
                :,
                candidate_index,
            ],
            serial_result[1],
        ):
            raise AssertionError(
                "batch and repeated serial "
                "unions differ"
            )

        if not np.allclose(
            batch_scores[
                :,
                candidate_index,
            ],
            serial_result[2],
            rtol=0.0,
            atol=ATOL,
        ):
            raise AssertionError(
                "batch and repeated serial "
                "scores differ"
            )


def validate_configuration(
    *,
    seed: int,
    document_count: int,
    shingle_count: int,
    target_overlap: float,
    length_mode: str,
) -> None:
    variable_lengths = (
        length_mode == "variable"
    )

    workload = generate_lexical_workload(
        document_count=document_count,
        shingles_per_document=shingle_count,
        candidate_shingle_count=shingle_count,
        target_overlap=target_overlap,
        variable_lengths=variable_lengths,
        minimum_length_ratio=0.25,
        seed=seed,
    )

    configuration_name = (
        f"seed={seed}, "
        f"documents={document_count}, "
        f"shingles={shingle_count}, "
        f"overlap={target_overlap:.2f}, "
        f"length_mode={length_mode}"
    )

    validate_packed_workload(workload)

    expected = python_jaccard_reference(
        workload.candidate_shingles,
        workload.corpus_shingles,
        workload.document_offsets,
    )

    serial_result = _cpu.jaccard_corpus_serial(
        workload.candidate_shingles,
        workload.corpus_shingles,
        workload.document_offsets,
    )

    openmp_result = _cpu.jaccard_corpus_openmp(
        workload.candidate_shingles,
        workload.corpus_shingles,
        workload.document_offsets,
    )

    assert_result_equivalence(
        expected=expected,
        actual=serial_result,
        implementation_name="Fortran serial",
        configuration_name=configuration_name,
    )

    assert_result_equivalence(
        expected=expected,
        actual=openmp_result,
        implementation_name="Fortran OpenMP",
        configuration_name=configuration_name,
    )

    if not np.array_equal(
        serial_result[0],
        openmp_result[0],
    ):
        raise AssertionError(
            "serial/OpenMP intersections differ "
            f"for {configuration_name}"
        )

    if not np.array_equal(
        serial_result[1],
        openmp_result[1],
    ):
        raise AssertionError(
            "serial/OpenMP unions differ "
            f"for {configuration_name}"
        )

    if not np.allclose(
        serial_result[2],
        openmp_result[2],
        rtol=0.0,
        atol=ATOL,
    ):
        raise AssertionError(
            "serial/OpenMP scores differ "
            f"for {configuration_name}"
        )

    validate_batch_equivalence(
        workload=workload,
        configuration_name=configuration_name,
    )


def main() -> None:
    configuration_count = (
        len(SEEDS)
        * len(DOCUMENT_COUNTS)
        * len(SHINGLE_COUNTS)
        * len(TARGET_OVERLAPS)
        * len(LENGTH_MODES)
    )

    completed_count = 0

    print(
        "Lexical backend equivalence validation"
    )

    print("=" * 72)

    print(
        f"Configurations: {configuration_count}"
    )

    print(
        "Implementations: Python set, Fortran serial, "
        "Fortran OpenMP, batched OpenMP"
    )

    print("=" * 72)

    for seed in SEEDS:
        for document_count in DOCUMENT_COUNTS:
            for shingle_count in SHINGLE_COUNTS:
                for target_overlap in TARGET_OVERLAPS:
                    for length_mode in LENGTH_MODES:
                        validate_configuration(
                            seed=seed,
                            document_count=(
                                document_count
                            ),
                            shingle_count=(
                                shingle_count
                            ),
                            target_overlap=(
                                target_overlap
                            ),
                            length_mode=length_mode,
                        )

                        completed_count += 1

                        if (
                            completed_count % 25 == 0
                            or completed_count
                            == configuration_count
                        ):
                            print(
                                "Validated "
                                f"{completed_count}/"
                                f"{configuration_count}"
                            )

    print("=" * 72)

    print(
        "Lexical equivalence validation: PASS"
    )

    print(
        f"Validated configurations: "
        f"{completed_count}"
    )


if __name__ == "__main__":
    main()