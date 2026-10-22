from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LexicalWorkload:
    candidate_shingles: np.ndarray
    corpus_shingles: np.ndarray
    document_offsets: np.ndarray
    document_lengths: np.ndarray
    target_overlap: float
    seed: int

    @property
    def document_count(self) -> int:
        return int(self.document_lengths.size)

    @property
    def candidate_shingle_count(self) -> int:
        return int(self.candidate_shingles.size)

    @property
    def corpus_shingle_count(self) -> int:
        return int(self.corpus_shingles.size)


def _validate_generation_inputs(
    document_count: int,
    shingles_per_document: int,
    candidate_shingle_count: int,
    target_overlap: float,
    variable_lengths: bool,
    minimum_length_ratio: float,
) -> None:
    if document_count < 1:
        raise ValueError(
            "document_count must be at least 1"
        )

    if shingles_per_document < 1:
        raise ValueError(
            "shingles_per_document must be at least 1"
        )

    if candidate_shingle_count < 1:
        raise ValueError(
            "candidate_shingle_count must be at least 1"
        )

    if not 0.0 <= target_overlap <= 1.0:
        raise ValueError(
            "target_overlap must be between 0 and 1"
        )

    if variable_lengths and not (
        0.0 < minimum_length_ratio <= 1.0
    ):
        raise ValueError(
            "minimum_length_ratio must be in (0, 1]"
        )


def _generate_document_lengths(
    rng: np.random.Generator,
    document_count: int,
    shingles_per_document: int,
    *,
    variable_lengths: bool,
    minimum_length_ratio: float,
) -> np.ndarray:
    if not variable_lengths:
        return np.full(
            document_count,
            shingles_per_document,
            dtype=np.int64,
        )

    minimum_length = max(
        1,
        int(
            round(
                shingles_per_document
                * minimum_length_ratio
            )
        ),
    )

    maximum_length = max(
        minimum_length,
        shingles_per_document,
    )

    return rng.integers(
        low=minimum_length,
        high=maximum_length + 1,
        size=document_count,
        dtype=np.int64,
    )


def _target_intersection_size(
    candidate_size: int,
    document_size: int,
    target_jaccard: float,
) -> int:
    """
    Solve approximately for intersection I:

        J = I / (|A| + |B| - I)

    Therefore:

        I = J(|A| + |B|) / (1 + J)
    """

    if target_jaccard == 0.0:
        return 0

    intersection = round(
        target_jaccard
        * (candidate_size + document_size)
        / (1.0 + target_jaccard)
    )

    return min(
        int(intersection),
        candidate_size,
        document_size,
    )


def _generate_candidate(
    candidate_shingle_count: int,
) -> np.ndarray:
    # Even values are reserved for the candidate.
    # Corpus-only shingles use odd values.
    return np.arange(
        0,
        candidate_shingle_count * 2,
        2,
        dtype=np.int64,
    )


def _generate_document(
    rng: np.random.Generator,
    candidate: np.ndarray,
    document_size: int,
    target_overlap: float,
    unique_base: int,
) -> np.ndarray:
    intersection_size = _target_intersection_size(
        candidate.size,
        document_size,
        target_overlap,
    )

    if intersection_size > 0:
        shared = rng.choice(
            candidate,
            size=intersection_size,
            replace=False,
        )
    else:
        shared = np.empty(
            0,
            dtype=np.int64,
        )

    unique_size = document_size - intersection_size

    # Odd-valued document-only shingles cannot collide
    # with the candidate's even-valued shingles.
    unique = (
        unique_base
        + 2 * np.arange(
            unique_size,
            dtype=np.int64,
        )
        + 1
    )

    document = np.concatenate(
        [
            shared,
            unique,
        ]
    )

    document.sort()

    return np.ascontiguousarray(
        document,
        dtype=np.int64,
    )


def generate_lexical_workload(
    *,
    document_count: int,
    shingles_per_document: int,
    candidate_shingle_count: int,
    target_overlap: float,
    variable_lengths: bool = False,
    minimum_length_ratio: float = 0.25,
    seed: int = 20261022,
) -> LexicalWorkload:
    """
    Generate one deterministic candidate and a packed corpus.

    The candidate and every document contain sorted, unique int64
    shingles. Each document is constructed to approximate the requested
    Jaccard overlap with the candidate.
    """

    _validate_generation_inputs(
        document_count,
        shingles_per_document,
        candidate_shingle_count,
        target_overlap,
        variable_lengths,
        minimum_length_ratio,
    )

    rng = np.random.default_rng(seed)

    candidate = _generate_candidate(
        candidate_shingle_count
    )

    document_lengths = _generate_document_lengths(
        rng,
        document_count,
        shingles_per_document,
        variable_lengths=variable_lengths,
        minimum_length_ratio=minimum_length_ratio,
    )

    document_offsets = np.empty(
        document_count + 1,
        dtype=np.int64,
    )

    document_offsets[0] = 0

    np.cumsum(
        document_lengths,
        dtype=np.int64,
        out=document_offsets[1:],
    )

    corpus_shingles = np.empty(
        int(document_offsets[-1]),
        dtype=np.int64,
    )

    unique_base = (
        candidate_shingle_count * 4
        + seed * 2
    )

    for document_index in range(document_count):
        document_size = int(
            document_lengths[document_index]
        )

        document = _generate_document(
            rng,
            candidate,
            document_size,
            target_overlap,
            unique_base,
        )

        start = int(
            document_offsets[document_index]
        )

        end = int(
            document_offsets[document_index + 1]
        )

        corpus_shingles[start:end] = document

        # Reserve a disjoint odd-valued range for the next
        # document's non-overlapping shingles.
        unique_base += document_size * 4 + 2

    return LexicalWorkload(
        candidate_shingles=np.ascontiguousarray(
            candidate,
            dtype=np.int64,
        ),
        corpus_shingles=np.ascontiguousarray(
            corpus_shingles,
            dtype=np.int64,
        ),
        document_offsets=np.ascontiguousarray(
            document_offsets,
            dtype=np.int64,
        ),
        document_lengths=np.ascontiguousarray(
            document_lengths,
            dtype=np.int64,
        ),
        target_overlap=float(target_overlap),
        seed=seed,
    )


def calculate_actual_jaccard_scores(
    workload: LexicalWorkload,
) -> np.ndarray:
    """Calculate exact Python-set scores for validation."""

    candidate_set = set(
        workload.candidate_shingles.tolist()
    )

    scores = np.empty(
        workload.document_count,
        dtype=np.float64,
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

        document_set = set(
            workload.corpus_shingles[
                start:end
            ].tolist()
        )

        intersection_count = len(
            candidate_set & document_set
        )

        union_count = len(
            candidate_set | document_set
        )

        if union_count == 0:
            scores[document_index] = 0.0
        else:
            scores[document_index] = (
                intersection_count
                / union_count
            )

    return scores


if __name__ == "__main__":
    workload = generate_lexical_workload(
        document_count=5,
        shingles_per_document=64,
        candidate_shingle_count=64,
        target_overlap=0.5,
        variable_lengths=False,
        seed=20261022,
    )

    actual_scores = calculate_actual_jaccard_scores(
        workload
    )

    print(
        "Documents:",
        workload.document_count,
    )

    print(
        "Candidate shingles:",
        workload.candidate_shingle_count,
    )

    print(
        "Corpus shingles:",
        workload.corpus_shingle_count,
    )

    print(
        "Document lengths:",
        workload.document_lengths,
    )

    print(
        "Actual Jaccard scores:",
        actual_scores,
    )