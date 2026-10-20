import numpy as np


def jaccard_sorted_reference(
    candidate_shingles: np.ndarray,
    existing_shingles: np.ndarray,
) -> tuple[int, int, float]:
    """Calculate exact Jaccard similarity using Python sets."""

    candidate_set = set(candidate_shingles.tolist())
    existing_set = set(existing_shingles.tolist())

    intersection_count = len(
        candidate_set.intersection(existing_set)
    )

    union_count = len(
        candidate_set.union(existing_set)
    )

    if union_count == 0:
        score = 0.0
    else:
        score = intersection_count / union_count

    return intersection_count, union_count, score