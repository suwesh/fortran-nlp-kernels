import numpy as np

def inner_product_scores(
    query: np.ndarray,
    database: np.ndarray,
) -> np.ndarray:
    """Calculate exact inner-product scores."""
    return database @ query

def squared_l2_scores(
    query: np.ndarray,
    database: np.ndarray,
) -> np.ndarray:
    """Calculate exact squared-L2 distances."""
    differences = database - query
    return np.sum(
        differences * differences,
        axis=1,
        dtype=np.float32,
    )

def topk_largest(
    values: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the k largest values and zero-based IDs."""

    indices = np.argsort(
        -values,
        kind="stable",
    )[:k]

    return values[indices], indices.astype(np.int64)


def topk_smallest(
    values: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the k smallest values and zero-based IDs."""

    indices = np.argsort(
        values,
        kind="stable",
    )[:k]

    return values[indices], indices.astype(np.int64)


def search_ip(
    query: np.ndarray,
    database: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = inner_product_scores(query, database)
    return topk_largest(scores, k)


def search_l2(
    query: np.ndarray,
    database: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = squared_l2_scores(query, database)
    return topk_smallest(distances, k)