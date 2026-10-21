import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return one L2-normalized float32 vector."""

    vector = np.ascontiguousarray(
        vector,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")

    if vector.size == 0:
        raise ValueError("vector must not be empty")

    norm = np.linalg.norm(vector)

    if norm == 0.0:
        raise ValueError("zero vectors cannot be normalized")

    return np.ascontiguousarray(
        vector / norm,
        dtype=np.float32,
    )


def normalize_database(
    database: np.ndarray,
) -> np.ndarray:
    """L2-normalize each database vector independently."""

    database = np.ascontiguousarray(
        database,
        dtype=np.float32,
    )

    if database.ndim != 2:
        raise ValueError("database must be two-dimensional")

    if database.shape[0] == 0:
        raise ValueError(
            "database must contain at least one vector"
        )

    if database.shape[1] == 0:
        raise ValueError(
            "vector dimension must be greater than zero"
        )

    norms = np.linalg.norm(
        database,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0.0):
        raise ValueError(
            "database cannot contain zero vectors"
        )

    return np.ascontiguousarray(
        database / norms,
        dtype=np.float32,
    )