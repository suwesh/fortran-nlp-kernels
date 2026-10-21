import numpy as np
from fortran_nlp import _cpu
from .normalization import normalize_database, normalize_vector

SUPPORTED_BACKENDS = {
    "fortran",
    "fortran_openmp",
}

class _FlatIndexBase:
    """Shared validation and storage for exact flat indexes."""

    def __init__(
        self,
        database_vectors: np.ndarray,
        *,
        backend: str = "fortran",
    ):
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Available backends: "
                f"{', '.join(sorted(SUPPORTED_BACKENDS))}"
            )

        database = np.asarray(
            database_vectors,
            dtype=np.float32,
        )

        if database.ndim != 2:
            raise ValueError(
                "database_vectors must be two-dimensional"
            )

        if database.shape[0] == 0:
            raise ValueError(
                "database must contain at least one vector"
            )

        if database.shape[1] == 0:
            raise ValueError(
                "vector dimension must be greater than zero"
            )

        self._backend = backend
        self._vector_count = database.shape[0]
        self._dimension = database.shape[1]

        self._set_database(database)

    def _set_database(
        self,
        database: np.ndarray,
    ) -> None:
        """
        Store the database in native Fortran layout.

        Python layout:
            (vector_count, dimension)

        Fortran layout:
            (dimension, vector_count)
        """

        self._database = np.asfortranarray(
            database.T,
            dtype=np.float32,
        )

    def _prepare_query(
        self,
        query_vector: np.ndarray,
    ) -> np.ndarray:
        query = np.ascontiguousarray(
            query_vector,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "query_vector must be one-dimensional"
            )

        if query.shape[0] != self._dimension:
            raise ValueError(
                "query and database vectors must have "
                "equal dimensions"
            )

        return query

    def _validate_k(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")

        if k > self._vector_count:
            raise ValueError(
                "k cannot exceed the database size"
            )

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def vector_count(self) -> int:
        return self._vector_count


class FlatIndexIP(_FlatIndexBase):
    """Exact inner-product vector-search index."""

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_k(k)
        query = self._prepare_query(query_vector)

        if self._backend == "fortran":
            return _cpu.ip_search_serial(
                query,
                self._database,
                k,
            )

        if self._backend == "fortran_openmp":
            return _cpu.ip_search_openmp(
                query,
                self._database,
                k,
            )

        raise RuntimeError(
            f"Unsupported backend state: {self._backend}"
        )


class FlatIndexL2(_FlatIndexBase):
    """Exact squared-L2 vector-search index."""

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_k(k)
        query = self._prepare_query(query_vector)

        if self._backend == "fortran":
            return _cpu.l2_search_serial(
                query,
                self._database,
                k,
            )

        if self._backend == "fortran_openmp":
            return _cpu.l2_search_openmp(
                query,
                self._database,
                k,
            )

        raise RuntimeError(
            f"Unsupported backend state: {self._backend}"
        )


class FlatIndexCosine(FlatIndexIP):
    """
    Exact cosine-similarity vector-search index.

    Database vectors and queries are L2-normalized before
    inner-product search.
    """

    def __init__(
        self,
        database_vectors: np.ndarray,
        *,
        backend: str = "fortran",
    ):
        normalized_database = normalize_database(
            database_vectors
        )

        super().__init__(
            normalized_database,
            backend=backend,
        )

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_query = normalize_vector(
            query_vector
        )

        return super().search(
            normalized_query,
            k=k,
        )