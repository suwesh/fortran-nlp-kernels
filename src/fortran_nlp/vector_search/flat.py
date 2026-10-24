import numpy as np

from fortran_nlp import _cpu
from fortran_nlp.auto import (
    BackendDecision,
    VectorWorkload,
    resolve_vector_backend,
)
from fortran_nlp.runtime import apply_openmp_thread_policy, apply_openblas_thread_policy, get_available_memory_bytes

from .normalization import normalize_database, normalize_vector


SUPPORTED_BACKENDS = {
    "auto",
    "fortran",
    "fortran_openmp",
    "fortran_blas",
}


class _FlatIndexBase:
    """Shared validation, storage, and dispatch for exact flat indexes."""

    _metric: str

    def __init__(
        self,
        database_vectors: np.ndarray,
        *,
        backend: str = "fortran",
    ):
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                "Available backends: "
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

        # Preserve the user's request, including "auto".
        self._backend = backend
        self._vector_count = int(database.shape[0])
        self._dimension = int(database.shape[1])

        self._set_database(database)

    def _set_database(
        self,
        database: np.ndarray,
    ) -> None:
        """Store the database as (dimension, vector_count)."""

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

    def _prepare_query_batch(
        self,
        query_vectors: np.ndarray,
    ) -> np.ndarray:
        queries = np.asarray(
            query_vectors,
            dtype=np.float32,
        )

        if queries.ndim != 2:
            raise ValueError(
                "query_vectors must be two-dimensional"
            )

        if queries.shape[0] == 0:
            raise ValueError(
                "query batch must contain at least one vector"
            )

        if queries.shape[1] != self._dimension:
            raise ValueError(
                "queries and database vectors must have "
                "equal dimensions"
            )

        # Python:  (query_count, dimension)
        # Fortran: (dimension, query_count)
        return np.asfortranarray(
            queries.T,
            dtype=np.float32,
        )

    def _validate_k(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")

        if k > self._vector_count:
            raise ValueError(
                "k cannot exceed the database size"
            )

    def _make_workload(
        self,
        *,
        operation: str,
        query_count: int,
        k: int,
    ) -> VectorWorkload:
        return VectorWorkload(
            metric=self._metric,
            operation=operation,
            dimension=self._dimension,
            vector_count=self._vector_count,
            query_count=query_count,
            k=k,
            available_memory_bytes=(
                get_available_memory_bytes()
                if operation == "batch"
                else None
            ),
        )

    @property
    def backend(self) -> str:
        """Return the backend requested during construction."""

        return self._backend

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def vector_count(self) -> int:
        return self._vector_count


class FlatIndexIP(_FlatIndexBase):
    """Exact inner-product vector-search index."""

    _metric = "ip"

    def _execute_single(
        self,
        query: np.ndarray,
        k: int,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray]:
        if decision.strategy == "serial":
            return _cpu.ip_search_serial(
                query,
                self._database,
                k,
            )

        if decision.strategy == "openmp":
            return _cpu.ip_search_openmp(
                query,
                self._database,
                k,
            )

        if decision.strategy == "blas_single":
            return _cpu.ip_search_blas(
                query,
                self._database,
                k,
            )

        raise RuntimeError(
            "Unsupported IP single-query strategy: "
            f"{decision.strategy}"
        )

    def _execute_repeated_batch(
        self,
        queries: np.ndarray,
        k: int,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray]:
        values_rows = []
        indices_rows = []

        for query_index in range(queries.shape[1]):
            query = np.ascontiguousarray(
                queries[:, query_index],
                dtype=np.float32,
            )

            if decision.strategy == "repeated_serial":
                values, indices = _cpu.ip_search_serial(
                    query,
                    self._database,
                    k,
                )
            elif decision.strategy == "repeated_openmp":
                values, indices = _cpu.ip_search_openmp(
                    query,
                    self._database,
                    k,
                )
            elif decision.strategy == "repeated_blas_single":
                values, indices = _cpu.ip_search_blas(
                    query,
                    self._database,
                    k,
                )
            else:
                raise RuntimeError(
                    "Unsupported repeated IP strategy: "
                    f"{decision.strategy}"
                )

            values_rows.append(values)
            indices_rows.append(indices)

        return (
            np.stack(values_rows),
            np.stack(indices_rows),
        )

    def _execute_batch(
        self,
        queries: np.ndarray,
        k: int,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray]:
        if decision.strategy in {
            "repeated_serial",
            "repeated_openmp",
            "repeated_blas_single",
        }:
            return self._execute_repeated_batch(
                queries,
                k,
                decision,
            )

        if decision.strategy == "blas_batch":
            values, indices = _cpu.ip_search_batch_blas(
                queries,
                self._database,
                k,
            )
            return values.T, indices.T

        raise RuntimeError(
            "Unsupported IP batch strategy: "
            f"{decision.strategy}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_k(k)
        query = self._prepare_query(query_vector)

        decision = resolve_vector_backend(
            self._backend,
            self._make_workload(
                operation="single",
                query_count=1,
                k=k,
            ),
        )

        if decision.backend == "fortran_blas":
            with apply_openblas_thread_policy(
                decision.thread_policy
            ):
                return self._execute_single(
                    query,
                    k,
                    decision,
                )

        if decision.backend == "fortran_openmp":
            with apply_openmp_thread_policy(
                decision.thread_policy
            ):
                return self._execute_single(
                    query,
                    k,
                    decision,
                )

        return self._execute_single(
            query,
            k,
            decision,
        )

    def search_batch(
        self,
        query_vectors: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_k(k)
        queries = self._prepare_query_batch(query_vectors)

        decision = resolve_vector_backend(
            self._backend,
            self._make_workload(
                operation="batch",
                query_count=int(queries.shape[1]),
                k=k,
            ),
        )

        if decision.backend == "fortran_blas":
            with apply_openblas_thread_policy(
                decision.thread_policy
            ):
                return self._execute_batch(
                    queries,
                    k,
                    decision,
                )

        if decision.backend == "fortran_openmp":
            with apply_openmp_thread_policy(
                decision.thread_policy
            ):
                return self._execute_batch(
                    queries,
                    k,
                    decision,
                )

        return self._execute_batch(
            queries,
            k,
            decision,
        )


class FlatIndexL2(_FlatIndexBase):
    """Exact squared-L2 vector-search index."""

    _metric = "l2"

    def _execute_single(
        self,
        query: np.ndarray,
        k: int,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray]:
        if decision.strategy == "serial":
            return _cpu.l2_search_serial(
                query,
                self._database,
                k,
            )

        if decision.strategy == "openmp":
            return _cpu.l2_search_openmp(
                query,
                self._database,
                k,
            )

        if decision.strategy == "blas_single":
            return _cpu.l2_search_blas(
                query,
                self._database,
                k,
            )

        raise RuntimeError(
            "Unsupported L2 single-query strategy: "
            f"{decision.strategy}"
        )

    def _execute_repeated_batch(
        self,
        queries: np.ndarray,
        k: int,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray]:
        values_rows = []
        indices_rows = []

        for query_index in range(queries.shape[1]):
            query = np.ascontiguousarray(
                queries[:, query_index],
                dtype=np.float32,
            )

            if decision.strategy == "repeated_serial":
                values, indices = _cpu.l2_search_serial(
                    query,
                    self._database,
                    k,
                )
            elif decision.strategy == "repeated_openmp":
                values, indices = _cpu.l2_search_openmp(
                    query,
                    self._database,
                    k,
                )
            else:
                raise RuntimeError(
                    "Unsupported repeated L2 strategy: "
                    f"{decision.strategy}"
                )

            values_rows.append(values)
            indices_rows.append(indices)

        return (
            np.stack(values_rows),
            np.stack(indices_rows),
        )

    def _execute_batch(
        self,
        queries: np.ndarray,
        k: int,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray]:
        if decision.strategy in {
            "repeated_serial",
            "repeated_openmp",
        }:
            return self._execute_repeated_batch(
                queries,
                k,
                decision,
            )

        if decision.strategy == "blas_batch":
            values, indices = _cpu.l2_search_batch_blas(
                queries,
                self._database,
                k,
            )
            return values.T, indices.T

        raise RuntimeError(
            "Unsupported L2 batch strategy: "
            f"{decision.strategy}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_k(k)
        query = self._prepare_query(query_vector)

        decision = resolve_vector_backend(
            self._backend,
            self._make_workload(
                operation="single",
                query_count=1,
                k=k,
            ),
        )

        if decision.backend == "fortran_blas":
            with apply_openblas_thread_policy(
                decision.thread_policy
            ):
                return self._execute_single(
                    query,
                    k,
                    decision,
                )

        if decision.backend == "fortran_openmp":
            with apply_openmp_thread_policy(
                decision.thread_policy
            ):
                return self._execute_single(
                    query,
                    k,
                    decision,
                )

        return self._execute_single(
            query,
            k,
            decision,
        )

    def search_batch(
        self,
        query_vectors: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_k(k)
        queries = self._prepare_query_batch(query_vectors)

        decision = resolve_vector_backend(
            self._backend,
            self._make_workload(
                operation="batch",
                query_count=int(queries.shape[1]),
                k=k,
            ),
        )

        if decision.backend == "fortran_blas":
            with apply_openblas_thread_policy(
                decision.thread_policy
            ):
                return self._execute_batch(
                    queries,
                    k,
                    decision,
                )

        if decision.backend == "fortran_openmp":
            with apply_openmp_thread_policy(
                decision.thread_policy
            ):
                return self._execute_batch(
                    queries,
                    k,
                    decision,
                )

        return self._execute_batch(
            queries,
            k,
            decision,
        )


class FlatIndexCosine(FlatIndexIP):
    """Exact cosine similarity through normalized inner product."""

    _metric = "cosine"

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

    def search_batch(
        self,
        query_vectors: np.ndarray,
        *,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_queries = normalize_database(
            query_vectors
        )

        return super().search_batch(
            normalized_queries,
            k=k,
        )
