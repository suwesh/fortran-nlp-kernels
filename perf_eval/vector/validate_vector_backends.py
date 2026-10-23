from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import fortran_nlp
from fortran_nlp import _cpu


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = (
    REPOSITORY_ROOT
    / "perf_eval/data/packed/vector_real_database.npz"
)

QUERY_INDICES = (0, 17, 67, 103, 208, 351, 474)
K_VALUES = (1, 5, 10, 50)
SCORE_RTOL = 1.0e-5
SCORE_ATOL = 2.0e-5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate real-vector IP, L2, cosine, SGEMV, and SGEMM "
            "results across the published native CPU backends."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
    )
    return parser.parse_args()


def load_database(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as packed:
        database_rows = np.ascontiguousarray(
            packed["database_vectors"],
            dtype=np.float32,
        )

    if database_rows.ndim != 2:
        raise AssertionError("database_vectors must be rank two")
    if database_rows.dtype != np.float32:
        raise AssertionError("database_vectors must use float32")
    if not database_rows.flags.c_contiguous:
        raise AssertionError("row-major database must be C-contiguous")
    if not np.isfinite(database_rows).all():
        raise AssertionError("database contains NaN or infinity")

    database_native = np.asfortranarray(
        database_rows.T,
        dtype=np.float32,
    )

    return database_rows, database_native


def numpy_ip_reference(
    query: np.ndarray,
    database_rows: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        database_rows @ query,
        dtype=np.float32,
    )


def numpy_l2_reference(
    query: np.ndarray,
    database_rows: np.ndarray,
) -> np.ndarray:
    differences = database_rows - query
    return np.asarray(
        np.einsum("ij,ij->i", differences, differences),
        dtype=np.float32,
    )


def assert_topk_valid(
    *,
    values: np.ndarray,
    ids: np.ndarray,
    reference: np.ndarray,
    k: int,
    largest: bool,
    description: str,
) -> None:
    if values.shape != (k,):
        raise AssertionError(
            f"{description}: expected values shape {(k,)}, got {values.shape}"
        )
    if ids.shape != (k,):
        raise AssertionError(
            f"{description}: expected IDs shape {(k,)}, got {ids.shape}"
        )
    if values.dtype != np.float32:
        raise AssertionError(f"{description}: values must use float32")
    if ids.dtype != np.int64:
        raise AssertionError(f"{description}: IDs must use int64")
    if np.unique(ids).size != k:
        raise AssertionError(f"{description}: duplicate top-k IDs")
    if np.any(ids < 0) or np.any(ids >= reference.size):
        raise AssertionError(f"{description}: out-of-range top-k ID")

    referenced_values = reference[ids]
    if not np.allclose(
        values,
        referenced_values,
        rtol=SCORE_RTOL,
        atol=SCORE_ATOL,
    ):
        maximum_error = float(np.max(np.abs(values - referenced_values)))
        raise AssertionError(
            f"{description}: returned values do not match returned IDs; "
            f"maximum error={maximum_error}"
        )

    differences = np.diff(values)
    if largest:
        if np.any(differences > SCORE_ATOL):
            raise AssertionError(f"{description}: values are not descending")
        threshold = float(values[-1])
        excluded = np.ones(reference.size, dtype=bool)
        excluded[ids] = False
        if np.any(reference[excluded] > threshold + SCORE_ATOL):
            raise AssertionError(
                f"{description}: a better IP score was excluded from top-k"
            )
    else:
        if np.any(differences < -SCORE_ATOL):
            raise AssertionError(f"{description}: distances are not ascending")
        threshold = float(values[-1])
        excluded = np.ones(reference.size, dtype=bool)
        excluded[ids] = False
        if np.any(reference[excluded] < threshold - SCORE_ATOL):
            raise AssertionError(
                f"{description}: a better L2 distance was excluded from top-k"
            )


def assert_backend_equivalence(
    expected: tuple[np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray],
    *,
    description: str,
) -> None:
    if not np.array_equal(expected[1], actual[1]):
        raise AssertionError(f"{description}: top-k IDs differ")
    if not np.allclose(
        expected[0],
        actual[0],
        rtol=SCORE_RTOL,
        atol=SCORE_ATOL,
    ):
        maximum_error = float(np.max(np.abs(expected[0] - actual[0])))
        raise AssertionError(
            f"{description}: values differ; maximum error={maximum_error}"
        )


def validate_single_queries(
    database_rows: np.ndarray,
    database_native: np.ndarray,
    query_indices: tuple[int, ...],
) -> None:
    for query_index in query_indices:
        query = np.ascontiguousarray(
            database_rows[query_index],
            dtype=np.float32,
        )
        ip_reference = numpy_ip_reference(query, database_rows)
        l2_reference = numpy_l2_reference(query, database_rows)

        for k in K_VALUES:
            ip_serial = _cpu.ip_search_serial(query, database_native, k)
            ip_openmp = _cpu.ip_search_openmp(query, database_native, k)
            ip_blas = _cpu.ip_search_blas(query, database_native, k)

            l2_serial = _cpu.l2_search_serial(query, database_native, k)
            l2_openmp = _cpu.l2_search_openmp(query, database_native, k)
            l2_blas = _cpu.l2_search_blas(query, database_native, k)

            for name, result in (
                ("IP serial", ip_serial),
                ("IP OpenMP", ip_openmp),
                ("IP SGEMV", ip_blas),
            ):
                assert_topk_valid(
                    values=result[0],
                    ids=result[1],
                    reference=ip_reference,
                    k=k,
                    largest=True,
                    description=f"{name}, query={query_index}, k={k}",
                )

            for name, result in (
                ("L2 serial", l2_serial),
                ("L2 OpenMP", l2_openmp),
                ("L2 BLAS", l2_blas),
            ):
                assert_topk_valid(
                    values=result[0],
                    ids=result[1],
                    reference=l2_reference,
                    k=k,
                    largest=False,
                    description=f"{name}, query={query_index}, k={k}",
                )

            assert_backend_equivalence(
                ip_serial,
                ip_openmp,
                description=f"IP serial/OpenMP, query={query_index}, k={k}",
            )
            assert_backend_equivalence(
                ip_serial,
                ip_blas,
                description=f"IP serial/SGEMV, query={query_index}, k={k}",
            )
            assert_backend_equivalence(
                l2_serial,
                l2_openmp,
                description=f"L2 serial/OpenMP, query={query_index}, k={k}",
            )
            assert_backend_equivalence(
                l2_serial,
                l2_blas,
                description=f"L2 serial/BLAS, query={query_index}, k={k}",
            )

            if not np.array_equal(ip_serial[1], l2_serial[1]):
                raise AssertionError(
                    f"IP/L2 ranking differs for normalized query "
                    f"{query_index}, k={k}"
                )

            if k == 1:
                if int(ip_serial[1][0]) != query_index:
                    raise AssertionError(
                        f"IP self-match failed for query {query_index}"
                    )
                if int(l2_serial[1][0]) != query_index:
                    raise AssertionError(
                        f"L2 self-match failed for query {query_index}"
                    )

        print(f"Single query {query_index}: PASS")


def validate_batch_queries(
    database_rows: np.ndarray,
    database_native: np.ndarray,
    query_indices: tuple[int, ...],
) -> None:
    query_rows = np.ascontiguousarray(
        database_rows[list(query_indices)],
        dtype=np.float32,
    )
    queries_native = np.asfortranarray(
        query_rows.T,
        dtype=np.float32,
    )
    query_count = len(query_indices)

    for k in K_VALUES:
        ip_batch = _cpu.ip_search_batch_blas(
            queries_native,
            database_native,
            k,
        )
        l2_batch = _cpu.l2_search_batch_blas(
            queries_native,
            database_native,
            k,
        )

        expected_shape = (k, query_count)
        for name, result in (
            ("IP SGEMM", ip_batch),
            ("L2 SGEMM", l2_batch),
        ):
            if result[0].shape != expected_shape:
                raise AssertionError(
                    f"{name}: expected values shape {expected_shape}, "
                    f"got {result[0].shape}"
                )
            if result[1].shape != expected_shape:
                raise AssertionError(
                    f"{name}: expected IDs shape {expected_shape}, "
                    f"got {result[1].shape}"
                )

        for column, query_index in enumerate(query_indices):
            query = np.ascontiguousarray(
                database_rows[query_index],
                dtype=np.float32,
            )
            ip_reference = numpy_ip_reference(query, database_rows)
            l2_reference = numpy_l2_reference(query, database_rows)

            ip_column = (
                np.ascontiguousarray(ip_batch[0][:, column]),
                np.ascontiguousarray(ip_batch[1][:, column]),
            )
            l2_column = (
                np.ascontiguousarray(l2_batch[0][:, column]),
                np.ascontiguousarray(l2_batch[1][:, column]),
            )

            assert_topk_valid(
                values=ip_column[0],
                ids=ip_column[1],
                reference=ip_reference,
                k=k,
                largest=True,
                description=f"IP SGEMM, query={query_index}, k={k}",
            )
            assert_topk_valid(
                values=l2_column[0],
                ids=l2_column[1],
                reference=l2_reference,
                k=k,
                largest=False,
                description=f"L2 SGEMM, query={query_index}, k={k}",
            )

            ip_sgemv = _cpu.ip_search_blas(query, database_native, k)
            l2_sgemv = _cpu.l2_search_blas(query, database_native, k)

            assert_backend_equivalence(
                ip_sgemv,
                ip_column,
                description=f"IP SGEMV/SGEMM, query={query_index}, k={k}",
            )
            assert_backend_equivalence(
                l2_sgemv,
                l2_column,
                description=f"L2 SGEMV/SGEMM, query={query_index}, k={k}",
            )

            if not np.array_equal(ip_column[1], l2_column[1]):
                raise AssertionError(
                    f"Batched IP/L2 rankings differ for query "
                    f"{query_index}, k={k}"
                )

        print(f"Batch k={k}: PASS")


def main() -> None:
    arguments = parse_arguments()
    database_path = arguments.database.expanduser().resolve()

    database_rows, database_native = load_database(database_path)
    vector_count, dimension = database_rows.shape

    query_indices = tuple(
        index for index in QUERY_INDICES if index < vector_count
    )

    if not query_indices:
        raise ValueError("No configured query index fits the database")
    if max(K_VALUES) > vector_count:
        raise ValueError("A configured k exceeds the database size")

    print("Real vector backend validation")
    print("=" * 72)
    print(f"fortran-nlp-kernels: {fortran_nlp.__version__}")
    print(f"Database vectors: {vector_count}")
    print(f"Dimension: {dimension}")
    print(f"Queries: {query_indices}")
    print(f"k values: {K_VALUES}")
    print("=" * 72)

    validate_single_queries(
        database_rows,
        database_native,
        query_indices,
    )

    print("-" * 72)

    validate_batch_queries(
        database_rows,
        database_native,
        query_indices,
    )

    print("=" * 72)
    print("IP NumPy/serial/OpenMP/SGEMV equivalence: PASS")
    print("L2 NumPy/serial/OpenMP/BLAS equivalence: PASS")
    print("SGEMM/repeated-SGEMV equivalence: PASS")
    print("Normalized IP/L2 ranking relationship: PASS")
    print("Real vector backend validation: PASS")


if __name__ == "__main__":
    main()
