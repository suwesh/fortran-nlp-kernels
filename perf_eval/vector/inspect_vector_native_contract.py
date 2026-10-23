from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fortran_nlp import _cpu


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = (
    REPOSITORY_ROOT
    / "perf_eval/data/packed/vector_real_database.npz"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and validate the published native vector-search "
            "function contracts using the packed real vector database."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )
    return parser.parse_args()


def describe_result(
    name: str,
    result: tuple,
) -> None:
    print(name)
    print("-" * 72)
    print("Tuple length:", len(result))

    for index, array in enumerate(result):
        print(
            f"result[{index}]: "
            f"shape={array.shape}, "
            f"dtype={array.dtype}, "
            f"C={array.flags.c_contiguous}, "
            f"F={array.flags.f_contiguous}"
        )
        print(array)

    print()


def assert_same_single_result(
    expected: tuple,
    actual: tuple,
    *,
    name: str,
) -> None:
    if len(expected) != 2 or len(actual) != 2:
        raise AssertionError(
            f"{name} must return a two-array tuple"
        )

    if not np.allclose(
        expected[0],
        actual[0],
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise AssertionError(
            f"{name} score/distance values differ"
        )

    if not np.array_equal(expected[1], actual[1]):
        raise AssertionError(
            f"{name} top-k IDs differ"
        )


def main() -> None:
    arguments = parse_arguments()

    if arguments.query_count < 1:
        raise ValueError(
            "query-count must be at least 1"
        )

    database_path = (
        arguments.database
        .expanduser()
        .resolve()
    )

    with np.load(
        database_path,
        allow_pickle=False,
    ) as packed:
        database_rows = np.ascontiguousarray(
            packed["database_vectors"],
            dtype=np.float32,
        )

    vector_count, dimension = database_rows.shape

    if arguments.query_count > vector_count:
        raise ValueError(
            "query-count cannot exceed the database size"
        )

    if not 1 <= arguments.k <= vector_count:
        raise ValueError(
            "k must be between 1 and the database size"
        )

    # Native ABI layout:
    # database_vectors -> (dimension, vector_count), Fortran contiguous
    # query_vector     -> (dimension,), C contiguous
    # query_vectors    -> (dimension, query_count), Fortran contiguous
    database_f = np.asfortranarray(
        database_rows.T,
        dtype=np.float32,
    )

    query = np.ascontiguousarray(
        database_rows[0],
        dtype=np.float32,
    )

    queries_f = np.asfortranarray(
        database_rows[: arguments.query_count].T,
        dtype=np.float32,
    )

    print("Native vector-search contract inspection")
    print("=" * 72)
    print("Database row-major source shape:", database_rows.shape)
    print("Native database shape:", database_f.shape)
    print("Native database C-contiguous:", database_f.flags.c_contiguous)
    print("Native database F-contiguous:", database_f.flags.f_contiguous)
    print("Single query shape:", query.shape)
    print("Single query C-contiguous:", query.flags.c_contiguous)
    print("Query batch shape:", queries_f.shape)
    print("Query batch F-contiguous:", queries_f.flags.f_contiguous)
    print("k:", arguments.k)
    print("=" * 72)
    print()

    single_functions = [
        (
            "ip_search_serial",
            lambda: _cpu.ip_search_serial(
                query,
                database_f,
                arguments.k,
            ),
        ),
        (
            "ip_search_openmp",
            lambda: _cpu.ip_search_openmp(
                query,
                database_f,
                arguments.k,
            ),
        ),
        (
            "ip_search_blas",
            lambda: _cpu.ip_search_blas(
                query,
                database_f,
                arguments.k,
            ),
        ),
        (
            "l2_search_serial",
            lambda: _cpu.l2_search_serial(
                query,
                database_f,
                arguments.k,
            ),
        ),
        (
            "l2_search_openmp",
            lambda: _cpu.l2_search_openmp(
                query,
                database_f,
                arguments.k,
            ),
        ),
        (
            "l2_search_blas",
            lambda: _cpu.l2_search_blas(
                query,
                database_f,
                arguments.k,
            ),
        ),
    ]

    single_results: dict[str, tuple] = {}

    for name, function in single_functions:
        result = function()
        single_results[name] = result
        describe_result(name, result)

    assert_same_single_result(
        single_results["ip_search_serial"],
        single_results["ip_search_openmp"],
        name="IP serial/OpenMP",
    )
    assert_same_single_result(
        single_results["ip_search_serial"],
        single_results["ip_search_blas"],
        name="IP serial/BLAS",
    )
    assert_same_single_result(
        single_results["l2_search_serial"],
        single_results["l2_search_openmp"],
        name="L2 serial/OpenMP",
    )
    assert_same_single_result(
        single_results["l2_search_serial"],
        single_results["l2_search_blas"],
        name="L2 serial/BLAS",
    )

    batch_functions = [
        (
            "ip_search_batch_blas",
            lambda: _cpu.ip_search_batch_blas(
                queries_f,
                database_f,
                arguments.k,
            ),
        ),
        (
            "l2_search_batch_blas",
            lambda: _cpu.l2_search_batch_blas(
                queries_f,
                database_f,
                arguments.k,
            ),
        ),
    ]

    batch_results: dict[str, tuple] = {}

    for name, function in batch_functions:
        result = function()
        batch_results[name] = result
        describe_result(name, result)

    print("=" * 72)
    print("Single-query serial/OpenMP/BLAS equivalence: PASS")
    print("Batch native call execution: PASS")
    print("Native vector-search contract inspection: PASS")


if __name__ == "__main__":
    main()
