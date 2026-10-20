import os

import numpy as np

from fortran_nlp import _cpu


candidate = np.array(
    [10, 20, 40, 70],
    dtype=np.int64,
)

existing_documents = [
    np.array([10, 20, 40], dtype=np.int64),
    np.array([15, 20], dtype=np.int64),
    np.array([10, 30, 60, 70], dtype=np.int64),
]

corpus_shingles = np.concatenate(existing_documents)

document_lengths = np.array(
    [len(document) for document in existing_documents],
    dtype=np.int64,
)

document_offsets = np.concatenate(
    (
        np.array([0], dtype=np.int64),
        np.cumsum(document_lengths, dtype=np.int64),
    )
)


serial_result = _cpu.jaccard_corpus_serial(
    candidate,
    corpus_shingles,
    document_offsets,
)

openmp_result = _cpu.jaccard_corpus_openmp(
    candidate,
    corpus_shingles,
    document_offsets,
)


expected_intersections = np.array(
    [3, 1, 2],
    dtype=np.int64,
)

expected_unions = np.array(
    [4, 5, 6],
    dtype=np.int64,
)

expected_scores = np.array(
    [0.75, 0.20, 1.0 / 3.0],
    dtype=np.float64,
)


assert np.array_equal(
    serial_result[0],
    expected_intersections,
)

assert np.array_equal(
    serial_result[1],
    expected_unions,
)

assert np.allclose(
    serial_result[2],
    expected_scores,
    rtol=0.0,
    atol=1e-15,
)


assert np.array_equal(
    serial_result[0],
    openmp_result[0],
)

assert np.array_equal(
    serial_result[1],
    openmp_result[1],
)

assert np.allclose(
    serial_result[2],
    openmp_result[2],
    rtol=0.0,
    atol=1e-15,
)


print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("Offsets:", document_offsets)

print("\nSerial results:")
print("Intersections:", serial_result[0])
print("Unions:", serial_result[1])
print("Scores:", serial_result[2])

print("\nOpenMP results:")
print("Intersections:", openmp_result[0])
print("Unions:", openmp_result[1])
print("Scores:", openmp_result[2])

print("\nExpected results: PASS")
print("Python/native serial/OpenMP equivalence: PASS")