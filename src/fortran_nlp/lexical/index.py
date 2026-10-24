from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fortran_nlp import _cpu
from fortran_nlp.auto import (
    BackendDecision,
    LexicalWorkload,
    resolve_lexical_backend,
)
from fortran_nlp.runtime import (
    apply_openmp_thread_policy,
    get_available_memory_bytes,
)

from .preprocessing import (
    DEFAULT_SHINGLE_SIZE,
    encode_text_shingles,
    pack_corpus,
)


SUPPORTED_BACKENDS = {
    "auto",
    "fortran",
    "fortran_openmp",
}


@dataclass(frozen=True)
class JaccardResult:
    document_id: int
    score: float
    intersection_count: int
    union_count: int


class JaccardIndex:
    """Exact word-shingle Jaccard search index."""

    def __init__(
        self,
        corpus_texts: Sequence[str],
        *,
        backend: str = "fortran",
        shingle_size: int = DEFAULT_SHINGLE_SIZE,
    ):
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                "Available backends: "
                f"{', '.join(sorted(SUPPORTED_BACKENDS))}"
            )

        if shingle_size < 1:
            raise ValueError(
                "shingle_size must be at least 1"
            )

        self._corpus_texts = tuple(corpus_texts)

        # Preserve the user's original request, including "auto".
        self._backend = backend
        self._shingle_size = shingle_size

        (
            self._corpus_shingles,
            self._document_offsets,
        ) = pack_corpus(
            self._corpus_texts,
            shingle_size=self._shingle_size,
        )

    @property
    def backend(self) -> str:
        """Return the backend requested during construction."""

        return self._backend

    @property
    def document_count(self) -> int:
        return len(self._corpus_texts)

    @property
    def shingle_size(self) -> int:
        return self._shingle_size

    @property
    def total_corpus_shingles(self) -> int:
        return int(self._corpus_shingles.size)

    def _rank_results(
        self,
        intersections: np.ndarray,
        unions: np.ndarray,
        scores: np.ndarray,
        *,
        k: int | None,
        minimum_score: float,
    ) -> list[JaccardResult]:
        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )

        results = []

        for document_index in ranked_indices:
            score = float(scores[document_index])

            if score < minimum_score:
                continue

            results.append(
                JaccardResult(
                    document_id=int(document_index),
                    score=score,
                    intersection_count=int(
                        intersections[document_index]
                    ),
                    union_count=int(
                        unions[document_index]
                    ),
                )
            )

            if k is not None and len(results) >= k:
                break

        return results

    def _execute_single(
        self,
        candidate_shingles: np.ndarray,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute one candidate using the resolved strategy."""

        if decision.strategy == "serial":
            return _cpu.jaccard_corpus_serial(
                candidate_shingles,
                self._corpus_shingles,
                self._document_offsets,
            )

        if decision.strategy == "openmp":
            return _cpu.jaccard_corpus_openmp(
                candidate_shingles,
                self._corpus_shingles,
                self._document_offsets,
            )

        raise RuntimeError(
            "Unsupported lexical single-query strategy: "
            f"{decision.strategy}"
        )

    def _execute_repeated_batch(
        self,
        candidate_shingles: np.ndarray,
        candidate_offsets: np.ndarray,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute batched search through repeated single-query calls."""

        candidate_count = candidate_offsets.size - 1

        intersection_columns = []
        union_columns = []
        score_columns = []

        for candidate_index in range(candidate_count):
            start = int(candidate_offsets[candidate_index])
            end = int(candidate_offsets[candidate_index + 1])

            candidate = candidate_shingles[start:end]

            if decision.strategy == "repeated_serial":
                native_result = _cpu.jaccard_corpus_serial(
                    candidate,
                    self._corpus_shingles,
                    self._document_offsets,
                )

            elif decision.strategy == "repeated_openmp":
                native_result = _cpu.jaccard_corpus_openmp(
                    candidate,
                    self._corpus_shingles,
                    self._document_offsets,
                )

            else:
                raise RuntimeError(
                    "Unsupported repeated lexical strategy: "
                    f"{decision.strategy}"
                )

            intersections, unions, scores = native_result

            intersection_columns.append(intersections)
            union_columns.append(unions)
            score_columns.append(scores)

        return (
            np.column_stack(intersection_columns),
            np.column_stack(union_columns),
            np.column_stack(score_columns),
        )

    def _execute_batch(
        self,
        candidate_shingles: np.ndarray,
        candidate_offsets: np.ndarray,
        decision: BackendDecision,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute a resolved lexical batch strategy."""

        if decision.strategy in {
            "repeated_serial",
            "repeated_openmp",
        }:
            return self._execute_repeated_batch(
                candidate_shingles,
                candidate_offsets,
                decision,
            )

        if decision.strategy == "native_batch_openmp":
            return _cpu.jaccard_batch_corpus(
                candidate_shingles,
                candidate_offsets,
                self._corpus_shingles,
                self._document_offsets,
            )

        raise RuntimeError(
            "Unsupported lexical batch strategy: "
            f"{decision.strategy}"
        )

    def search(
        self,
        candidate_text: str,
        *,
        k: int | None = None,
        minimum_score: float = 0.0, ) -> list[JaccardResult]:
        if k is not None and k < 1:
            raise ValueError("k must be at least 1")

        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0 and 1"
            )

        candidate_shingles = encode_text_shingles(
            candidate_text,
            shingle_size=self._shingle_size,
        )

        workload = LexicalWorkload(
            operation="single",
            document_count=self.document_count,
            total_corpus_shingles=self.total_corpus_shingles,
            candidate_count=1,
            candidate_shingle_counts=(
                int(candidate_shingles.size),
            ),
        )

        decision = resolve_lexical_backend(
            self._backend,
            workload,
        )

        with apply_openmp_thread_policy(decision.thread_policy):
            intersections, unions, scores = self._execute_single(
                candidate_shingles,
                decision,
            )

        return self._rank_results(
            intersections,
            unions,
            scores,
            k=k,
            minimum_score=minimum_score,
        )

    def search_batch(
        self,
        candidate_texts: Sequence[str],
        *,
        k: int | None = None,
        minimum_score: float = 0.0,
    ) -> list[list[JaccardResult]]:
        if k is not None and k < 1:
            raise ValueError("k must be at least 1")

        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0 and 1"
            )

        candidates = tuple(candidate_texts)

        if len(candidates) == 0:
            raise ValueError(
                "candidate_texts must contain at least one candidate"
            )

        (
            candidate_shingles,
            candidate_offsets,
        ) = pack_corpus(
            candidates,
            shingle_size=self._shingle_size,
        )

        candidate_shingle_counts = tuple(
            int(length)
            for length in np.diff(candidate_offsets)
        )

        workload = LexicalWorkload(
            operation="batch",
            document_count=self.document_count,
            total_corpus_shingles=self.total_corpus_shingles,
            candidate_count=len(candidates),
            candidate_shingle_counts=(
                candidate_shingle_counts
            ),
            available_memory_bytes=get_available_memory_bytes(),
        )

        decision = resolve_lexical_backend(
            self._backend,
            workload,
        )

        with apply_openmp_thread_policy(decision.thread_policy):
            (
                intersection_matrix,
                union_matrix,
                score_matrix,
            ) = self._execute_batch(
                candidate_shingles,
                candidate_offsets,
                decision,
            )

        batch_results = []

        for candidate_index in range(len(candidates)):
            candidate_results = self._rank_results(
                intersection_matrix[:, candidate_index],
                union_matrix[:, candidate_index],
                score_matrix[:, candidate_index],
                k=k,
                minimum_score=minimum_score,
            )

            batch_results.append(candidate_results)

        return batch_results