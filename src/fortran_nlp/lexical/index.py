from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
from fortran_nlp import _cpu
from .preprocessing import (DEFAULT_SHINGLE_SIZE, encode_text_shingles, pack_corpus)

SUPPORTED_BACKENDS = {
    "auto",
    "fortran",
    "fortran_openmp",
}

@dataclass(frozen=True)
class JaccardResult:
    document_id: int
    score:float
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
            raise ValueError(f"Unsupported backend '{backend}'. Available backends: {', '.join(sorted(SUPPORTED_BACKENDS))}")
        if shingle_size < 1:
            raise ValueError(f"shingle_size must be at least 1")
        if backend == "auto":
            backend = "fortran_openmp"

        self._corpus_texts = tuple(corpus_texts)
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
        return self._backend
    @property
    def document_count(self) -> int:
        return len(self._corpus_texts)
    @property
    def shingle_size(self) -> int:
        return self._shingle_size

    def search(self, candidate_text: str, *, k: int | None=None, minimum_score: float=0.0, ) -> list[JaccardResult]:
        if k is not None and k < 1:
            raise ValueError("k must be at least 1")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        candidate_shingles = encode_text_shingles(candidate_text, shingle_size=self._shingle_size)

        if self._backend == "fortran":
            native_result = _cpu.jaccard_corpus_serial(candidate_shingles, self._corpus_shingles, self._document_offsets)
        elif self._backend == "fortran_openmp":
            native_result = _cpu.jaccard_corpus_openmp(candidate_shingles, self._corpus_shingles, self._document_offsets)
        else:
            raise RuntimeError(f"Unsupported backend state: {self._backend}")
        intersections, unions, scores = native_result
        ranked_indices = np.argsort(scores, kind="stable")[::-1]
        results = []
        for document_index in ranked_indices:
            score = float(scores[document_index])
            if score < minimum_score:
                continue
            results.append(
                JaccardResult(
                    document_id=int(document_index),
                    score=score,
                    intersection_count=int(intersections[document_index]),
                    union_count=int(unions[document_index]),
                )
            )
            if k is not None and len(results) >= k:
                break
        return results
    