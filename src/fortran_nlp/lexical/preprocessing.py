import hashlib
import re
import unicodedata
from collections.abc import Sequence
import numpy as np

DEFAULT_SHINGLE_SIZE = 3

NON_WORD_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)

def normalize_text(text: str) -> str:
    """Normalize text before tokenization."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = NON_WORD_PATTERN.sub(" ", normalized)
    return " ".join(normalized.split())

def tokenize_text(text: str) -> list[str]:
    """Normalize and split text into tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()

def hash_shingle(shingle: str) -> np.int64:
    """Generate a deterministic signed 64-bit shingle hash."""
    digest = hashlib.blake2b(
        shingle.encode("utf-8"),
        digest_size=8,
    ).digest()
    return np.int64(
        int.from_bytes(
            digest,
            byteorder="little",
            signed=True,
        )
    )

def encode_text_shingles(text: str, shingle_size: int = DEFAULT_SHINGLE_SIZE) -> np.ndarray:
    """Convert text into sorted, unique int64 shingle hashes."""
    if shingle_size < 1:
        raise ValueError("shingle_size must be at least 1")
    tokens = tokenize_text(text)
    if len(tokens) < shingle_size:
        return np.empty(0, dtype=np.int64)
    hashes = [
        hash_shingle(
            " ".join(tokens[index:index + shingle_size])
        )
        for index in range(
            len(tokens) - shingle_size + 1
        )
    ]
    return np.unique(np.asarray(hashes, dtype=np.int64))

def pack_corpus(corpus_texts: Sequence[str], shingle_size: int = DEFAULT_SHINGLE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode and pack corpus documents.
    Returns:
        corpus_shingles:
            Flattened int64 shingle buffer.
        document_offsets:
            Zero-based, end-exclusive document boundaries.
    """
    document_shingles = [
        encode_text_shingles(
            text,
            shingle_size=shingle_size,
        )
        for text in corpus_texts
    ]
    document_lengths = np.asarray(
        [len(shingles) for shingles in document_shingles],
        dtype=np.int64,
    )
    document_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(document_lengths, dtype=np.int64),
        )
    )
    nonempty_documents = [
        shingles
        for shingles in document_shingles
        if shingles.size > 0
    ]
    if nonempty_documents:
        corpus_shingles = np.concatenate(
            nonempty_documents
        )
    else:
        corpus_shingles = np.empty(
            0,
            dtype=np.int64,
        )
    return (np.ascontiguousarray(corpus_shingles), np.ascontiguousarray(document_offsets))