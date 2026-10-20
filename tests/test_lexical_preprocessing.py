import numpy as np

from fortran_nlp.lexical.preprocessing import (
    encode_text_shingles,
    normalize_text,
    pack_corpus,
)


def test_normalization():
    assert normalize_text("  Hello, WORLD!  ") == "hello world"


def test_deterministic_shingles():
    first = encode_text_shingles(
        "one two three four five"
    )

    second = encode_text_shingles(
        "ONE two, three four five!"
    )

    assert np.array_equal(first, second)
    assert first.dtype == np.int64
    assert np.all(first[:-1] <= first[1:])


def test_corpus_packing():
    corpus, offsets = pack_corpus(
        [
            "one two three four",
            "alpha beta gamma delta",
            "too short",
        ]
    )

    assert corpus.dtype == np.int64
    assert offsets.dtype == np.int64
    assert np.array_equal(offsets, [0, 2, 4, 4])
    assert offsets[-1] == len(corpus)