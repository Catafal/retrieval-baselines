"""
BM25's k1 and b are pinned to the values the protocol defends, not merely to each other.

A mutation sweep found K1 and B could be set to any value with zero test failures.
`test_lexical_equivalence.py` compares the fast vectorised path against the naive reference
implementation, and both read the SAME module-level constants — so the two move together and
agree at every value. That test proves the two implementations match. It cannot prove they
match at the right constants, and it was the only thing looking.

This matters more here than it would in most codebases. 002's whole argument is that a
competently tuned BM25 is hard to beat, and the closure control that would catch a drifted k1
against the published Anserini figure only runs as a pipeline step against a real corpus, not
in the suite. Between commits, nothing held these two floats in place.

k1 = 1.2 and b = 0.75 are the standard Robertson/Sparck Jones defaults and the values BEIR's
published BM25 baselines use, which is what makes the closure control's comparison meaningful:
a different k1 would still produce a working retriever, just not the one the published number
describes.
"""
import pytest

from rb.experiments.ladder.retrievers import lexical


def test_k1_and_b_are_the_published_defaults():
    """KILLS: silently retuning BM25 into a baseline the published figure no longer describes."""
    assert lexical.K1 == 1.2
    assert lexical.B == 0.75


def test_the_constants_actually_reach_the_score():
    """
    The pin above is worthless if the scorer stopped reading the constants.

    Asserts the saturation and length-normalisation terms respond to k1 and b, so a refactor
    that hard-coded a value inside the scoring loop while leaving the module constants intact
    would fail here rather than pass both tests.
    """
    corpus = {
        "d1": "retrieval retrieval retrieval baseline",
        "d2": "retrieval baseline " + "filler " * 60,
        "d3": "unrelated document about something else entirely",
    }
    queries = {"q1": "retrieval baseline"}

    baseline = _scores(corpus, queries)

    k1_orig, b_orig = lexical.K1, lexical.B
    try:
        # Raising k1 weakens term-frequency saturation, so d1's triple occurrence must gain.
        lexical.K1 = 8.0
        raised_k1 = _scores(corpus, queries)
        # Dropping b to 0 removes length normalisation, so the long d2 must stop being penalised.
        lexical.K1, lexical.B = k1_orig, 0.0
        no_len_norm = _scores(corpus, queries)
    finally:
        lexical.K1, lexical.B = k1_orig, b_orig

    assert raised_k1["d1"] != baseline["d1"], "k1 does not reach the saturation term"
    assert no_len_norm["d2"] != baseline["d2"], "b does not reach the length-normalisation term"


def _scores(corpus, queries):
    r = lexical.LexicalRetriever(idf=True, tf_saturation=True, length_norm=True)
    return r.retrieve(corpus, queries, top_k=len(corpus))["q1"]
