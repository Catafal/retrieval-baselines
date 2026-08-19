"""
The four pre-registered query properties (protocols/002-amendment-1-dense.md
section 7), checked against hand-computed values so a future edit cannot
silently change what "query length" or "IDF" means here without a test
noticing.
"""

import pytest

from rb.experiments.ladder.query_properties import (
    compute_query_properties,
    gold_count,
    query_gold_jaccard,
    query_idf,
    query_length,
)
from rb.experiments.ladder.retrievers.lexical import build_index


def test_query_length_uses_the_scorers_tokenizer():
    # "Hello, World!!" tokenizes to ["hello", "world"] under _tokenize
    # (lowercase, split on non-alphanumerics) — a whitespace split would
    # instead count "World!!" as one token.
    assert query_length("Hello, World!!") == 2


def test_query_length_counts_repeats_not_distinct_terms():
    assert query_length("alpha alpha alpha") == 3


def test_query_idf_empty_query_is_zero():
    index = build_index({"d1": "alpha beta"})
    assert query_idf("", index) == {"max_idf": 0.0, "mean_idf": 0.0}


def test_query_idf_rare_term_scores_higher_than_common_term():
    # "alpha" appears in every document (df=3), "zeta" in none (df=0, OOV).
    corpus = {f"d{i}": "alpha beta gamma" for i in range(3)}
    index = build_index(corpus)
    common = query_idf("alpha", index)
    rare = query_idf("zeta", index)
    assert rare["max_idf"] > common["max_idf"]


def test_query_idf_max_and_mean_over_distinct_terms_only():
    """A repeated term must not be double-counted in the mean — IDF is a
    per-term-TYPE value, matching the lexical scorer's own outer sum over
    `sorted(set(_tokenize(query)))`."""
    corpus = {"d1": "alpha beta", "d2": "alpha"}
    index = build_index(corpus)
    once = query_idf("alpha beta", index)
    repeated = query_idf("alpha alpha beta", index)
    assert once == repeated


def test_query_gold_jaccard_identical_texts_is_one():
    corpus = {"gold_doc": "alpha beta gamma"}
    assert query_gold_jaccard("alpha beta gamma", ["gold_doc"], corpus) == pytest.approx(1.0)


def test_query_gold_jaccard_disjoint_texts_is_zero():
    corpus = {"gold_doc": "delta epsilon"}
    assert query_gold_jaccard("alpha beta", ["gold_doc"], corpus) == pytest.approx(0.0)


def test_query_gold_jaccard_averages_across_multiple_gold_documents():
    corpus = {"g1": "alpha beta", "g2": "delta epsilon"}
    # query overlaps fully with g1 (jaccard=1.0) and not at all with g2 (0.0)
    result = query_gold_jaccard("alpha beta", ["g1", "g2"], corpus)
    assert result == pytest.approx(0.5)


def test_query_gold_jaccard_no_gold_documents_is_zero():
    assert query_gold_jaccard("alpha", [], {}) == 0.0


def test_gold_count_is_len_of_gold_ids():
    assert gold_count(["a", "b", "c"]) == 3
    assert gold_count([]) == 0


def test_compute_query_properties_returns_all_four_fields():
    corpus = {"gold_doc": "alpha beta", "other": "gamma delta"}
    index = build_index(corpus)
    props = compute_query_properties("alpha beta gamma", ["gold_doc"], corpus, index)
    assert set(props) == {"query_length", "max_idf", "mean_idf", "gold_jaccard", "gold_count"}
    assert props["query_length"] == 3
    assert props["gold_count"] == 1
