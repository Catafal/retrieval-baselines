"""Unit tests for the pieces of grep-baseline-v1 that can silently corrupt a rank."""

from pathlib import Path

import pytest

from rb.grep_baseline import materialise, rank, run_query, tokenize
from rb.metrics import set_recall


def test_tokenize_drops_stopwords_and_duplicates():
    assert tokenize("What is the effect of the drug on the drug?") == ["effect", "drug"]


def test_tokenize_empty_when_all_stopwords():
    assert tokenize("what is it") == []


def test_materialise_collapses_newlines(tmp_path: Path):
    """A document containing a newline must not become two lines, or every later rank shifts."""
    corpus = {"a": "first\nline", "b": "second\tdoc"}
    path = tmp_path / "c.txt"
    doc_ids = materialise(corpus, path)
    assert doc_ids == ["a", "b"]
    assert path.read_text().splitlines() == ["first line", "second doc"]


def test_materialise_detects_broken_invariant(tmp_path: Path):
    path = tmp_path / "c.txt"
    path.write_text("one\ntwo\nthree\n")
    with pytest.raises(RuntimeError, match="one-doc-per-line"):
        materialise({"a": "one", "b": "two"}, path)


def test_rank_is_deterministic_on_ties():
    """Equal scores must order by document id, so two runs agree exactly."""
    doc_ids = ["z", "y", "x"]
    hits = {1: (2, 5), 2: (2, 5), 3: (2, 5)}
    assert [d for d, _ in rank(hits, doc_ids)] == ["x", "y", "z"]


def test_rank_prefers_more_distinct_terms_over_more_matches():
    doc_ids = ["a", "b"]
    hits = {1: (1, 99), 2: (2, 2)}
    assert [d for d, _ in rank(hits, doc_ids)][0] == "b"


def test_empty_query_retrieves_nothing(tmp_path: Path):
    corpus = {"a": "some text here"}
    path = tmp_path / "c.txt"
    doc_ids = materialise(corpus, path)
    results, hits, _, terms = run_query("what is it", path, doc_ids)
    assert terms == [] and results == [] and hits == {}


def test_word_boundary_matching(tmp_path: Path):
    """insulin must not match insulinoma under the pre-registered word-bounded rule."""
    corpus = {"a": "insulinoma is a tumour", "b": "insulin lowers glucose"}
    path = tmp_path / "c.txt"
    doc_ids = materialise(corpus, path)
    results, hits, _, _ = run_query("insulin", path, doc_ids)
    assert [d for d, _ in results] == ["b"]
    assert len(hits) == 1, "word-bounded matching must not return the insulinoma document"


def test_set_recall():
    assert set_recall({"a": 1, "b": 1}, {"a", "z"}) == 0.5
    assert set_recall({}, {"a"}) == 0.0


def test_rank_scores_are_strictly_decreasing():
    """
    trec_eval sorts by score, so equal scores let IT choose the order, not us.

    The pre-registered tie-break is document id. That only holds if the scores handed to
    the scorer are strictly decreasing in our own rank order.
    """
    doc_ids = ["a", "b", "c"]
    hits = {1: (2, 5), 2: (2, 5), 3: (2, 5)}
    scores = [s for _, s in rank(hits, doc_ids)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)
