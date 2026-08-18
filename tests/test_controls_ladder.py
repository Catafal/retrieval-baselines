"""Experiment 002's two new controls, tested as functions against known numbers
(the embedding-shuffle control is also exercised end to end in test_dense.py)."""

import pytest

from rb.controls import bm25_closure, embedding_shuffle


def test_bm25_closure_passes_within_tolerance():
    result = bm25_closure(our_ndcg=0.665, published_ndcg=0.660, tolerance=0.02)
    assert result["passed"]
    assert result["absolute_difference"] == 0.005


def test_bm25_closure_fails_outside_tolerance():
    result = bm25_closure(our_ndcg=0.700, published_ndcg=0.665, tolerance=0.02)
    assert not result["passed"]


def test_bm25_closure_boundary_is_inclusive():
    result = bm25_closure(our_ndcg=0.520, published_ndcg=0.500, tolerance=0.02)
    assert result["passed"], "difference exactly equal to tolerance must pass"


def test_embedding_shuffle_passes_when_shuffled_ndcg_below_ceiling():
    result = embedding_shuffle(normal_ndcg=0.60, shuffled_ndcg=0.05, chance_ceiling=0.15)
    assert result["passed"]


def test_embedding_shuffle_fails_when_shuffled_ndcg_does_not_collapse():
    result = embedding_shuffle(normal_ndcg=0.60, shuffled_ndcg=0.55, chance_ceiling=0.15)
    assert not result["passed"], "a shuffled run scoring nearly as well as normal indicates a broken control"


def test_bm25_closure_gates_on_published_not_on_the_in_repo_anchor():
    """
    The real SciFact case, which is why the control changed shape.

    Our BM25 (no stopwords, no stemming) scores 0.6605. The in-repo bm25s anchor
    (stopwords + Snowball) scores 0.6863, and the published Anserini figure is
    0.6650. Gating on the in-repo anchor at 0.02 would fail a correct
    implementation; gating on the published figure passes it comfortably, and the
    anchor is still reported so a large gap stays visible.
    """
    result = bm25_closure(our_ndcg=0.6605, published_ndcg=0.6650, anchor_ndcg=0.6863)
    assert result["passed"]
    assert result["absolute_difference"] == pytest.approx(0.0045, abs=1e-6)
    assert result["difference_to_in_repo_anchor"] == pytest.approx(0.0258, abs=1e-6)
