"""Experiment 002's two new controls, tested as functions against known numbers
(the embedding-shuffle control is also exercised end to end in test_dense.py)."""

from rb.controls import bm25_closure, embedding_shuffle


def test_bm25_closure_passes_within_tolerance():
    result = bm25_closure(our_ndcg=0.665, anchor_ndcg=0.660, tolerance=0.02)
    assert result["passed"]
    assert result["absolute_difference"] == 0.005


def test_bm25_closure_fails_outside_tolerance():
    result = bm25_closure(our_ndcg=0.700, anchor_ndcg=0.665, tolerance=0.02)
    assert not result["passed"]


def test_bm25_closure_boundary_is_inclusive():
    result = bm25_closure(our_ndcg=0.520, anchor_ndcg=0.500, tolerance=0.02)
    assert result["passed"], "difference exactly equal to tolerance must pass"


def test_embedding_shuffle_passes_when_shuffled_ndcg_below_ceiling():
    result = embedding_shuffle(normal_ndcg=0.60, shuffled_ndcg=0.05, chance_ceiling=0.15)
    assert result["passed"]


def test_embedding_shuffle_fails_when_shuffled_ndcg_does_not_collapse():
    result = embedding_shuffle(normal_ndcg=0.60, shuffled_ndcg=0.55, chance_ceiling=0.15)
    assert not result["passed"], "a shuffled run scoring nearly as well as normal indicates a broken control"
