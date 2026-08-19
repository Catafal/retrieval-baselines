"""
Win/loss-by-property analysis (rb.experiments.ladder.analysis), tested against
a small fabricated set of per-query nDCG values and properties so the binning
and interval logic can be checked without a real retrieval run.
"""

import pytest

from rb.experiments.ladder.analysis import SPARSE_THRESHOLD, win_loss_by_property


def _make(n, dense_better_from=0):
    """n queries q0..q{n-1}; dense_ndcg beats bm25_ndcg for queries at or past
    `dense_better_from`, loses before it. `properties` gives each query an
    ascending query_length so quantile binning has something to split on, and
    a constant value for every other property so those bins collapse to one."""
    qids = [f"q{i}" for i in range(n)]
    dense = {q: 0.5 for q in qids}
    bm25 = {q: 0.5 for q in qids}
    for i, q in enumerate(qids):
        if i >= dense_better_from:
            dense[q] = 0.9
        else:
            bm25[q] = 0.9
    properties = {
        q: {
            "query_length": i,
            "max_idf": 1.0,
            "mean_idf": 1.0,
            "gold_jaccard": 0.0,
            "gold_count": 1,
        }
        for i, q in enumerate(qids)
    }
    return dense, bm25, properties


def test_win_loss_reports_all_five_properties():
    dense, bm25, properties = _make(20, dense_better_from=10)
    result = win_loss_by_property(dense, bm25, properties)
    for prop in ("query_length", "max_idf", "mean_idf", "gold_jaccard", "gold_count"):
        assert prop in result
    assert "per_query_diff" in result


def test_per_query_diff_is_the_raw_paired_difference():
    dense, bm25, properties = _make(4, dense_better_from=2)
    result = win_loss_by_property(dense, bm25, properties)
    # q0 is a bm25-wins query (dense=0.5, bm25=0.9); q3 is a dense-wins query
    # (dense=0.9, bm25=0.5) since dense_better_from=2.
    assert result["per_query_diff"]["q0"] == pytest.approx(0.5 - 0.9)
    assert result["per_query_diff"]["q3"] == pytest.approx(0.9 - 0.5)


def test_discrete_property_bins_by_exact_value_not_quantile():
    """gold_count is constant (1) for every query in this fixture, so it must
    collapse to exactly one bin, not be split by quantile the way a
    continuous property would be."""
    dense, bm25, properties = _make(20, dense_better_from=10)
    result = win_loss_by_property(dense, bm25, properties)
    assert len(result["gold_count"]) == 1
    assert result["gold_count"][0]["n_queries"] == 20


def test_continuous_property_splits_into_up_to_four_quantile_bins():
    dense, bm25, properties = _make(20, dense_better_from=10)
    result = win_loss_by_property(dense, bm25, properties)
    assert len(result["query_length"]) == 4
    total = sum(b["n_queries"] for b in result["query_length"])
    assert total == 20  # every query lands in exactly one bin


def test_win_rate_reflects_which_arm_won_in_each_bin():
    dense, bm25, properties = _make(20, dense_better_from=10)
    result = win_loss_by_property(dense, bm25, properties)
    # query_length ascends 0..19; dense wins queries 10..19, so the two lowest
    # quantile bins (query_length 0-9) are all BM25 wins and the two highest
    # (10-19) are all dense wins.
    bins = result["query_length"]
    assert bins[0]["dense_win_rate"] == 0.0
    assert bins[-1]["dense_win_rate"] == 1.0


def test_sparse_bins_are_flagged_not_dropped():
    """Fewer queries than SPARSE_THRESHOLD in a bin must still appear in the
    output, marked sparse, per the amendment: 'never dropped.' Uses the
    DISCRETE gold_count property (bins by exact value, not by rank), so tied
    values collapse into one small bin rather than being spread across the
    quantile split a continuous property like query_length would get."""
    n = SPARSE_THRESHOLD - 1
    dense = {f"q{i}": 0.5 for i in range(n)}
    bm25 = {f"q{i}": 0.5 for i in range(n)}
    properties = {
        f"q{i}": {"query_length": 1, "max_idf": 1.0, "mean_idf": 1.0, "gold_jaccard": 0.0, "gold_count": 1}
        for i in range(n)
    }
    result = win_loss_by_property(dense, bm25, properties)
    assert len(result["gold_count"]) == 1
    assert result["gold_count"][0]["n_queries"] == n
    assert result["gold_count"][0]["sparse"] is True


def test_win_loss_requires_identical_query_sets():
    with pytest.raises(ValueError):
        win_loss_by_property({"q1": 0.5}, {"q2": 0.5}, {"q1": {}, "q2": {}})


def test_every_bin_carries_a_bootstrap_interval():
    dense, bm25, properties = _make(20, dense_better_from=10)
    result = win_loss_by_property(dense, bm25, properties)
    for b in result["query_length"]:
        assert "ci95" in b and len(b["ci95"]) == 2
        assert "p_value" in b
        assert "mean_diff" in b
