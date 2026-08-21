"""
NB-26 D3 and D4 — the shared percentile rule, and the seed diagnostic's node-set guard.
"""

import pytest

from rb.experiments.graph import run_controls
from rb.experiments.graph.extraction_score import normalise
from rb.stats import _percentile_index, percentile_ci, upper_percentile


def test_percentile_indices_cut_the_same_fraction_from_each_tail():
    """KILLS: writing the upper bound as `int(0.975 * n)`.

    The rule is 'exclude `tail` of the draws beyond the bound'. Both tails must therefore cut the
    SAME number of draws. The hand-written upper expression cut one fewer, which is the entire
    defect: every published upper bound sat one draw too high.
    """
    for n in (1000, 2000, 10000):
        lo = _percentile_index(n, 0.025, upper=False)
        hi = _percentile_index(n, 0.025, upper=True)
        assert lo == n - 1 - hi, f"tails are asymmetric at n={n}"
        assert lo == int(0.025 * n)
        assert hi == int(0.975 * n) - 1, "the correct index, not the hand-written one"
        assert hi != int(0.975 * n), "if these were equal the defect would be invisible"


def test_one_sided_threshold_leaves_exactly_the_tail_above_it():
    """KILLS: `draws[int(0.95 * b)]` in bridge_reachability — the section 8.3 GATE's threshold.

    At b=1000 the hand-written index leaves 49 draws above it (4.9%), not 50 (5.0%), so the gate
    was fractionally easier to pass than it claimed. This is the one converted site that decides
    rather than reports.
    """
    n = 1000
    hi = _percentile_index(n, 0.05, upper=True)
    assert n - 1 - hi == 50, "exactly 5% of draws must sit above a 95th-percentile threshold"
    assert hi == 949 and int(0.95 * n) == 950, "the defective index selected a different draw"


def test_percentile_ci_and_upper_percentile_select_the_expected_draws():
    draws = [float(i) for i in range(1000)]  # sorted, distinct: every index is distinguishable
    assert percentile_ci(draws) == (25.0, 974.0)
    assert upper_percentile(draws) == 949.0


def test_percentile_handles_small_and_degenerate_inputs():
    """bridge_reachability can produce a short draw list; the rule must not go out of bounds."""
    assert percentile_ci([1.0]) == (1.0, 1.0)
    assert upper_percentile([1.0, 2.0]) == 2.0
    with pytest.raises(ValueError):
        percentile_ci([])


def test_seed_match_node_set_excludes_entities_that_normalise_to_empty(monkeypatch):
    """KILLS: dropping the `if normalise(e)` guard in seed_match_rate (NB-26 D4).

    A surface form of pure punctuation normalises to "" and is skipped by `build.build`, so the
    real graph has no such node. Unguarded, the diagnostic added "" to its node set and would
    credit a query seeded only by punctuation with a link the retriever cannot make.
    """
    assert normalise("&") == "", "fixture assumption: this surface form normalises to empty"

    monkeypatch.setattr(run_controls.datasets, "load_queries", lambda name: {"q1": "irrelevant"})
    monkeypatch.setattr(run_controls.datasets, "load_qrels", lambda name: {"q1": {"d1": 1}})
    # The query's only entity is punctuation, which normalises to the empty string.
    monkeypatch.setattr(run_controls.extractor, "extract", lambda text: [("&", "ORG")])
    monkeypatch.setattr(run_controls.extractor, "node_strings", lambda ents: ["&"])

    out = run_controls.seed_match_rate({"d1": ["&", "Paris"]})
    assert out["with_a_linked_entity"] == 0, (
        "the query's only entity normalises to empty, which is not a node in the real graph"
    )
    assert out["seed_match_rate"] == 0.0


def test_the_section_8_3_gate_uses_the_correct_one_sided_threshold():
    """
    KILLS: reverting bridge_reachability's `null_hi` to `draws[int(0.95 * b)]`.

    THE ONLY TEST HERE THAT PINS A DECISION. The first mutation sweep of this branch found the
    gate site SURVIVING — every other converted site was covered, and the one that decides rather
    than reports was not. That is the same shape as the vacuous test this branch repaired, so it
    is pinned with a fixture where the two candidate indices give a DIFFERENT VERDICT rather than
    a different fourth decimal.

    40 documents in 4 groups of 10, sharing within a group; 60 gold pairs of which exactly 20
    share. The null's draws separate at the boundary — draws[949] = 0.3167, draws[950] = 0.3333 —
    and observed is exactly 20/60 = 0.3333. So:

        correct index:   0.3333 > 0.3167  -> passed = True
        defective index: 0.3333 > 0.3333  -> passed = False

    The gate flips. A pre-registered gate whose verdict depends on an off-by-one nobody tested is
    the worst case this branch exists to close.
    """
    from rb.experiments.graph.extraction_score import bridge_reachability

    entities = {f"d{i}": [f"E{i}", f"S{i // 10}"] for i in range(40)}
    sharing = [(f"d{i}", f"d{i + 1}") for g in range(4) for i in range(g * 10, g * 10 + 5)]
    non_sharing = [(f"d{i}", f"d{i + 20}") for i in range(10)] * 4
    result = bridge_reachability(entities, sharing + non_sharing)

    assert result["gold_pairs_scored"] == 60
    assert result["observed_share_rate"] == 0.3333
    assert result["random_pair_rate_p95"] == 0.3167, (
        "the defective index would report 0.3333 here, and the gate would fail"
    )
    assert result["passed"] is True
