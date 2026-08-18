"""Paired bootstrap, Holm correction and Shapley attribution — external behaviour
only, against small hand-constructed inputs whose correct answer can be checked
by hand."""

from rb.stats import holm_correction, paired_bootstrap, shapley_values


def test_paired_bootstrap_identical_runs_gives_interval_containing_zero():
    a = [0.5, 0.6, 0.7, 0.4, 0.55] * 20
    b = list(a)  # identical run: every per-query difference is exactly zero
    result = paired_bootstrap(a, b)
    lo, hi = result["ci95"]
    assert lo <= 0.0 <= hi
    assert result["mean_diff"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_bootstrap_strictly_better_run_gives_interval_excluding_zero():
    # b is consistently ~0.4 below a with small variance — a real, non-noise gap.
    a = [0.9, 0.8, 0.85, 0.95, 0.7] * 20
    b = [0.5, 0.4, 0.45, 0.55, 0.3] * 20
    result = paired_bootstrap(a, b)
    lo, hi = result["ci95"]
    assert lo > 0.0, "a consistent positive gap should exclude zero from the interval"
    assert result["p_value"] < 0.05


def test_paired_bootstrap_requires_matched_length():
    import pytest
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_holm_correction_step_down_on_known_p_values():
    """Hand-worked example, m=4, alpha=0.05:
    sorted p=[0.01, 0.02, 0.03, 0.20], thresholds=[0.0125, 0.0167, 0.025, 0.05].
    0.01 <= 0.0125 passes. 0.02 <= 0.0167 fails -> step-down stops there."""
    p_values = [0.03, 0.01, 0.20, 0.02]  # deliberately out of sorted order
    significant = holm_correction(p_values, alpha=0.05)
    assert significant == [False, True, False, False]


def test_holm_correction_all_significant_when_all_p_values_tiny():
    p_values = [0.0001, 0.0002, 0.0003]
    assert holm_correction(p_values) == [True, True, True]


def test_holm_correction_preserves_input_order():
    p_values = [0.9, 0.0001, 0.5]
    significant = holm_correction(p_values)
    assert significant[1] is True
    assert significant[0] is False and significant[2] is False


def test_shapley_values_hand_computed_additive_game():
    """No synergy between players: v(S) = |S|. Each player's marginal
    contribution is 1 regardless of who else is present, so the Shapley value
    of every player must be exactly 1 (the textbook additive-game result)."""
    players = ["A", "B", "C"]
    values = {frozenset(s): len(s) for r in range(4) for s in _subsets(players, r)}
    phi = shapley_values(values, players)
    assert phi == {"A": 1.0, "B": 1.0, "C": 1.0}


def test_shapley_values_hand_computed_with_synergy():
    """A and B are worthless alone but worth 10 together; C always worth 1 alone
    and adds nothing to any coalition. By symmetry A and B must get equal credit
    for the 10, and their shares must sum with C's to the grand coalition's total
    value of 11."""
    v = {
        frozenset(): 0,
        frozenset({"A"}): 0,
        frozenset({"B"}): 0,
        frozenset({"C"}): 1,
        frozenset({"A", "B"}): 10,
        frozenset({"A", "C"}): 1,
        frozenset({"B", "C"}): 1,
        frozenset({"A", "B", "C"}): 11,
    }
    phi = shapley_values(v, ["A", "B", "C"])
    assert phi["A"] == phi["B"]
    assert abs(sum(phi.values()) - 11) < 1e-9, "Shapley values must sum to the grand coalition's total value"


def _subsets(players, r):
    from itertools import combinations
    return combinations(players, r)
