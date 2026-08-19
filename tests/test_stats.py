"""Paired bootstrap, Holm correction, Shapley attribution, the Shapley
bootstrap and Spearman rank correlation — external behaviour only, against
small hand-constructed inputs whose correct answer can be checked by hand."""

import math
import random
from itertools import combinations

from rb.stats import (
    holm_correction,
    paired_bootstrap,
    shapley_bootstrap,
    shapley_values,
    spearman_correlation,
)


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


def _synthetic_additive_cells(contribs: dict[str, list[float]], players: list[str]) -> dict[frozenset, list[float]]:
    """
    Build the frozenset-keyed, query-aligned cell dict shapley_bootstrap
    expects, from a purely additive per-player per-query contribution (no
    synergy between players): cell S's score at query q is the sum of S's
    active players' contributions at q.

    For a purely additive game, a player's marginal contribution to ANY
    coalition is exactly its own standalone contribution — so the exact
    Shapley value for player p is the mean of contribs[p] over the queries
    used, which is what makes the constructed tests below checkable by
    inspection rather than by running the Shapley formula and trusting it.
    """
    n = len(next(iter(contribs.values())))
    cells = {}
    for r in range(len(players) + 1):
        for subset in combinations(players, r):
            s = frozenset(subset)
            cells[s] = [sum(contribs[p][q] for p in s) for q in range(n)]
    return cells


def test_shapley_bootstrap_dominant_mechanism_excludes_others():
    """A's true per-query contribution (0.30, small noise) is far larger than
    B's or C's (0.03 each, independently noisy): A's interval must sit
    entirely above both, and A must outrank each of them in nearly every
    round — this is the "gap is real" case, the counterpart to the
    near-identical case below."""
    rng = random.Random(13)
    n = 150
    contribs = {
        "A": [0.30 + rng.uniform(-0.01, 0.01) for _ in range(n)],
        "B": [0.03 + rng.uniform(-0.01, 0.01) for _ in range(n)],
        "C": [0.03 + rng.uniform(-0.01, 0.01) for _ in range(n)],
    }
    cells = _synthetic_additive_cells(contribs, ["A", "B", "C"])
    result = shapley_bootstrap(cells, ["A", "B", "C"])

    a_lo, _ = result["phi_ci95"]["A"]
    _, b_hi = result["phi_ci95"]["B"]
    _, c_hi = result["phi_ci95"]["C"]
    assert a_lo > b_hi, "A's interval must sit entirely above B's when A truly dominates"
    assert a_lo > c_hi, "A's interval must sit entirely above C's when A truly dominates"
    assert result["pairwise_ordering"]["A>B"] > 0.95
    assert result["pairwise_ordering"]["A>C"] > 0.95


def test_shapley_bootstrap_identical_mechanisms_give_overlapping_intervals_and_ordering_near_half():
    """B and C share the SAME true contribution (0.10), independently noisy —
    the whole point of the bootstrap is admitting this rather than forcing a
    ranking out of whichever averaged slightly higher: overlapping intervals
    and a pairwise ordering fraction near one half. This matters more than
    the dominance case above, per the spec."""
    rng = random.Random(7)
    n = 150
    b_contrib = [0.10 + rng.uniform(-0.03, 0.03) for _ in range(n)]
    c_contrib = [0.10 + rng.uniform(-0.03, 0.03) for _ in range(n)]
    # A random 150-point draw's own sample mean is not exactly its population
    # mean, so B and C's noisy draws would otherwise differ by a small but
    # consistent constant offset — enough for the bootstrap to reliably rank
    # one above the other every round, which is a fixed-data artifact of this
    # test's random draw, not the "true tie" the test means to construct.
    # Re-centring C onto B's exact sample mean removes that offset while
    # keeping each query's own noise shape, so any remaining ordering signal
    # comes only from resampling variance, which is what this test checks.
    c_contrib = [c - (sum(c_contrib) / n) + (sum(b_contrib) / n) for c in c_contrib]
    contribs = {
        "A": [0.30 + rng.uniform(-0.01, 0.01) for _ in range(n)],
        "B": b_contrib,
        "C": c_contrib,
    }
    cells = _synthetic_additive_cells(contribs, ["A", "B", "C"])
    result = shapley_bootstrap(cells, ["A", "B", "C"])

    b_lo, b_hi = result["phi_ci95"]["B"]
    c_lo, c_hi = result["phi_ci95"]["C"]
    assert not (b_hi < c_lo or c_hi < b_lo), "identical mechanisms must produce overlapping intervals"
    frac = result["pairwise_ordering"]["B>C"]
    assert 0.3 <= frac <= 0.7, f"identical mechanisms should be close to a coin flip, got {frac}"


def test_shapley_bootstrap_shared_query_draws_narrower_than_independent_draws():
    """
    The property the spec calls the single most important one in this job:
    resampling the SAME queries for every cell within a round must give
    narrower intervals than resampling each cell independently.

    Cells here are built from one shared per-query "quality" factor scaled by
    a per-coalition weight, so every cell moves together on a given query,
    exactly like real nDCG@10 does (a hard query is hard across every lexical
    config). Under shared draws, that correlation cancels out of every
    marginal-contribution difference Shapley depends on; under independent
    draws it does not, and the interval must widen.

    The independent-draw variant is deliberately reimplemented here rather
    than reached via a toggle on shapley_bootstrap — no such toggle exists in
    production code, on purpose (see implementation notes), so this test can
    only pass by shapley_bootstrap itself sharing the draw.
    """
    rng = random.Random(11)
    n = 120
    quality = [rng.uniform(0.3, 0.9) for _ in range(n)]
    players = ["A", "B", "C"]
    cells: dict[frozenset, list[float]] = {}
    for r in range(len(players) + 1):
        for subset in combinations(players, r):
            s = frozenset(subset)
            weight = 1.0 + 0.3 * len(s)  # depends only on coalition size
            cells[s] = [quality[q] * weight for q in range(n)]

    shared = shapley_bootstrap(cells, players)

    def independent_bootstrap(cells, players, rounds=2000, seed=20260818):
        subsets = sorted(cells, key=lambda s: tuple(sorted(s)))
        rng_local = random.Random(seed)
        draws = {p: [] for p in players}
        for _ in range(rounds):
            round_means = {}
            for subset in subsets:
                idx = rng_local.choices(range(n), k=n)  # INDEPENDENT draw per cell — the bug under test
                round_means[subset] = sum(cells[subset][i] for i in idx) / n
            phi = shapley_values(round_means, players)
            for p in players:
                draws[p].append(phi[p])
        return {p: [sorted(draws[p])[int(0.025 * rounds)], sorted(draws[p])[int(0.975 * rounds) - 1]] for p in players}

    independent = independent_bootstrap(cells, players)

    for p in players:
        shared_width = shared["phi_ci95"][p][1] - shared["phi_ci95"][p][0]
        independent_width = independent[p][1] - independent[p][0]
        assert shared_width < independent_width, (
            f"{p}: shared-draw interval ({shared_width}) should be narrower than the "
            f"independent-draw interval ({independent_width}) — the pairing is not working"
        )


def test_shapley_bootstrap_deterministic_across_invocations():
    rng = random.Random(3)
    n = 60
    contribs = {p: [0.1 * i + rng.uniform(-0.02, 0.02) for _ in range(n)] for i, p in enumerate(["A", "B", "C"], 1)}
    cells = _synthetic_additive_cells(contribs, ["A", "B", "C"])
    result1 = shapley_bootstrap(cells, ["A", "B", "C"])
    result2 = shapley_bootstrap(cells, ["A", "B", "C"])
    assert result1 == result2


def test_shapley_bootstrap_deterministic_across_processes():
    """Same seed, same input, same interval — including across a fresh Python
    process, not just a fresh call in the same one, ruling out any hidden
    reliance on process-local state (e.g. PYTHONHASHSEED-sensitive iteration)."""
    import os
    import subprocess
    import sys

    script = (
        "import json\n"
        "from rb.stats import shapley_bootstrap\n"
        "cells = {\n"
        "    frozenset(): [0.10, 0.20, 0.15, 0.05, 0.12] * 6,\n"
        "    frozenset({'A'}): [0.40, 0.50, 0.45, 0.35, 0.41] * 6,\n"
        "    frozenset({'B'}): [0.30, 0.35, 0.32, 0.28, 0.31] * 6,\n"
        "    frozenset({'A', 'B'}): [0.70, 0.75, 0.72, 0.68, 0.71] * 6,\n"
        "}\n"
        "print(json.dumps(shapley_bootstrap(cells, ['A', 'B']), sort_keys=True))\n"
    )
    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=os.environ.copy()
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1], "shapley_bootstrap must be deterministic across separate processes"


def test_spearman_correlation_hand_computed_with_tied_case():
    """x has a tied pair (the two 2s); average ranks give rank_x =
    [1, 2.5, 2.5, 4, 5], rank_y = [5, 4, 3, 2, 1] (no ties in y). Pearson's r
    on those ranks works out to -9.5 / sqrt(9.5 * 10) = -9.5 / sqrt(95),
    hand-computed below rather than re-derived by the implementation."""
    x = [1, 2, 2, 4, 5]
    y = [5, 4, 3, 2, 1]
    expected_rho = -9.5 / math.sqrt(95)
    result = spearman_correlation(x, y)
    assert abs(result["rho"] - expected_rho) < 1e-9
    lo, hi = result["ci95"]
    assert -1.0 <= lo <= hi <= 1.0


def test_spearman_correlation_perfect_monotone_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    result = spearman_correlation(x, y)
    assert result["rho"] == 1.0


def test_spearman_correlation_requires_matched_length():
    import pytest
    with pytest.raises(ValueError):
        spearman_correlation([1.0, 2.0], [1.0])
