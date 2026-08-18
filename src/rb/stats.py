"""
Shared statistics for comparing rungs honestly.

001 reported a bootstrap interval per arm and stopped there, which cannot answer
whether an adjacent-rung gap is real or noise — two overlapping per-arm intervals
say nothing about the paired difference. This module provides the three pieces
002 needs instead:

  paired_bootstrap  — interval + p-value on the per-query DIFFERENCE between two
                       runs, so a gap is judged on the pairing, not on two
                       marginal distributions.
  holm_correction   — step-down correction across the several adjacent-rung
                       comparisons made on one dataset, since six-plus comparisons
                       on one test set inflate the false-positive rate if left
                       uncorrected.
  shapley_values    — order-independent attribution across the three interacting
                       lexical mechanisms, computed from the full eight-cell
                       factorial rather than from a single ladder ordering.

Shared across experiments — none of this is specific to lexical/dense/hybrid.
"""

import math
import random
from itertools import combinations

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20260818  # same seed as 001's rescore.py, for consistency


def paired_bootstrap(
    a: list[float], b: list[float], rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED
) -> dict:
    """
    95% interval on the per-query difference a - b, plus a two-sided bootstrap
    p-value. Paired, not two independent bootstraps: a and b are scores on the
    SAME queries in the SAME order, and resampling the differences (rather than
    resampling each side separately) is what keeps that pairing intact.
    """
    if len(a) != len(b):
        raise ValueError("paired_bootstrap requires equal-length, query-aligned score lists")
    n = len(a)
    if n == 0:
        raise ValueError("paired_bootstrap requires at least one query")

    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(rounds))
    observed = sum(diffs) / n
    lo, hi = means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]

    # Two-sided p-value from the bootstrap distribution itself: twice the smaller
    # tail past zero, capped at 1 so a distribution that never crosses zero (e.g.
    # two identical runs, all diffs zero) reports p=1 rather than an undefined 0.
    below = sum(1 for m in means if m <= 0) / rounds
    above = sum(1 for m in means if m >= 0) / rounds
    p_value = min(1.0, 2 * min(below, above))

    return {"mean_diff": observed, "ci95": [lo, hi], "p_value": p_value}


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Holm-Bonferroni step-down correction across `m` comparisons.

    Returns significance decisions in the SAME order as `p_values` was given
    (not sorted order), so a caller can zip this directly against its own list of
    comparison labels without re-deriving the sort.

    Step-down: sort ascending, test the smallest p-value against alpha/m, the next
    against alpha/(m-1), and so on; the first failure and everything less
    significant after it are non-significant. This is uniformly more powerful
    than a flat Bonferroni correction while controlling the same family-wise
    error rate.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    significant = [False] * m
    for rank, idx in enumerate(order):  # rank is 0-indexed
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            significant[idx] = True
        else:
            break  # everything less significant than a failed comparison also fails
    return significant


def shapley_values(values: dict[frozenset, float], players: list[str]) -> dict[str, float]:
    """
    Exact Shapley value per player from a full 2^n-cell factorial.

    Used for the three lexical mechanisms (n=3, eight cells) instead of reading
    gains off a single ladder ordering: with interacting mechanisms, a ladder
    hands all of the shared credit to whichever mechanism happened to go first,
    and the resulting number would be quoted as a property of the mechanism
    rather than of the arbitrary order chosen. With only three players the full
    factorial is cheap, so there is no reason to accept the order-dependent
    answer.

    `values` must contain every subset of `players` as a frozenset key, including
    frozenset() for the all-off baseline and frozenset(players) for the all-on
    corner.
    """
    n = len(players)
    fact = math.factorial
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                s = frozenset(subset)
                # Standard Shapley weight: the probability that player p arrives
                # after exactly this subset, averaged over every arrival order.
                weight = fact(len(s)) * fact(n - len(s) - 1) / fact(n)
                total += weight * (values[s | {p}] - values[s])
        phi[p] = total
    return phi
