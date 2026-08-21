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
  shapley_bootstrap — 95% intervals and pairwise ordering fractions on top of
                       shapley_values, so "saturation scored higher than idf"
                       can be checked against the noise instead of read off a
                       single point estimate (002's factorial shipped without
                       this and a 0.008 gap was misread as a ranking).
  spearman_correlation — rank correlation with its own bootstrap interval, for
                       testing a monotone-but-not-linear relationship (corpus
                       length vs. attribution, in a later experiment).

Shared across experiments — none of this is specific to lexical/dense/hybrid.
"""

import math
import random
from itertools import combinations

BOOTSTRAP_ROUNDS = 2000  # pre-registered in protocols/002-ladder.md section 5
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
    # 95% percentile interval: cut 2.5% off each tail of the sorted `rounds` draws.
    # The -1 on the upper index only is deliberate, not a typo: int(0.025 * rounds)
    # already lands on the correct 0-indexed element for the lower 2.5% cut, but
    # int(0.975 * rounds) is one PAST the element at the 97.5th percentile in a
    # 0-indexed list, so it needs the -1 to point at the same element the lower
    # index's convention would. Dropping it silently widens every interval's upper
    # bound by one draw and passed the whole suite once before this test existed:
    # tests/test_stats.py::test_percentile_bounds_pin_exact_values_on_a_known_distribution.
    lo, hi = means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]

    # NOT routed through `percentile_ci` below. The two agree at the round counts this repo
    # registers (n = 1000, 2000, 10000, where 0.975 * n is integral) and DIVERGE at arbitrary n:
    # at n = 100 this expression cuts 3 draws above and 2 below, while `percentile_ci` cuts 2 and
    # 2. `percentile_ci` is the more correct of the two by the derivation `_percentile_index`
    # carries. NB-26's council ruled that refactoring the
    # module 001 and 002 publish against, inside a correctness fix scoped to 003's graph arm,
    # bundles a cosmetic change into a diff a reviewer must check for moved numbers. Left as a
    # named follow-up rather than done quietly here. See NB-26 R6.
    # Two-sided p-value from the bootstrap distribution itself: twice the smaller
    # tail past zero, capped at 1 so a distribution that never crosses zero (e.g.
    # two identical runs, all diffs zero) reports p=1 rather than an undefined 0.
    #
    # KNOWN DEFECT, DELIBERATELY NOT FIXED HERE. This naive form returns exactly 0.0
    # when no resample crosses, and 002's committed artifacts carry 75 such zeros
    # across 27 Holm families. `bootstrap_p_value` below is the corrected estimator
    # and 003 uses it. This call site is NOT switched over, because doing so would
    # change numbers in a published entry, and that is an author's decision rather
    # than a refactor's. Measured before leaving it: correcting these p-values flips
    # ZERO Holm decisions in 002, so only the reported figures differ. See NB-25 R6.
    below = sum(1 for m in means if m <= 0) / rounds
    above = sum(1 for m in means if m >= 0) / rounds
    p_value = min(1.0, 2 * min(below, above))

    return {"mean_diff": observed, "ci95": [lo, hi], "p_value": p_value}


def _percentile_index(n: int, tail: float, upper: bool) -> int:
    """
    Index of the percentile bound in a SORTED list of `n` draws.

    ONE RULE, DERIVED, rather than a `-1` copied between call sites. A percentile bound here
    means: exclude `tail` of the draws beyond it, and return the outermost draw that SURVIVES.
    So `k = int(tail * n)` draws are cut, and the bound sits at index `k` from the bottom or
    `n - 1 - k` from the top.

    That derivation is why the two indices are not symmetric expressions. Written by hand as
    `int(0.975 * n)` the upper bound lands one PAST the surviving draw, cutting one fewer than
    the lower bound cuts, which silently widens every interval's upper side. Reproduces the
    existing convention exactly at n = 1000, 2000 and 10000, and generalises to the one-sided
    case that `int(0.95 * n)` got wrong: at n = 1000 that index leaves 4.9% above it, not 5%.
    """
    if n <= 0:
        raise ValueError("percentile of an empty draw list is undefined")
    # A tail at or above 0.5 cuts more than half from one end: the two bounds cross, and on the
    # upper side `n - 1 - k` goes NEGATIVE and silently wraps to a plausible-looking draw from the
    # wrong end of the distribution. Rejecting beats returning a number nobody can tell is wrong —
    # this function exists so the ninth call site cannot pick a wrong answer quietly.
    if not 0 <= tail < 0.5:
        raise ValueError(f"tail must be in [0, 0.5); got {tail}")
    k = int(tail * n)
    return n - 1 - k if upper else k


def percentile_ci(sorted_draws: list[float], tail: float = 0.025) -> tuple[float, float]:
    """Two-sided percentile interval, cutting `tail` from each end. Input must be sorted."""
    n = len(sorted_draws)
    return sorted_draws[_percentile_index(n, tail, False)], sorted_draws[_percentile_index(n, tail, True)]


def upper_percentile(sorted_draws: list[float], tail: float = 0.05) -> float:
    """
    One-sided upper threshold, cutting `tail` above it. Input must be sorted.

    Separate from `percentile_ci` because a one-sided rejection threshold is a different
    statistical object from an interval bound, and §8.3's gate decides on this one
    (`passed = observed > null_hi`). Sharing `_percentile_index` is what keeps the two
    honest about using the same rule.
    """
    return sorted_draws[_percentile_index(len(sorted_draws), tail, True)]


def bootstrap_p_value(draws: list[float], rounds: int | None = None) -> float:
    """
    Two-sided bootstrap p-value: twice the smaller tail past zero, add-one smoothed.

    WHY THE +1, AND WHY IT IS NOT A FLOOR. The naive form `2 * min(c_le, c_ge) / B`
    returns EXACTLY ZERO whenever no resample crosses the null, and a p-value of exactly
    zero is a claim no finite resampling procedure can make. The standard remedy
    (Davison & Hinkley 1997; Phipson & Smyth 2010) is the add-one estimator below, which
    cannot return zero by construction.

    Deliberately NOT `max(p, floor)`. A floor needs a companion flag so a reader can tell
    "the procedure bottomed out" from "the procedure measured this" — two moving parts
    patching a biased estimator, where this is one correct estimator and no flag. It is
    also why the floor value would NOT have been 1/B: the counts are integers and the
    `2 *` doubles them, so the naive statistic can only emit multiples of 2/B, and a 1/B
    floor would report a resolution finer than the procedure actually has.
    """
    b = rounds if rounds is not None else len(draws)
    if b <= 0:
        raise ValueError("bootstrap_p_value requires at least one draw")
    c_le = sum(1 for x in draws if x <= 0)
    c_ge = sum(1 for x in draws if x >= 0)
    return min(1.0, 2 * min(c_le + 1, c_ge + 1) / (b + 1))


def holm_adjusted(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni ADJUSTED p-values, in the SAME order as the input.

    Input order, not sorted order, so a caller can zip these against its own comparison
    labels. `holm_correction` depends on it and 002's ladder passes deliberately-unsorted
    p-values, so sorted output would silently mislabel a published code path.

    The running maximum is load-bearing: adjusted p-values must be non-decreasing in rank,
    so once a comparison fails every less significant one inherits at least its value.
    Dropping it breaks the equivalence `holm_correction` relies on.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):  # rank is 0-indexed
        running = min(1.0, max(running, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Holm-Bonferroni step-down decisions across `m` comparisons, in input order.

    Expressed on top of `holm_adjusted` rather than as its own loop. Rejecting where the
    adjusted p-value is at or below alpha reproduces the step-down procedure exactly: the
    running maximum already propagates the first failure forward, which is what the
    previous implementation's early `break` did by hand. Verified equivalent over
    randomised trials including ties and boundary values — see tests/test_audit_fixes.py.

    One implementation rather than two. 002 consumes the decisions and 003 the adjusted
    values; while those were separate functions they were free to drift apart untested.
    """
    return [adj <= alpha for adj in holm_adjusted(p_values)]


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


def shapley_bootstrap(
    per_cell_scores: dict[frozenset, list[float]],
    players: list[str],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    95% interval and pairwise ordering fractions on the Shapley attribution,
    resampling QUERIES rather than cells.

    The eight cells of the lexical factorial are a complete enumeration of the
    idf/tf_saturation/length_norm switches, not a sample — there is nothing to
    resample there. What IS a sample is the query subsample each cell's mean
    nDCG@10 was computed over, so that is what gets resampled: draw queries
    with replacement, recompute every cell's mean nDCG@10 over that draw, run
    the existing exact `shapley_values` on those eight recomputed numbers, and
    repeat. The percentile spread of the resulting per-round Shapley values is
    the interval a reader actually wants: would this attribution hold on a
    different sample of queries from the same corpus.

    `per_cell_scores` maps each of the 2^n subsets of `players` (frozensets,
    same convention `shapley_values`/`shapley_from_ndcg` use, including
    frozenset() and frozenset(players)) to that cell's per-query nDCG@10,
    query-aligned: index i in every list must be the same query across every
    cell. Iteration below sorts cells by their member tuple rather than trusting
    dict order, so the result cannot depend on how the caller happened to build
    the dict.

    THE PAIRING. Within one round, the SAME drawn query indices are applied to
    recompute every one of the eight cells before Shapley runs on that round's
    numbers — exactly how `paired_bootstrap` keeps its two arms aligned.
    Drawing independent query samples per cell would let per-query noise that
    is common to a query (a hard query is hard for every config) cancel
    randomly instead of moving every cell together, which inflates every
    interval. See tests/test_stats.py's pairing test, which reproduces this
    failure deliberately by resampling independently and checking the interval
    widens.

    Round count and seed default to the same values as `paired_bootstrap`, so
    the two families of interval reported in one entry come from the same
    procedure.
    """
    if not per_cell_scores:
        raise ValueError("shapley_bootstrap requires at least one cell")
    n_queries = len(next(iter(per_cell_scores.values())))
    if n_queries == 0:
        raise ValueError("shapley_bootstrap requires at least one query")
    for subset, scores in per_cell_scores.items():
        if len(scores) != n_queries:
            raise ValueError(
                f"shapley_bootstrap requires every cell to be query-aligned (equal length); "
                f"cell {sorted(subset)} has {len(scores)}, expected {n_queries}"
            )

    # Sorted by member tuple, not dict insertion order: two callers building the
    # same cells in different order must get the identical bootstrap result.
    subsets = sorted(per_cell_scores, key=lambda s: tuple(sorted(s)))
    pairs = list(combinations(players, 2))

    rng = random.Random(seed)
    per_player_draws: dict[str, list[float]] = {p: [] for p in players}
    pair_wins = {pair: 0 for pair in pairs}

    pair_ties = {pair: 0 for pair in pairs}

    for _ in range(rounds):
        # ONE draw per round, reused for every cell below — this is the pairing
        # the module docstring above calls the single most important property.
        idx = rng.choices(range(n_queries), k=n_queries)
        round_means = {
            subset: sum(per_cell_scores[subset][i] for i in idx) / n_queries for subset in subsets
        }
        phi = shapley_values(round_means, players)
        for p in players:
            per_player_draws[p].append(phi[p])
        for a, b in pairs:
            if phi[a] > phi[b]:
                pair_wins[(a, b)] += 1
            elif phi[a] == phi[b]:
                pair_ties[(a, b)] += 1

    phi_ci95 = {}
    for p in players:
        draws = sorted(per_player_draws[p])
        # Same 2.5%/97.5% percentile-index convention as paired_bootstrap, including
        # the upper-only -1 — see the comment there for why the two indices are not
        # symmetric.
        phi_ci95[p] = [draws[int(0.025 * rounds)], draws[int(0.975 * rounds) - 1]]

    # Fraction of rounds a outranked b, for every pair — this is what lets the
    # entry say "we cannot tell" instead of forcing an order out of two point
    # estimates that happen to differ. A fraction near 0.5 in either direction
    # means the ordering flips depending on which queries you happened to draw.
    pairwise_ordering = {f"{a}>{b}": pair_wins[(a, b)] / rounds for a, b in pairs}

    # Ties are reported rather than left implicit. The comparison above is strict, so
    # two mechanisms that come out exactly equal every round give 0.0 in BOTH
    # directions, and a reader scanning only "a>b: 0.0" would conclude b dominates
    # when the truth is that they never separated. In a table that has already been
    # misread once, the difference between "never won" and "never differed" has to be
    # visible rather than inferred.
    pairwise_ties = {f"{a}={b}": pair_ties[(a, b)] / rounds for a, b in pairs}

    return {
        "phi_ci95": phi_ci95,
        "pairwise_ordering": pairwise_ordering,
        "pairwise_ties": pairwise_ties,
    }


def _fractional_ranks(values: list[float]) -> list[float]:
    """
    Average ("fractional") rank per value, 1-indexed: tied values share the
    mean of the ranks their tie block occupies. This is the standard tie
    adjustment for Spearman's rho — without it, two tied inputs would get an
    arbitrary rank order that depends on sort stability rather than on the
    data, and rho would depend on something that isn't a property of the data.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Every index in this tie block gets the mean of ranks i+1..j+1.
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation on raw values. Spearman's rho is exactly this
    applied to fractional ranks instead of raw values, which is how it is
    computed below rather than via the tie-free rank-difference shortcut
    formula, since that shortcut is only valid without ties."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    var_x = sum((xi - mx) ** 2 for xi in x)
    var_y = sum((yi - my) ** 2 for yi in y)
    if var_x == 0 or var_y == 0:
        # One side is constant (every rank tied): correlation is undefined by
        # the textbook formula (division by zero). Reporting 0 rather than
        # raising lets a bootstrap round that happens to resample a constant
        # column contribute a defined, conservative value instead of crashing
        # the whole interval.
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / math.sqrt(var_x * var_y)


def spearman_correlation(
    x: list[float], y: list[float], rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED
) -> dict:
    """
    Spearman rank correlation between paired series `x` and `y`, with a 95%
    percentile bootstrap interval over the pairs.

    Used later to correlate corpus mean document length against each lexical
    mechanism's Shapley value, where the predicted relationship is monotone
    (more length -> more normalisation payoff) rather than linear, which is
    why rank correlation rather than Pearson's r is the right statistic. Not
    wired into any caller yet — this entry's out-of-sample corpus extension
    (protocols/002 Job 2) has not been built.

    Bootstraps by resampling (x[i], y[i]) PAIRS with replacement, same
    approach as `paired_bootstrap`'s per-query resampling: recompute rho on
    each resampled pair set, then take the percentile interval over rounds.
    """
    if len(x) != len(y):
        raise ValueError("spearman_correlation requires equal-length, paired series")
    n = len(x)
    if n < 2:
        raise ValueError("spearman_correlation requires at least two paired points")

    rho = _pearson(_fractional_ranks(x), _fractional_ranks(y))

    rng = random.Random(seed)
    draws = []
    for _ in range(rounds):
        idx = rng.choices(range(n), k=n)
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        draws.append(_pearson(_fractional_ranks(xs), _fractional_ranks(ys)))
    draws.sort()
    # Same percentile-index convention as paired_bootstrap; see there for the -1.
    lo, hi = draws[int(0.025 * rounds)], draws[int(0.975 * rounds) - 1]

    return {"rho": rho, "ci95": [lo, hi]}
