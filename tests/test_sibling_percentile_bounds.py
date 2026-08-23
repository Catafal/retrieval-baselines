"""
The percentile off-by-one is pinned at EVERY call site, not only the one that was audited.

`paired_bootstrap`'s upper index is pinned by
test_stats.py::test_percentile_bounds_pin_exact_values_on_a_known_distribution. Its two
siblings, `shapley_bootstrap` (stats.py:319) and `spearman_correlation` (stats.py:420), copy
the same expression and the first of them says so in a comment claiming parity — but neither
had a test. A mutation sweep confirmed the gap: dropping the `-1` at either sibling survives
the entire suite, while the identical mutation at `paired_bootstrap` is killed.

That is exactly the shape of the defect this repo already published once. The audit fixed the
site it was looking at and left two copies of it behind, which is the failure mode of auditing
by location instead of by defect class.

Both tests below carry the same degeneracy guard as the original: if the two candidate indices
ever select the same draw, the fixture proves nothing and says so rather than passing quietly.
The original version of the paired_bootstrap test was vacuous for its entire life for precisely
that reason.
"""
import pytest

from rb.stats import _fractional_ranks, _pearson, shapley_values, shapley_bootstrap, spearman_correlation


def test_shapley_interval_pins_the_upper_percentile_index():
    """KILLS: dropping the `-1` from shapley_bootstrap's upper percentile index."""
    players = ["idf", "tf_sat", "len_norm"]
    # A full factorial over 3 players = 8 cells. Scores vary per query and per cell so the
    # resampled Shapley draws form a continuous-enough distribution for adjacent order
    # statistics to differ.
    # Each player contributes a DIFFERENT amount on each query, so a resample that draws a
    # different query mix produces a different Shapley value. The first attempt at this
    # fixture gave every player a constant contribution; the Shapley draws were then
    # identical every round and the degeneracy guard below caught it, which is the whole
    # reason that guard exists.
    contribution = {
        "idf": [0.30 + (q % 11) * 0.021 for q in range(80)],
        "tf_sat": [0.12 + (q % 7) * 0.033 for q in range(80)],
        "len_norm": [0.05 + (q % 13) * 0.014 for q in range(80)],
    }
    cells = {}
    for mask in range(8):
        members = frozenset(p for i, p in enumerate(players) if mask >> i & 1)
        cells[members] = [
            0.2 + sum(contribution[p][q] for p in members) for q in range(80)
        ]

    rounds = 1000
    out = shapley_bootstrap(cells, players, rounds=rounds)

    for p in players:
        lo, hi = out["phi_ci95"][p]
        assert lo <= hi, f"{p}: interval inverted"

    # The load-bearing assertion: the upper bound must be the draw at n-1-k, not at n-k.
    # Recompute the draw list the same way the function does and compare both candidates.
    draws = sorted(_shapley_draws(cells, players, rounds)["idf"])
    assert out["phi_ci95"]["idf"][1] == pytest.approx(draws[int(0.975 * rounds) - 1])
    assert draws[int(0.975 * rounds) - 1] != draws[int(0.975 * rounds)], (
        "fixture went degenerate: both index choices select the same draw"
    )


def test_spearman_interval_pins_the_upper_percentile_index():
    """KILLS: dropping the `-1` from spearman_correlation's upper percentile index."""
    # Monotone-but-noisy so the bootstrap rho distribution is continuous enough.
    x = [i * 1.0 for i in range(60)]
    y = [(i * 0.9) + (i % 7) * 1.3 for i in range(60)]

    rounds = 1000
    out = spearman_correlation(x, y, rounds=rounds)
    lo, hi = out["ci95"]

    assert lo <= hi, "interval inverted"

    draws = sorted(_spearman_draws(x, y, rounds))
    assert hi == pytest.approx(draws[int(0.975 * rounds) - 1])
    assert draws[int(0.975 * rounds) - 1] != draws[int(0.975 * rounds)], (
        "fixture went degenerate: both index choices select the same draw"
    )


# --- independently recomputed draw sequences, mirroring each function's own resampling ---
# Separately derived rather than read back out of the function, so the assertions above pin
# the index against an outside reference instead of against the implementation under test.


def _shapley_draws(cells, players, rounds):
    import random as _random

    n = len(next(iter(cells.values())))
    rng = _random.Random(20260818)
    out = {p: [] for p in players}
    for _ in range(rounds):
        idx = rng.choices(range(n), k=n)
        round_means = {s: sum(v[i] for i in idx) / n for s, v in cells.items()}
        phi = shapley_values(round_means, players)
        for p in players:
            out[p].append(phi[p])
    return out


def _spearman_draws(x, y, rounds):
    import random as _random

    n = len(x)
    rng = _random.Random(20260818)
    draws = []
    for _ in range(rounds):
        idx = rng.choices(range(n), k=n)
        draws.append(_pearson(_fractional_ranks([x[i] for i in idx]),
                              _fractional_ranks([y[i] for i in idx])))
    return draws
