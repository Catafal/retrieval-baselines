"""
The bootstrap p-value estimator and the single Holm implementation (NB-25 D2, D5).

MUTATION-CHECKED. NB-24 shipped three tests that passed whether or not the defect was present:
one raised the error it asserted, one used a topology where the term under test normalised
away, one asserted a sum the code's own guard restored even when the invariant was violated.
A test that passes is not evidence; only a test that FAILS against the defect is. Each test
below names the mutation it kills. See NB-25 for the audit these close.
"""

import json
import numpy as np
import pytest

from rb import stats


# ---------------------------------------------------------------- D2: the p-value estimator

def test_p_value_cannot_be_zero_when_no_resample_crosses():
    """KILLS: reverting to `2 * min(c_le, c_ge) / B`, which returns exactly 0.0 here.

    Every draw strictly positive, so the light tail has a count of zero — the case that
    produced all twelve published zeros.
    """
    p = stats.bootstrap_p_value([1.0] * 10_000, 10_000)
    assert p > 0.0
    assert p == pytest.approx(2 / 10_001)


def test_p_value_floor_is_not_one_over_b():
    """KILLS: `max(p, 1/B)`, the fix originally specified.

    1/B = 0.0001 is a value the two-sided statistic cannot emit — its counts are integers and
    the `2 *` doubles them. A floor there would claim a resolution finer than the procedure
    has. This pins the actual smallest value ABOVE 1/B.
    """
    p = stats.bootstrap_p_value([1.0] * 10_000, 10_000)
    assert p > 1 / 10_000


def test_p_value_is_not_uniformly_the_minimum():
    """KILLS: an implementation that always returns the smallest value.

    A distribution straddling zero must report a large p. Without this, "always return 2/(B+1)"
    would pass the two tests above.
    """
    draws = [-1.0] * 5_000 + [1.0] * 5_000
    assert stats.bootstrap_p_value(draws, 10_000) > 0.9


def test_p_value_is_two_sided_and_capped():
    """A perfectly symmetric distribution cannot report p > 1."""
    assert stats.bootstrap_p_value([-1.0, 1.0], 2) <= 1.0


# --------------------------------------------------------------------------- D5: one Holm

def _holm_adjusted_reference(p_values):
    """Frozen copy of the pre-refactor algorithm, for differential testing.

    Kept deliberately: the spec's hand-picked cases only cover edge cases someone thought of,
    and a differential test against the original covers the ones nobody did.
    """
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m, out, running = len(ordered), [0.0] * len(p_values), 0.0
    for i, idx in enumerate(ordered):
        running = min(1.0, max(running, (m - i) * p_values[idx]))
        out[idx] = running
    return out


def test_holm_adjusted_applies_the_running_maximum():
    """KILLS: `adj = min(1.0, (m - rank) * p)` without `max(running, ...)`.

    Scaled values here are [0.08, 0.045] — non-monotone. Correct output is [0.08, 0.08];
    the mutant returns [0.08, 0.045]. Adjusted p-values must be non-decreasing in rank.
    """
    assert stats.holm_adjusted([0.04, 0.045]) == pytest.approx([0.08, 0.08])


def test_holm_adjusted_returns_input_order_not_sorted_order():
    """KILLS: returning the values in ascending-p order.

    002's ladder passes deliberately-unsorted p-values and zips the result against its own
    labels, so sorted output would silently mislabel every published comparison.
    """
    assert stats.holm_adjusted([0.5, 0.01])[1] < stats.holm_adjusted([0.5, 0.01])[0]


def test_holm_adjusted_handles_ties():
    """Tied p-values must receive identical adjusted values, not order-dependent ones."""
    adj = stats.holm_adjusted([0.02, 0.02, 0.02])
    assert adj[0] == adj[1] == adj[2]


def test_holm_adjusted_matches_the_frozen_reference_implementation():
    """DIFFERENTIAL: the refactor must not change any value, on any input shape."""
    rng = np.random.default_rng(20260821)
    for _ in range(2_000):
        m = int(rng.integers(1, 8))
        # Coarse rounding deliberately manufactures ties, the case most likely to diverge.
        ps = [round(float(x), 2) for x in rng.random(m)]
        assert stats.holm_adjusted(ps) == pytest.approx(_holm_adjusted_reference(ps))


def test_holm_correction_still_matches_the_step_down_decisions():
    """KILLS: any change to holm_correction's decisions. 002 is PUBLISHED on this function.

    Reimplements the original break-based step-down and requires identical output.
    """
    def original(p_values, alpha=0.05):
        m = len(p_values)
        order = sorted(range(m), key=lambda i: p_values[i])
        sig = [False] * m
        for rank, idx in enumerate(order):
            if p_values[idx] <= alpha / (m - rank):
                sig[idx] = True
            else:
                break
        return sig

    rng = np.random.default_rng(4242)
    for _ in range(2_000):
        m = int(rng.integers(1, 8))
        ps = [round(float(x), 3) for x in rng.random(m) * 0.2]
        assert stats.holm_correction(ps) == original(ps)


def test_holm_correction_on_002s_committed_p_values_is_unchanged():
    """The published artifact, not a synthetic fixture: every stored decision must still hold."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for path in sorted((root / "results" / "002").glob("*/*.json")):
        payload = json.loads(path.read_text())

        def walk(node):
            nonlocal checked
            if isinstance(node, dict):
                for value in node.values():
                    if (isinstance(value, list) and value and isinstance(value[0], dict)
                            and "p_value" in value[0] and "holm_significant" in value[0]):
                        ps = [c["p_value"] for c in value]
                        stored = [c["holm_significant"] for c in value]
                        assert stats.holm_correction(ps) == stored, path
                        checked += 1
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(payload)
    assert checked > 0, "no 002 Holm families found — this test would be vacuous"
