"""
`paired_bootstrap` must not report a p-value of exactly zero.

A p-value of exactly zero is a claim no finite resampling procedure can make: with B
resamples the smallest observable two-sided statistic is bounded away from zero. The naive
form this call site used until 2026-08-23 returned 0.0 whenever no resample crossed the null,
and 002 published 75 such zeros.

This test is written to FAIL against that naive form. Verified by mutation, not by assumption:
restoring `2 * min(below, above) / rounds` at the call site makes the first assertion below
fail. A test that does not fail against the defect is not evidence that the defect is fixed.
"""
from rb.stats import BOOTSTRAP_ROUNDS, paired_bootstrap


def test_p_value_is_never_exactly_zero_when_no_resample_crosses():
    # Arm A beats arm B on every single query by a wide margin, so no resample of the
    # per-query differences can land at or below zero. This is precisely the input that
    # drove the naive estimator to 0.0.
    hi = [1.0] * 200
    lo = [0.0] * 200

    out = paired_bootstrap(hi, lo)

    assert out["p_value"] > 0.0, "a finite bootstrap cannot support p = 0"
    # The add-one estimator's smallest observable value for this shape: 2 * (0 + 1) / (B + 1),
    # at the module's registered round count. Pinned against BOOTSTRAP_ROUNDS rather than a
    # literal so the test states the relationship rather than a number that drifts with it.
    assert out["p_value"] == 2 / (BOOTSTRAP_ROUNDS + 1)


def test_identical_arms_still_report_p_of_one():
    # The opposite end, and the reason the naive form had a `min(1.0, ...)` cap: every
    # difference is zero, so every resample sits at zero and both tails are full.
    same = [0.4, 0.9, 0.1, 0.7] * 50

    assert paired_bootstrap(same, same)["p_value"] == 1.0
