# Experiment 002 — corrections

Every figure in 002 that changed after it was published, what changed it, and why the change
was made.

This file exists because the alternative is a quiet edit. Entry 001 was retracted for publishing
numbers with no reproducible command behind them, and the corollary the retraction did not state
is that when a published number moves, the move itself has to be reproducible: which value, from
what to what, and why.

**No correction listed here changes a registered conclusion.** All 24 Holm decisions across the
eight analysis artifacts are identical before and after, and that is asserted by a test rather
than by this sentence: `tests/test_002_holm_decisions_unmoved.py`, against a fixture that freezes
the decisions as published.

---

## C1 — 75 p-values of exactly 0.0 (2026-08-23)

**What was published.** `paired_bootstrap` computed the two-sided p-value as
`min(1, 2 * min(below, above))` over the bootstrap distribution. When no resample crossed the
null, both counts hit zero and the function returned exactly `0.0`. 002's eight committed
analysis artifacts carried 75 such values.

**Why it is wrong.** A p-value of exactly zero is a claim no finite resampling procedure can
make. With B resamples the smallest statistic the procedure can support is bounded away from
zero. Reporting 0.0 states more certainty than the experiment purchased.

**The correction.** The call site now uses `bootstrap_p_value`, the add-one estimator of
Davison & Hinkley (1997) and Phipson & Smyth (2010): `2 * min(c_le + 1, c_ge + 1) / (B + 1)`.
That estimator already existed in the same module and 003 already used it. Only this call site
was left on the naive form, deliberately and with a comment saying so, because changing numbers
in a published entry is an author's decision rather than a refactor's.

**What moved, measured rather than asserted.** All eight artifacts were regenerated from the
committed `per_query.jsonl` scores. Comparing every leaf value before and after:

| | |
|---|---|
| p-values changed | 137 (every one, not only the 75 zeros) |
| p-values still exactly 0.0 | 0 |
| Holm decisions moved | **0 of 24** |
| non-p-value changes | 7, all wall-clock timings |

The seven are `cost/scoring_seconds`, `cost/total_seconds` and `cost/index_build_seconds`, which
record how long a run took and therefore differ on any re-run. Nothing published reads them:
`sync-002-results.mjs` does not touch the cost block. Worth stating plainly all the same, because
it means `lexical_factorial.json` is not byte-reproducible and the reproduce claim covers the
measurements, not the file.

**Which pass should have caught it.** NB-25 found it and NB-26 confirmed it, both before this
correction; the defect was documented in `stats.py` and left in place as a named follow-up. It
was not an unknown, it was a deferral, and it was deferred longer than it should have been.

**Reproduce.**

    python -m rb.experiments.ladder.run --dataset {scifact,quora,hotpotqa} --rung lexical-factorial
    python -m rb.experiments.ladder.run --dataset {scifact,quora,hotpotqa} --rung dense-hybrid-analysis --encoder {minilm,bge}

HotpotQA's factorial is the expensive one at ~50 minutes; the dense/hybrid analyses reconstruct
from committed artifacts and take seconds.
