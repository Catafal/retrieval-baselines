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

---

## C2-C7 — found by an adversarial review council, 2026-08-23

The council was four independent seats plus a fresh validator, run over the code, the
statistics, the pre-registration record and the published entry. Its verdict on the numbers
was that they are sound: the bootstrap, Holm, Shapley, RRF and nDCG implementations are all
correct, one published comparison was reproduced bit-for-bit from committed per-query data at
the pinned seed, and no conclusion moves. Everything below is prose or test coverage. That
distinction is uncomfortable rather than reassuring, because this entry's own argument is that
the numbers are not the hard part.

### C2 — the entry contradicted its own table

"The bins in the overlap table hold 125 queries each." SciFact's hold 75; Quora's and
HotpotQA's hold 125, and the entry's own table shows the 75 about 130 lines earlier. Corrected
to state both.

### C3 — a pre-registration claim with nothing behind it

The entry described a "ceiling squash" failure mode, Quora's bounded scores letting capability
pile up rather than shift the gap, and presented it as registered in amendment 2 alongside the
flattening threshold. It is not there. An exhaustive search of all four protocol documents and
the repository's entire history finds no such passage; the word does not occur anywhere. The
paragraph now states the numeric falsifier that WAS registered, and says plainly that the
second mode was not.

This is the same class as the false verification claim retracted in 003: an assertion about
what was established in advance, with nothing traceable behind it. It is the second time, which
is why it is written here rather than quietly edited.

### C4 — a published claim false against its own artifacts

"Of the four properties I was allowed to look at, overlap is the only one that tracked the
answer on all three corpora." Using the entry's own decline measure, first bin minus last bin:

| property | scifact | quora | hotpotqa |
|---|---|---|---|
| gold_jaccard | 0.128 | 0.255 | 0.149 |
| max_idf | 0.070 | 0.088 | 0.118 |
| mean_idf | 0.096 | 0.177 | 0.087 |

Three of four decline on all three corpora. Overlap is the steepest, not the only one, and the
claim is now stated that way. A charitable reading was tested and also fails: gold_jaccard is
not monotone on any corpus either, so monotonicity does not rescue "only".

The mechanism matters more than the sentence. `spearman_correlation` exists in `rb.stats` and
is wired into no caller, so no trend statistic is computed for any property anywhere in this
repository. The word "only" was an impression, in an entry about the difference between an
impression and a measurement.

### C5 — the entry went stale the same afternoon it was corrected

C1's re-run moved HotpotQA's lexical `cost.total_seconds` from 2,671.9 to 2,971.3. The entry
quoted "roughly 2,600". C1 itself recorded that cost fields are not byte-reproducible and then
did not check whether the entry quoted one. Corrected, and the entry now says why the figure is
given round.

### C6 — C1's own account understated the defect and overstated its evidence

Three items, all from the correction written earlier the same day:

- The entry said the zeros were "on Quora and on HotpotQA". SciFact carried thirteen more, five
  of them inside its own Holm families. Counts: Quora 47, HotpotQA 15, SciFact 13 = 75.
- A comment added to `stats.py` said "27 Holm families". It is 24. It was written in the commit
  that fixed a defect about publishing numbers that are not real.
- The fixture docstring in `test_002_holm_decisions_unmoved.py` said its values were read from
  the artifacts "before the regeneration". The fixture landed in the same commit as the last
  regeneration, so the history does not show that. The content was afterwards verified
  independently against `a126b92`, genuinely pre-correction, and all 24 match. The claim now
  rests on that check rather than on an ordering the record cannot support.

### C7 — the direction of every comparison was untested, and untestable by the existing test

A mutation sweep found that swapping the two arms of the `dense_vs_full_bm25` comparison
survives the entire test suite, as does inverting the better-component selection.
`run_dense_hybrid_analysis`, `run_dense` and `run_hybrid` had no test referencing them at all,
and they produce every dense/hybrid artifact in this experiment.

The Holm fixture cannot cover this, and not by oversight. `paired_bootstrap`'s p-value is
symmetric under swapping its arguments: the add-one estimator's `min(c_le + 1, c_ge + 1)` is
invariant to negating every difference. An arm swap therefore leaves every `holm_significant`
boolean untouched and flips only the sign of `mean_diff`. Freezing Holm decisions is
structurally incapable of catching it.

`tests/test_002_comparison_directions.py` now freezes the direction of all 24 published
comparisons, and pins the symmetry itself so the file is not later deleted as redundant.
Verified by injecting the swap and regenerating an artifact: the Holm fixture passes, the
direction test fails.

The same sweep found the percentile off-by-one — 003's own defect class — surviving at
`shapley_bootstrap` and `spearman_correlation`, while the audited site at `paired_bootstrap` is
covered. Auditing by location rather than by defect class leaves the copies behind.
`tests/test_sibling_percentile_bounds.py` closes both, each mutation-verified in isolation.
