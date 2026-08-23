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

### C8 — the encoder swap is not the single-variable comparison the protocol claims

`protocols/002-amendment-2-second-encoder.md` section 5 says "The only variable is the encoder."
Three things differ between the two arms: architecture, the context window (256 word pieces for
MiniLM against 512 for BGE), and a query prefix BGE takes and MiniLM does not. Each follows the
respective model card, which is the right way to use each model, but the sentence asserts an
isolation the configuration does not have.

The tagged protocol is not edited — a frozen document that gets quietly corrected is worth
nothing. The entry's threats-to-validity section now states the confound and what it costs: the
gradient claim survives it, because the gradient is about the tilt of the line and truncation
moves its height; the corpus-level margin does not survive it cleanly. The clean comparison is
BGE truncated to 256, which has not been run.

Also in this pass: the "documents average 225 tokens" figure is carried from amendment 1's prose
and is not computed anywhere in this repository, so the entry now says so rather than presenting
it as measured. And `K1 = 1.2` / `B = 0.75` were pinned by no test — a mutation sweep set them to
any value with the suite still green, because the equivalence test compares two implementations
that read the same constants and therefore always agree with each other.
`tests/test_bm25_constants_pinned.py` closes that, and also asserts the constants still reach the
score, so hard-coding a value inside the scoring loop fails rather than passing both tests.

### A note on how C8 was found

It was not found by the council. The council reported it, and the fix summary claimed all twelve
of its findings were closed when eleven were. C8 is the one that was named, acknowledged, and
then not done, while being reported as done. The claim was checked afterwards and corrected. That
is the third time in this project that a completion claim has been published ahead of the work,
after 003's false verification claim and this file's own C6.

### C9 — the closure control could not fail, and the percentile rule had four copies

Two items the review council left open, closed together under amendments 5 and 6.

**The closure tolerance was 0.10 against deltas of 0.0045, 0.0212 and 0.0272** — 4x to 22x looser
than what it was gating. It is now 0.05, chosen by measuring what defects actually cost rather
than by preference: it sits above the largest legitimate difference with 1.8x headroom and below
the wrong-formula defects (IDF off 0.1085, TF saturation off 0.1075) with a 2.2x margin. At 0.10
that second margin was 1.08x, meaning the gate cleared the largest defect that exists by luck.

The finding underneath is more useful than the number. Tightening this does **not** let the
control catch subtle scoring defects, and implying otherwise would have been easy. A k1 or b drift
moves nDCG@10 by up to 0.0211 and the legitimate difference to the published figure is already
0.0272, so no threshold separates them; the same holds for length normalisation at 0.0148. Those
classes are below this control's resolution by construction. They are covered instead by
`tests/test_bm25_constants_pinned.py` and the factorial equivalence tests, and the artifact now
carries a `cannot_detect` field naming the blind spot where a reader will see it.

**The percentile expression had four copies**, and a mutation sweep had already shown two of them
were covered by no test. All four now route through the one derived rule. Mutating that rule fails
11 tests; before unification the identical mutation at `shapley_bootstrap` or
`spearman_correlation` failed none.

No published number moved, proven twice — by reconstructing five committed intervals under both
rules with zero mismatches, and then by regenerating all three lexical factorials and diffing them
leaf by leaf. The only non-wall-clock change in any artifact is `tolerance: 0.1 -> 0.05` and the
two added `cannot_detect` entries.

**One defect was introduced and caught during this work.** The percentile commit added a comment
citing `protocols/002-amendment-6-percentile-unification.md` before that file existed. The
amendment was written and tagged immediately, so the citation resolves, but a commit referencing a
document that did not exist is in the history. That is the same class as C6's fixture docstring and
003's verification claim: a reference asserted ahead of the thing it references, for the third time.

**A methodology defect was also found, affecting every mutation sweep in this project.** Restoring
a mutated file with `cp` sets its mtime to the current second; if the `__pycache__` entry was
written in that same second and the two candidate values are the same length, Python's
mtime-plus-size invalidation does not fire and a stale `.pyc` keeps serving the mutated code. The
mutation is then never loaded, the tests pass, and the sweep records **SURVIVED** — a false hole.
Kills remain trustworthy, since a test can only fail if the mutated code actually ran, and the
002 sweep's four survivors were each independently re-demonstrated afterwards. Future sweeps must
run under `PYTHONDONTWRITEBYTECODE=1`, as this one did after the discovery.
