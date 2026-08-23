# Amendment 4 to protocol-002 — the p-value estimator, and the multiplicity marker

**Status: frozen on tagging as `protocol-002-amendment-4`.** Written after the change was made
and the artifacts regenerated, which is the honest description of the order and is stated here
rather than implied otherwise. The change itself is dated 2026-08-23; this document is dated the
same day, after a review council found that no amendment had been written for it.

That gap is the first thing this amendment records. `protocols/002-ladder.md` says: "Any change
after that tag is an amendment, made in its own commit with a stated reason." A statistical
estimator on a published protocol was replaced, and the reason went into `results/002/corrections.md`
and a code comment instead of here. Both are honest and neither is what the protocol promises a
reader who follows the amendment trail. Amendments 1 to 3 were tagged before the runs they
governed; this one was not, and no claim is made that it was.

## 1. What changed

`rb.stats.paired_bootstrap` computed its two-sided p-value as `min(1, 2 * min(below, above))`
over the bootstrap distribution. When no resample crossed the null both counts were zero and the
function returned exactly `0.0`. 002's eight analysis artifacts carried 75 such values.

It now returns `rb.stats.bootstrap_p_value`, the add-one estimator of Davison & Hinkley (1997)
and Phipson & Smyth (2010): `2 * min(c_le + 1, c_ge + 1) / (B + 1)`.

## 2. Why this is a correction and not a revision of section 5

Section 5 registers "paired bootstrap (`rb.stats.paired_bootstrap`) on the per-query difference
between adjacent rungs: 2000 resamples, seed 20260818", and Holm at family-wise alpha 0.05. It
registers the procedure, the resample count and the seed. It does not register the arithmetic by
which the tail count becomes a p-value, and a p-value of exactly zero was never a value that
procedure could produce — B resamples cannot support a probability below the resolution B gives
you. The naive form was therefore not an alternative reading of section 5. It was wrong under it.

## 3. What moved, measured rather than asserted

All eight artifacts were regenerated from the committed `per_query.jsonl` scores.

| | |
|---|---|
| p-values changed | 137 — all of them, not only the 75 zeros |
| p-values still exactly 0.0 | 0 |
| Holm decisions moved | 0 of 24 |
| query-property bins crossing p = 0.05 | 0 of 113 |
| non-p-value changes | 7, all wall-clock timing fields |

The 24 are asserted by `tests/test_002_holm_decisions_unmoved.py` against a fixture freezing the
decisions as published. The 113 were checked independently during review. The seven are
`cost/scoring_seconds`, `cost/total_seconds` and `cost/index_build_seconds`, which record how long
a run took and therefore differ on any re-run; `lexical_factorial.json` is consequently not
byte-reproducible, and the reproduce claim covers the measurements rather than the file.

## 4. The multiplicity marker, added at the same time

`query_property_win_loss` now carries a `multiplicity` object recording `corrected: false`, the
p-value count, and a note that these bins are exploratory and no published claim rests on any one
of them.

This changes no number. It exists because 113 uncorrected p-values sat in the same artifacts as 24
Holm-gated ones with nothing in the data distinguishing them. Section 5's registered family is the
adjacent-rung comparisons, and amendment 1 section 7 registers the bins as win rate "with
intervals" — no per-bin decision is asserted anywhere, so no correction is owed. The marker states
that where the numbers live rather than only in prose a reader may not have.

## 5. What is NOT changed by this amendment

No registered prediction. No conclusion. No control threshold. No seed, resample count, or alpha.
The percentile-index convention at `stats.py:66` is untouched and remains a named follow-up; it is
provably identical to the derived rule at every round count this protocol registers.
