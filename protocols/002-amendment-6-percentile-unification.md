# Amendment 6 to protocol-002 — one percentile rule, four call sites

**Status: frozen on tagging as `protocol-002-amendment-6`.** Written with the proof attached,
before the change is reported anywhere. This amendment exists because the change touches the
statistics module 001 and 002 both publish against, not because any registered quantity moves —
none does, and section 4 below is the evidence rather than the assertion.

## 1. What changed

`rb.stats.paired_bootstrap`, `rb.stats.shapley_bootstrap`, `rb.stats.spearman_correlation` and
`rb.rescore.bootstrap_ci` each carried their own copy of the percentile-bound expression:

    means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]

All four now call `rb.stats.percentile_ci`, which derives both indices from the single rule in
`_percentile_index`: cut `int(tail * n)` draws from the end in question and return the outermost
draw that survives.

## 2. Why this is not a cosmetic refactor

The asymmetric `-1` was correct, and it was correct for a reason that had to be re-argued in a
comment at every site that copied it. Three sites copied it. A mutation sweep then found that two
of those copies — `shapley_bootstrap` and `spearman_correlation` — had no test at all: dropping
the `-1` at either survived the entire suite, while the identical mutation at `paired_bootstrap`
was caught.

That is how a defect class survives being audited. 003 was audited for exactly this defect, the
site under audit was fixed, and the copies were left behind because the audit was scoped by
location rather than by class. Both siblings are now covered by
`tests/test_sibling_percentile_bounds.py`, and after this amendment there is one implementation
for a test to cover rather than four.

## 3. Why it was deferred, and why the deferral stops applying

NB-26's council ruled against making this change inside a correctness fix scoped to 003's graph
arm: a cosmetic edit to the shared statistics module, bundled into a diff a reviewer must already
scan for moved numbers, buys a reviewer nothing and costs them attention. That reasoning was
right, and it is conditional. Done alone, in its own commit, with the proof below attached, the
objection does not apply. The deferral comment has been deleted from `stats.py` because it
stopped being true.

## 4. The proof that nothing moves

The two forms agree at every round count this protocol registers and diverge elsewhere, so the
no-op is a property of the registered configuration and not of the expressions:

| n | hand-written | derived rule | |
|---|---|---|---|
| 1,000 | (25, 974) | (25, 974) | agree |
| 2,000 | (50, 1949) | (50, 1949) | agree — 002's registered `BOOTSTRAP_ROUNDS` |
| 10,000 | (250, 9749) | (250, 9749) | agree — 003's registered count |
| 100 | (2, 96) | (2, 97) | **diverge** |
| 333 | (8, 323) | (8, 324) | **diverge** |

The divergence at unregistered `n` is stated deliberately: it is what makes the agreement above a
finding rather than a tautology, and it means this change would NOT be a no-op for any future
experiment that registers a round count where `0.975 * n` is not integral. The derived rule is the
more correct of the two by its own derivation, so such an experiment should use it — but it must
not silently inherit a claim of "no numbers moved" from this amendment.

Every published interval was also reconstructed from its committed inputs under both rules:

    results/002/scifact/dense_hybrid_analysis.json       old=(-0.051804, 0.020295)  new=identical
    results/002/scifact/dense_hybrid_analysis-bge.json   old=( 0.046386, 0.115583)  new=identical
    results/002/quora/dense_hybrid_analysis.json         old=( 0.085160, 0.139547)  new=identical
    results/002/quora/dense_hybrid_analysis-bge.json     old=( 0.099447, 0.152993)  new=identical
    results/002/hotpotqa/dense_hybrid_analysis.json      old=(-0.161217,-0.100546)  new=identical

Five reconstructed intervals, zero mismatches. No nDCG@10, no p-value, no Holm decision, no
comparison direction and no registered prediction is affected, in 001 or in 002.

## 5. What is not changed

The estimator, the resample count, the seed, alpha, the Holm family definition, and every control
threshold. Amendment 5's closure tolerance is unaffected. `_percentile_index`'s own derivation is
unchanged — this amendment routes callers to it, it does not modify it.
