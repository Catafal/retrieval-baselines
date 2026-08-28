# Protocol 005 amendment 1 — the `underpowered` rule uses the wrong estimand

**Status: frozen on tagging as `protocol-005-amendment-1`.** Written after the registered analysis
ran, and after it was established that the rule this amendment concerns **was never reached**. The
decisions in `results/005/analysis.json` are unchanged by it.

## What §5 registered

> That cell reports `underpowered` rather than `no_advantage` when its interval includes zero and
> its width exceeds the MDE

with the MDE quoted from Stage 0, computed by `stats.mde_two_proportion`.

## Why that is wrong

`mde_two_proportion` is a **two-independent-proportion** power calculation: two arms of equal size
n, each a Bernoulli variable, worst-case variance 0.25.

The statistic it was being compared against is prediction B's — a difference between the means of
**two unequal strata** (n_affected between 414 and 1,316; n_unaffected between 6,089 and 8,740) of
a per-query recall difference taking values in {−1, −0.5, 0, 0.5, 1}.

Three mismatches, any one sufficient:

- **Wrong variance.** The empirical variance of the per-query differences is 0.011–0.030, far
  below the Bernoulli worst case of 0.25. In the other direction, a difference on a [−1, 1] range
  can in principle reach variance 1, four times that bound. The substitution is therefore not
  provably conservative in either direction, which is the property it would need to be usable as a
  floor.
- **Wrong design.** The formula assumes equal arms. Here one stratum is five to twenty times the
  other.
- **Wrong quantity.** An MDE derived for proportions is not commensurable with a CI width on a
  difference of means, even when both are dimensionless and both look like "about 0.06".

Found by a statistics reviewer reading `analysis005._subset_contrast` against `stats.py`.

## What it did to the published result: nothing, and this is checkable

The MDE appears in a single `elif` branch of `analysis005.run()`, reached only after both
significance branches fail — that is, only for a cell whose Holm-adjusted p exceeds 0.05.

**Every one of the eight decisions resolved on significance.** The largest Holm-adjusted p across
both predictions and all four cells is 0.0008, and every interval excludes zero. The
MDE-dependent branch was never entered, and no decision in `results/005/analysis.json` depends on
it.

The cell this rule was written for — 2Wiki under spaCy, registered in §5 as underpowered by design
at MDE 0.0974 — returned an observed subset contrast of 0.3501 with a 95% interval of
[0.3182, 0.3820]. It is not underpowered. The registered expectation was simply wrong, which §5
allowed for by making the label conditional rather than assumed.

## What replaces it

Nothing, for this experiment. The rule is retained in the artifact exactly as it ran, because
rewriting a registered decision procedure after the fact — even a defective one that provably did
not fire — is a worse failure than the defect.

**For any future experiment reusing this machinery**, the correct floor for a difference of
stratified means is derived from the observed per-stratum variances and sizes, not from a
proportion formula:

    MDE ≈ (z_{α/2} + z_{power}) · sqrt(s²_affected/n_affected + s²_unaffected/n_unaffected)

`stats.mde_two_proportion` keeps its name and its behaviour, since 004's query-extraction gate
genuinely is a two-proportion comparison and its published figures depend on it. What is wrong is
not the function; it is where 005 pointed it.

## Recorded here rather than fixed quietly

Because the same reviewer noted the defect cannot manufacture a `supported` result, only mislabel
a null one, the temptation is to treat it as cosmetic. It is not cosmetic in the case that
matters: an experiment that reported `no_advantage` where it should have reported `underpowered`
would be claiming evidence of absence from a sample that could not have spoken. That this
experiment happened not to reach that branch is luck, not design.
