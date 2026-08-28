# Protocol 005 amendment 2 — a confirmatory test for the hub explanation

**Status: frozen on tagging as `protocol-005-amendment-2`.** Written and committed **before the
capped arms are built or scored**. The registered 2×2 in `protocol-005` is complete and unchanged;
nothing below revises it, and no decision already recorded in `results/005/analysis.json` moves.

## Why this exists

Stage 1's registered result was `supported` on all eight decisions. Two things then undercut the
reading of it:

1. Prediction B is close to circular. Its unaffected stratum is 98–99.5% exactly zero, which is
   structural for a personalized PageRank seeded from the query's own entities.
2. The exploratory dose-response is **negative on all four cells** — among affected queries, a
   larger reachability change predicts a *worse* outcome, the opposite of the mechanism.

The proposed explanation is hub merges. The registry contains a small number of aliases whose
resolution collapses a very large document set onto one node — `america → united states` moves
about 6,669 documents on HotpotQA — and node specificity, which is 1/document-frequency and which
HippoRAG's own ablation shows is load-bearing, goes to nearly zero for such a node.

**That explanation is currently correlational.** Hubs exist and the correlation is negative. Nobody
has shown the hubs cause it.

## The test

Rebuild the registry with one additional rule, and re-score.

**R9 — document-frequency cap.** A canonical whose merged document frequency would exceed **1% of
the corpus** is removed from the registry entirely, and every alias pointing at it reverts to
string identity. R1–R8 are otherwise unchanged.

1% of the pools is 665 documents on HotpotQA and 434 on 2Wiki. Both numbers follow from corpus
sizes fixed since 003; neither was chosen by trying values.

## The registered prediction, and why it is falsifiable

The cap is **not symmetric across the corpora, and that asymmetry is the test.**

HotpotQA's hub merges move about 6,845 documents — roughly 10% of its corpus, an order of
magnitude above the cap. 2Wiki's largest merge moves 67 documents, well below it. So:

> **Capping raises HotpotQA's typed arms and leaves 2Wiki's essentially unchanged, and the
> dose-response correlation on HotpotQA becomes less negative.**

- **If capping helps HotpotQA and not 2Wiki:** the hub explanation is confirmed. The corpora
  differ in outcome because they differ in hub content, which is exactly what the uncapped result
  implied but could not establish.
- **If capping helps both:** the effect is not about hubs; it is something general about merging
  less, and the hub story is wrong.
- **If capping helps neither, or hurts:** the negative dose-response has a different cause
  entirely — most likely that large-dose queries differ from small-dose queries in ways unrelated
  to merging, such as baseline difficulty. The hub explanation is then withdrawn, not softened.

Statistics as `protocol-005` §4: paired bootstrap on per-query R@2, B = 10,000, seed 20260820,
Holm across the family of new comparisons.

## Status of the result this produces

**Confirmatory for a post-hoc explanation, not a registered finding of the original experiment.**
The hub explanation was formed after seeing the dose-response. Pre-registering its test before
running it is what stops the explanation from being unfalsifiable, but it does not convert it into
something 005 predicted in advance, and the entry must not present it as one.

## What is deliberately not done

**The cap is not tuned.** One threshold, declared here, run once. If 1% turns out to be a poor
choice, that is reported as a poor choice rather than replaced by a better one — a swept threshold
would make the confirmatory test a fitting exercise.

**The capped arms do not replace the registered arms.** `results/005/analysis.json` stands as the
registered result. The capped arms are reported beside it under their own heading.
