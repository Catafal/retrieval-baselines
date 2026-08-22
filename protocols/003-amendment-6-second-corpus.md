# Amendment 6 to protocol 003 — a second corpus, as a registered crossover

**Status:** registered and tagged **before any arm is scored on 2WikiMultiHopQA.** Nothing in this
document was written with knowledge of a 2Wiki result, and the git history is the evidence.
**Amends:** `protocols/003-graph-arm.md` §3 (corpus), §7 (prediction).
**Precedent:** `003-amendment-2` added the dense arm after the tag on the same terms.

## Why a second corpus, and why this one

§2 of the tagged protocol already registered the contrast this amendment tests, citing the source
paper against the experiment:

> The graph makes its own base retriever worse here — ColBERTv2 64.7 → 60.5 R@2 — while on
> 2WikiMultiHopQA the same graph lifts ColBERTv2 59.2 → **70.7**. The paper attributes this to
> HotpotQA's "lower knowledge integration requirements", calls HotpotQA "a much weaker test for
> multi-hop reasoning due to many spurious signals", and reports its distractors are "not very
> effective".

So the corpus was named in advance as the place the same method behaves oppositely. Running only
HotpotQA and stopping leaves the experiment testing graph retrieval on the corpus the prior art
says rewards it least, and reporting the result as though it were about graphs.

**This is the completion of the comparison, not a second attempt at it.** The claim under test is
not "the graph wins somewhere". It is that **the same arm, unchanged, behaves oppositely on two
corpora**, and that the direction of the difference was written down first.

## What stays frozen

One variable moves. Everything else is byte-identical to the HotpotQA run:

spaCy 3.8.13 / `en_core_web_sm` 3.8.0, the entity-type whitelist, exact normalised string linking,
damping 0.5, node specificity on, B = 10,000, seed 20260820, margin 0.02, both coverage
definitions, the Holm family shape, and the same scoring code. Experiment 004 moves the extractor.
This amendment moves the corpus. Neither moves both.

## Corpus construction

2WikiMultiHopQA dev, 12,576 questions. Same distractor-pool recipe as §3: per question, the ten
candidate passages from its `context` field, pooled over every question and deduplicated by title.

Two differences from HotpotQA, recorded because they are not cosmetic. There is no BEIR release of
2Wiki, so the corpus is built from the dataset's own `context` passages and document ids are minted
from titles rather than inherited; and gold documents come from `supporting_facts` rather than from
published qrels. Both are asserted by a §9-style control that halts the run, not assumed.

Counts will be frozen in this file before scoring. A corpus count is a property of the data, not a
result, so measuring it now does not contaminate the predictions below.

## The predictions

Stated in the **corrected shape** that experiment 003's own decomposition forced: on the graph
arm's **own between-condition difference**, not on a differential against a baseline. Stating them
in the old shape would make this follow-up unfalsifiable in exactly the way amendment 5 documents.

**Prediction C — primary.** On 2Wiki, the graph arm's own between-class difference on R@2
(bridge-absent minus coverage-2) has a 95% bootstrap interval **excluding zero** and a point
estimate of **at least +0.02**.

On HotpotQA this quantity was **+0.0065, CI [−0.0082, +0.0212]** — indistinguishable from zero.
Prediction C says that on a corpus built to require traversal, it is not.

**Prediction D — the crossover, and the demanding one.** The graph arm's R@2 deficit against BM25
is **at least 10 points smaller** on 2Wiki than on HotpotQA, where it is −0.3342 (0.2148 vs
0.5490).

**Not predicted: an outright win.** My extractor scores roughly 39 R@2 points below a competent
graph arm on HotpotQA (21.5 against HippoRAG's published 60.5), and I have no basis for expecting
that gap to close. A sign flip would be a strong result and it is not what is being claimed.

**Prediction E — the negative control travels.** On 2Wiki's coverage-2 queries the graph arm still
shows no advantage over BM25. A graph that wins where nothing needs traversing is a bug on any
corpus.

## What falsifies this

**A flat per-class profile on 2Wiki.** If the graph arm is as indifferent to coverage on the corpus
the prior art says rewards traversal as it was on the corpus the prior art calls weak, then the
mechanism is not merely under-served by a weak extractor — it is absent from this implementation,
and no change of corpus rescues it. That result is published with the same prominence as a
confirmation.

This is written down because it is the outcome that would cost the most, and registering only the
predictions one hopes for is the failure mode this whole sequence exists to document.

## Adjudication

Holm across the 2Wiki family on the same shape as §7. HotpotQA's published numbers are **not
recomputed and not re-corrected**; they stand as published, and the crossover is a comparison
between two runs rather than a re-analysis of one.

---

# Corpus counts and construction decisions — frozen before any arm is scored

Measured from the dataset only. No arm has been run on 2Wiki at the time of writing, which the
commit that follows this text evidences. Two construction decisions had to be made and both are
recorded here rather than discovered later, because either could be used to move a result.

## Decision 1 — restrict to questions with exactly two gold documents

2Wiki's dev split has 12,576 questions, of which **2,751 (`bridge_comparison`) have four gold
documents** and the remaining **9,825 have exactly two**.

R@2 is this experiment's primary measure, chosen in §5 because *"every judged HotpotQA query has
exactly two gold documents"*. On a four-gold question R@2 cannot exceed 0.50 however good the
ranking is, so pooling the two groups would make R@2 mean a different thing on 2Wiki than on
HotpotQA, and the crossover comparison would be between two different quantities.

The experiment therefore scores the **9,825 two-gold questions**: `compositional` 5,236,
`comparison` 3,040, `inference` 1,549. The four-gold `bridge_comparison` group is excluded, is
reported as excluded, and is available for a later entry that uses a cutoff suited to it.

This is a restriction that makes the arms comparable, not one that selects favourable questions:
it is defined by gold-set size alone, is applied identically to all four arms, and was fixed
before any of them ran.

## Decision 2 — one text per title, by majority variant

**1,242 titles (2.9%) appear with more than one text.** Inspected before deciding: the variants are
whitespace and tokenisation artifacts, not different documents — `Anhalt- Zerbst( 17 March 1540`
against `Anhalt-Zerbst (17 March 1540` — with a median pairwise similarity of **0.994** and a
minimum of 0.833 over a 400-title sample.

Rule: **the most frequent variant wins, ties broken lexicographically.** Deterministic, independent
of iteration order, and recorded as a control count rather than silently applied. HotpotQA's pool
required zero title collisions and raised on any; 2Wiki cannot meet that bar because its passages
are redistributed per question, so the weaker guarantee is stated instead of the check being
quietly dropped.

## Frozen counts — asserted by a control that halts the run

| quantity | value |
|---|---|
| questions scored | **9,825** |
| title slots pooled | **98,250** |
| unique passages (pool size) | **43,487** |
| titles resolved by the majority rule | **1,242** |
| distinct gold titles | **16,468** |
| gold titles present in the pool | **16,468 / 16,468** |

The pool is 43,487 passages against HotpotQA's 66,581, so 2Wiki is the smaller haystack. That
difference favours every arm equally and is stated because it is a difference between the two runs
that the crossover comparison does not control for.
