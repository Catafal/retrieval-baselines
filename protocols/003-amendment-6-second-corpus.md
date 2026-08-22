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
