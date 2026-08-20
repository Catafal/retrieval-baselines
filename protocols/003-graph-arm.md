# Protocol 003 — does a graph win where the bridge entity is missing

**Status: frozen on tagging as `protocol-003`.** Written and committed before the first entity is
extracted and before any graph retrieval is scored.

## 1. Why this exists

Experiment 002 measured a loss it could not explain. On HotpotQA dense retrieval loses to full
BM25 by 0.130 nDCG@10 and loses in every query-overlap quartile. The entry names its suspected
mechanism — the entity connecting the two gold documents is absent from the question text — and
admits the design cannot isolate it, citing Sciavolino et al. (EMNLP 2021).

A graph is the instrument that tests exactly that. This protocol is not a survey of what a graph
arm buys; run that way it measures our extractor rather than graphs.

**Not GraphRAG.** A knowledge graph is a data structure. Microsoft's GraphRAG (arXiv 2404.16130)
is a query-focused summarisation system whose paper measures LLM-judged comprehensiveness of
generated answers. It emits no ranked document list and this harness cannot score it. Reporting
its loss here would be the Armstrong (CIKM 2009) weak-baseline strawman aimed at someone else's
method.

**Prior art is HippoRAG** (Gutiérrez et al., NeurIPS 2024, arXiv 2405.14831): OpenIE-built KG,
retrieval by Personalized PageRank from query-matched entities, evaluated on HotpotQA / MuSiQue /
2WikiMultiHopQA with passage recall@k.

## 2. The adverse prior, registered before running

HippoRAG Table 2, HotpotQA, R@2/R@5:

| system | R@2 | R@5 |
|---|---|---|
| BM25 | 55.4 | 72.2 |
| Contriever | 57.2 | 75.5 |
| ColBERTv2 | **64.7** | **79.3** |
| HippoRAG (Contriever) | 59.0 | 76.2 |
| HippoRAG (ColBERTv2) | 60.5 | 77.7 |
| HippoRAG w/ REBEL OpenIE (Table 5) | 43.9 | 59.2 |

Three facts are registered in advance because they run against this experiment.

**HippoRAG beats BM25 on HotpotQA** (60.5 vs 55.4 R@2). Any claim that "graphs lose to BM25" is
not supported by the prior art.

**The graph makes its own base retriever worse here** — ColBERTv2 64.7 → 60.5 R@2, QA F1 57.7 →
55.0 (Table 4) — while on 2WikiMultiHopQA the same graph lifts ColBERTv2 59.2 → 70.7. The paper
attributes this to HotpotQA's "lower knowledge integration requirements", calls HotpotQA "a much
weaker test for multi-hop reasoning due to many spurious signals", and reports its distractors are
"not very effective".

**Extractor quality dominates.** REBEL OpenIE drops HotpotQA to 43.9 R@2, below BM25. Our
extractor is deterministic spaCy NER, which performs a *different task* (named-entity recognition,
not OpenIE triple extraction), so its position relative to REBEL is measured in §8, not assumed.

This experiment therefore tests a subgroup hypothesis against an unfavourable aggregate prior, on
a corpus its own source paper calls weak, with an extractor weaker than one that already lost. A
positive subgroup finding is held to the higher bar in §7; a negative one is not.

## 3. Corpus

**HotpotQA distractor setting: all 7,405 questions, 66,581 unique passages.** Every arm is scored
against this identical candidate set.

Measured before tagging, from the datasets themselves, with nothing retrieved and nothing scored:

| Property | Value |
|---|---|
| Questions | 7,405 — question-id set identical to BEIR `hotpotqa` test qrels, 7,405/7,405 both directions |
| Pooled corpus | 66,581 unique passages, from 73,700 title slots |
| Pool ⊆ BEIR `hotpotqa` corpus | 66,581/66,581 = 100%, zero duplicate-title collisions |
| Gold titles vs BEIR gold ids | 7,405/7,405 = 100% agreement |
| HippoRAG's 1,000-question sample | strict subset, 1,000/1,000 |
| Gold `type` labels | 5,918 bridge / 1,487 comparison |

**Why not BEIR's full 5,233,329 documents.** Full-corpus extraction is unaffordable, which forces
re-ranking over a pool built from the union of our own arms' top-k. That pool is circular: where
BM25 and dense fail to surface the bridge document the graph cannot recover it regardless of
mechanism, and the bias suppresses precisely the coverage-1 advantage this protocol predicts. The
distractor pool's candidate set is defined by the dataset instead, which is what makes the
restriction principled rather than circular.

**Why not HippoRAG's 9,221-passage corpus as the main setting.** Only 1,000 questions; §6 shows
that is statistically hopeless for the differential.

**Why not 2WikiMultiHopQA.** Measured: its coverage classes are 507 / 493 with **no coverage-0
class at all**. It cannot test "the bridge entity is absent from the query"; it can only test
whether the graph works where topology favours it. The exclusion is structural, not budgetary.
It is also absent from BEIR.

**The cost, accepted and declared.** Restricting the candidate set makes retrieval easier.
**003 does not reproduce 002's −0.130 and does not extend or explain that number.** It tests the
hypothesis 002 raised, in a setting where testing is possible. Same questions, same document ids,
easier candidate set. This is stated in the entry body, not a footnote.

## 4. The query class, frozen here

For each query, count how many of its gold documents have a title occurring in the query text as a
contiguous, case- and punctuation-normalised token sequence. `gold_title_coverage` ∈ {0, 1, 2};
every query has exactly 2 gold documents.

| coverage | meaning | queries |
|---|---|---|
| 0 | neither gold title named | 1,821 |
| 1 | one named, one must be reached — the classic bridge | 3,634 |
| 2 | both named — comparison question | 1,950 |

**Normalisation, frozen.** Wikipedia titles carry disambiguating parentheticals — *"Kiss and Tell
(1945 film)"* — in 14.0% of gold titles and never in a natural question.

- **Primary:** strip one trailing parenthetical, then match.
- **Sensitivity:** exact title, unstripped.

The two disagree on **15.4% of queries (1,138/7,405)** — larger than any plausible retrieval
effect. Both are registered here and **both are reported whatever they show**. If the finding
survives under only one, the entry says that in those words.

**Corroboration, not validation.** Against HotpotQA's own human-assigned `type`:

| | cov 0 | cov 1 | cov 2 |
|---|---|---|---|
| bridge | 1,821 | 3,632 | 465 |
| comparison | 0 | 2 | 1,485 |

Agreement 6,938/7,405 = 93.7%. Both labels derive from the same gold-document set, so this is
corroboration between overlapping operationalisations and is reported as such.

**Coverage is primary; `type` is a robustness filter only.** The 465 type-bridge / coverage-2
questions are why: their answer chain is a bridge, yet both titles are named, so nothing is hidden
for traversal to find. The claim under test is about surface-form absence.

## 5. Metrics

**Primary: Recall@2 and Recall@5.** Every query has exactly two gold documents, so nDCG@10 spends
eight of ten rank positions scoring against relevant documents that do not exist. R@2/R@5 is also
the family the prior art reports and therefore the only one the closure control can check against.

**Secondary: nDCG@10**, retained for continuity with 001 and 002.

Adding R@2/R@5 is additive **and scoped to this experiment**. `rb.metrics.MEASURES` — what 001
and 002 published against — is left unchanged, and 003 passes its own set instead. Growing the
global set would not have moved any measured value, because trec_eval evaluates each measure
independently; it would have changed the SHAPE of artifacts already published, including the
output of `make reproduce`, which the Makefile records as a published promise. Every committed
`summary.json` is regression-tested to still carry exactly the three published measures.

## 6. Power, measured before tagging

The differential statistic was run as a **placebo on the dense arm** — a non-graph arm, which
should show nothing — over 002's committed per-query artifacts:

> difference-of-differences, coverage-1 minus coverage-2, nDCG@10:
> **+0.0291, CI95 [−0.0441, +0.1034], p = 0.447**, n cov1 = 248, cov2 = 130, at 002's 500-query
> subsample.

No evidence a non-graph arm manufactures the predicted pattern. But the interval is **0.148 wide**,
so any true differential below roughly ±0.07 is undetectable at ~500 questions, and ~1,000 is
barely better. **Hence all 7,405 queries are scored** — about 14.8×, shrinking the interval ~3.9×
to ~0.038, an MDE near ±0.019.

Per-class BM25 baselines from the same artifacts, for the headroom control in §7:

| coverage | BM25 nDCG@10 | headroom |
|---|---|---|
| 0 | 0.4607 | 0.5393 |
| 1 | 0.5896 | 0.4104 |
| 2 | 0.6576 | 0.3424 |

BM25 improves monotonically with coverage, independent evidence that the class captures real
lexical difficulty. Coverage-2 is not near ceiling, so the headroom confound is real but modest.

## 7. The prediction, and how it is adjudicated

**Registered expectation (not a numbered falsifier).** The graph arm does not beat full BM25
overall. This is an expectation about *our extractor* — see §2 and §8 — not a claim about graphs,
and it is not counted as a falsifier in either direction.

**Prediction A — the differential. The claim the entry lives on.** The graph arm's advantage over
full BM25 is greater on bridge-absent queries than on coverage-2 queries.

- Contrast: coverage ≤ 1 pooled versus coverage 2, on the primary class definition.
- Statistic: paired per-query difference (graph − BM25) per class; difference of the two class
  means; percentile bootstrap over queries, B = 10,000, seed 20260820.
- Decision rule: CI95 excluding zero **and** a point estimate of at least +0.02, the MDE
  established in §6. A direction without that margin is reported as not resolved.
- Reported for R@2 (primary), R@5 and nDCG@10, under both class definitions. Holm correction
  across the family: {A, B} × {R@2, R@5, nDCG@10} × {primary, sensitivity}.

**Prediction B — the negative control.** On coverage-2 queries the graph arm shows no advantage
over full BM25: CI95 overlapping zero, or negative. **A graph arm that wins everywhere is a bug,
not a triumph**, and this is what makes that visible.

**Headroom control, reported always.** Every delta is published beside the raw per-class BM25
baseline and a headroom-normalised secondary statistic, so a reader can judge whether a
differential is a mechanism or an artifact of available room.

**Coverage-0 is exploratory and excluded from the falsifiers**, for a stated reason rather than by
convenience: with neither gold title named the graph may have no reliable anchor either, so the
mechanism does not predict a win there. It is reported in full.

**Higher bar for a positive.** Given §2's adverse prior, a positive differential must clear the
decision rule under **both** class definitions to be reported as a finding. Clearing only one is
reported as unresolved, with both numbers shown.

## 8. Closure controls — two, because the extractor differs in kind

The first BM25 closure control in this repo gated against an in-repo `bm25s` anchor at 0.02
tolerance and **failed a correct implementation**: the 0.0258 gap was tokenisation, not noise. The
fix was to gate against an external published figure. A draft of this protocol walked into the same
trap from the other side, proposing a band between HippoRAG's REBEL (43.9) and GPT-3.5 (60.5) rows
— a band that already contains BM25 (55.4), Contriever (57.2) and both HippoRAG variants. Only
ColBERTv2's 64.7 falls outside it. A control almost every published system passes is not a control,
so it is discarded.

**8.1 Harness closure — gating.** Run BM25 over HippoRAG's nested 1,000-question subset against
their 9,221-passage corpus and compare to Table 2's **BM25 row** (55.4 / 72.2) at tolerance 0.05
absolute on R@2 and R@5. This is an external number computed under matching conditions, and it
checks our indexing, scoring and pooling. Failure halts the run before anything is written.

**8.2 Extraction closure — gating the graph arm.** Intrinsic extraction quality is measured
against a hand-annotated sample of 100 passages drawn with seed 20260820, and reported as its own
number **before any retrieval score exists**. The draw is deterministic and the drawn ids are
committed at `results/003/extraction-sample.jsonl` so a reader can check that the annotated
passages are the ones the seed selects. The `entities` field ships **empty and is not pre-filled
by any model**: seeding it with a model's guesses would make this control measure agreement rather
than extraction quality. Annotation is by the author and must be complete before tagging. Precision and recall of extracted entities against
the annotation, published as an artifact. This is what gates the graph arm, because spaCy performs
named-entity recognition and HippoRAG and REBEL perform OpenIE triple extraction; no published
retrieval row is a like-for-like reference for our arm.

**Table 2 rows are reported as context, never as a gate for the graph arm.** The entry states
plainly that no published figure exists for a spaCy-NER graph on this corpus rather than
manufacturing one.

**8.3 Inherited.** `gold_presence` on the pooled corpus, and the pool-construction assertions in
§9, both halt the run on failure.

## 9. Pool construction, verified in code

Deduping 73,700 title slots to 66,581 uniques and mapping them onto BEIR document ids is exactly
the indexing step that produces unreproducible numbers, and this repository exists because of a
retraction for unreproducible numbers. The counts in §3 are therefore re-asserted in code and not
trusted from this document: unique-passage count, the 100% subset property, zero title collisions,
and the 100% gold-title-to-BEIR-id agreement. Each is a control that halts the run.

## 10. Everything else unchanged

Same `Retriever` seam, same scoring path, same paired bootstrap and Holm machinery, same artifact
layout, same environment manifest, all inherited from 002. The corpus loader gains a title
accessor used only for the class computation; no arm receives titles as a retrieval signal, and
the string every retriever is scored on is unchanged.

## 11. Stopping rule

Published whatever comes out, including and especially a null. The paragraph that ships if the
differential does not resolve is **written before tagging** so that a null is reported rather than
spun.

Two limitations ship with the entry, stated plainly rather than discovered by a reader:

**The extractor gap.** A deterministic extractor cannot settle whether a better one changes the
conclusion. That test is filed as NB-20 / experiment 004 before this one runs.

**Construct validity of the smaller haystack.** Part of HippoRAG's thesis is that graphs help when
retrieval degrades in a large noisy corpus; cutting the haystack 78.6× may remove some of the
condition the graph needs, independent of statistical power. Mitigating evidence: HippoRAG measured
its own gains on a 9,221-passage pool.

## 12. The null paragraph, written before the run

Required by §11 and written here, before any entity is extracted, so that a null is reported
rather than spun. If prediction A does not resolve — the CI95 includes zero, or the point estimate
falls short of the +0.02 margin — the entry publishes the following, adjusted only for the actual
numbers:

> **The graph arm did not resolve the question.** On bridge-absent queries its advantage over full
> BM25 was [X] and on comparison queries [Y], a difference of [Z] with a 95% interval of [L, U].
> The interval includes zero, so this experiment does not support the claim that graphs help
> specifically where the bridge entity is missing, and it does not refute it either.
>
> Three explanations remain open and this design cannot separate them. The extractor may be too
> weak: it performs named-entity recognition rather than the OpenIE triple extraction the prior
> art uses, and its measured extraction quality was [P precision / R recall] (§8.2). The corpus
> may be wrong for the question: HotpotQA's own authors call it "a much weaker test for multi-hop
> reasoning due to many spurious signals", and HippoRAG's graph makes its own base retriever worse
> here. Or the mechanism may not exist — bridge-entity absence may simply not be what a graph
> repairs.
>
> The first of the three has a test already filed: Experiment 004 swaps the deterministic extractor
> for an LLM one and re-runs this identical protocol. That was filed before this entry ran, not
> after it failed. What this entry establishes either way is the instrument: a bridge-absence class
> that is computed mechanically, agrees with HotpotQA's own human labels on 93.7% of queries, and
> can be recomputed by anyone from the qrels and the query text.

**What must not appear in that paragraph**, stated now while it costs nothing:

- No claim that graphs do not help, in general or on this corpus. A null on one weak extractor is
  not a verdict on a family of methods, and 002's encoder swap is the standing evidence for why.
- No promotion of coverage-0, or of the sensitivity class definition, or of R@5 over R@2, to
  headline status because it happened to reach significance. The primary contrast, class and metric
  are fixed in §4, §5 and §7. Every registered figure is reported; which one leads does not change
  after the fact.
- No quiet omission of the overall verdict. The registered expectation in §7 is reported whether
  or not it held.

**If prediction B fails** — the graph arm shows a real advantage on coverage-2 queries, where
nothing needs traversing — that is reported as a probable defect in the arm rather than as a
finding, and the entry does not report prediction A as supported until the cause is found. A graph
that wins where the mechanism cannot operate is evidence about our code, not about graphs.

