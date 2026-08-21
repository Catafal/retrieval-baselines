# Amendment 1 to protocol-003 — §2b's prediction was wrong

**Status: written after `protocol-003` was tagged, and recorded as an amendment rather than as an
edit to §2b.** The tagged text stays exactly as it was. That is the whole point of tagging it.

## 1. What was predicted

§2b, written while building the arm and frozen in the tag, recorded a failure mode found before
any scored run: spaCy segments the same name differently by context, so the graph holds `kiss` and
`tell` while a query offers `kiss and tell`, and exact string linking matches neither.

From that, §2b predicted:

> **What this predicts, in advance.** The seed-match rate — the fraction of queries whose entities
> link to at least one node — will be low, and the arm's overall loss (§7's registered
> expectation) is partly attributable to linking rather than to graphs.

## 2. What was measured

| quantity | value |
|---|---|
| queries scored | 7,405 |
| queries with at least one linked entity | 6,096 |
| **seed-match rate** | **0.823** |

**The prediction failed.** 82.3% is not low by any reading of the word.

## 3. Why it failed, stated so the mistake is useful rather than merely admitted

The reasoning in §2b was correct about the mechanism and wrong about its consequence, because it
reasoned per-entity and the seed is per-query.

Span instability is real: `Kiss and Tell` genuinely fails to link, and the §8.2 error analysis put
83% of false positives on boundary disagreements, which is the same defect measured a second way.
But a HotpotQA question usually names several entities, and the walk needs **only one** of them to
land on a node. A per-entity failure rate that would be crippling for a single-entity query is
survivable when there are three or four chances per question.

The prediction conflated "this entity often fails to link" with "queries often fail to seed". Those
are different quantities and only the first was observed.

**The unquantified wording made it worse.** "Will be low" has no threshold, so the prediction was
weakly falsifiable by construction. Any future prediction in this sequence gets a number, in the
same way §7's differential was given an explicit +0.02 margin rather than a direction alone.

## 4. What does NOT change

§7's registered expectation — that the arm does not beat full BM25 overall — is untouched. It was
always an expectation about the extractor rather than about graphs, and this amendment removes one
of the reasons offered for it, not the expectation itself.

The §8.2 diagnostic stands as reported: precision 0.684, recall 0.657.

The clause in §2b that survives is the one about attribution: part of any loss remains
attributable to linking. What is now known is that the attribution is smaller than §2b assumed,
because most queries do seed.

## 5. What this changes about interpreting a loss

Before this measurement, a poor result could have been dismissed as "the arm never got a seed".
It cannot. The graph reaches a seed for 82.3% of queries, 99.1% of documents carry at least one
entity, and 67.8% of gold document pairs share an entity against a 6.5% random-pair baseline.

**The preconditions are met.** If the arm then loses, the loss is about the graph and the walk
rather than about the plumbing — which is a considerably more interesting entry than the one §2b
was bracing for.
