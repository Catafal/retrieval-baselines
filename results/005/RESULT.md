# Experiment 005, Stage 0 — what the identity registry covers

**Protocol:** `protocols/005-identity.md`, tagged `protocol-005-identity` before the first
Wikipedia request. Amended once, before any retrieval number, by
`protocol-005-identity-amendment-1`.

**No retrieval number appears here.** Section 8 forbids one at this stage. Stage 0 answers one
question: is there enough identity coverage on these corpora for the mechanism claim to be
testable at all?

**Answer: yes, in all four cells, and it was nearly reported as no.**

---

## The registry

Wikipedia redirects targeting pool titles, fetched once and committed with a manifest.

| corpus | pool titles | with ≥1 redirect | aliases | requests | failed |
|---|---|---|---|---|---|
| HotpotQA | 66,581 | 40,786 (61.3%) | 224,178 | 1,342 | 0 |
| 2Wiki | 43,487 | 24,402 (56.1%) | 82,768 | 870 | 0 |

Construction drops, per section 4:

| corpus | aliases seen | R5 self-title | R6 ambiguous | identity | kept |
|---|---|---|---|---|---|
| HotpotQA | 196,465 | 337 | 98 | 12,517 | 183,513 |
| 2Wiki | 74,557 | 132 | 32 | 5,356 | 69,037 |

Two things worth reading off that table. **Ambiguity is rare** — 98 aliases out of 196,465 point
at two pool documents. The `Cleveland State` university-versus-place failure NB-23 registered as a
precision risk is real but small at this scale, which is a finding about the corpora rather than a
vindication of the guard. And **12,517 aliases normalise to their own canonical**: redirects that
differ only in case or punctuation, which exact-string matching already handled. They are dropped
as non-merges, and their size is the reminder that a raw alias count would badly overstate what
identity actually adds.

## Coverage

| corpus | extractor | nodes | merged away | C1 nodes resolving | C5 alias-affected queries | MDE |
|---|---|---|---|---|---|---|
| HotpotQA | spaCy | 285,013 | 6,572 | 3.88% | 1,217 (16.4%) | 0.0568 |
| HotpotQA | GLM | 284,899 | 13,439 | 5.39% | 1,316 (17.8%) | 0.0546 |
| 2Wiki | spaCy | 173,124 | 3,353 | 3.53% | 414 (4.2%) | 0.0974 |
| 2Wiki | GLM | 179,091 | 4,972 | 4.27% | 1,085 (11.0%) | 0.0601 |

**The intervention is small.** Between 3.5% and 5.4% of graph nodes resolve through an alias, and
the node set shrinks by under 5%. Typed identity is not a rebuild of the graph; it is a modest
edit to it. Whether an edit that size moves retrieval is exactly what Stage 1 is for, and nothing
in this table predicts it.

**The floor is met in every cell.** No coverage threshold was declared in advance, on purpose. The
registered floor is the minimum detectable effect at the observed subset size, and it lands
between 0.055 and 0.097. The weakest cell is 2Wiki under spaCy, where 414 affected queries can
only resolve an effect of about ten points. The strongest three can resolve six.

## The correction that changed the answer

Section 6 originally defined the alias-affected subset as *queries with at least one entity
resolving through an alias*. Implemented literally, that reading gave this for 2Wiki under GLM:

| | narrow | corrected |
|---|---|---|
| alias-affected queries | 49 (0.5%) | 1,085 (11.0%) |
| MDE at 80% power | 0.283 | 0.060 |

**A factor of 22, and the difference between "untestable" and "testable".**

The narrow definition counts only queries that *name* an alias. It cannot see the other case, and
on 2Wiki the other case is almost all of them: the query names the canonical, a **document** named
an alias, and the merge makes that document reachable without the query entity ever being a
registry key. 2Wiki's questions are written from article titles, so its queries skew canonical,
which is precisely why the narrow figure collapses there and not on HotpotQA.

Had that been reported as written, Stage 0 would have concluded that redirect coverage does not
reach the forms that matter — the registered null in section 7 — on the strength of a measurement
artefact. The corrected definition is the mechanism's own: a query is affected when one of its
entities reaches a different set of documents under typed identity than under string identity.

Both figures are kept in the artifact. The gap between them *is* the amendment, and burying it
would make the correction invisible in exactly the place a reader would check.

## What Stage 1 inherits, and one thing it must not confound

Extraction quality and identity resolution are **not independent**. On 2Wiki the alias-affected
population is 4.2% under spaCy and 11.0% under GLM — the better extractor finds more entities, so
more of them have aliases to resolve. On HotpotQA the same comparison is 16.4% against 17.8%,
nearly flat.

So the interaction is real on one corpus and absent on the other, which is an argument for the
full 2×2 rather than against it: a design that swapped both at once would report a combined effect
and attribute it to whichever component the author had in mind. The cells are already scoped so
that cannot happen.

The alias-affected subsets above are fixed now, before any score exists. That is the population
Stage 1 decomposes over, and it cannot be reselected once it is known which queries improved.

## Reproducing this

    make reproduce-005-coverage

Offline. Replays the committed redirect snapshots and 004's committed extraction cache; spends
nothing and calls no model. The fetch itself is `make fetch-005-redirects` and is the only step
that touches the network.
