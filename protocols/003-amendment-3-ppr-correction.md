# Amendment 3 to protocol-003 — the walk was not a PageRank

**Status: a bug fix, classified as such by Sir on 2026-08-21 and recorded before the corrected
run's numbers were seen.** The tagged protocol is unchanged. What changed is the code, which did
not do what the protocol registered.

## 1. What was wrong

`personalized_pagerank` computed `spread = B^T (B rank)` and rescaled the whole vector to sum 1.

`B^T B` is the entity-entity co-occurrence matrix. Two things were missing:

- **No degree normalisation.** A random walk divides mass leaving a node by that node's degree,
  which is what discounts a hub's individual edges *because* it has many. Without it the
  iteration is damped power iteration toward a degree-dominated eigenvector. The single global
  rescale fixes total magnitude and not the direction mass flows.
- **A self-loop weighted by popularity.** `(B^T B)_ii = df(i)`, an entity co-occurring with
  itself once per document it appears in. So a hub both received more mass and leaked less.

## 2. Why this is a bug fix and not a method change

§2 registers "retrieval by graph traversal, following HippoRAG's shape", and HippoRAG's shape is
Personalized PageRank. The function's own docstring claimed PPR. It was not PPR. Correcting it
brings the implementation into line with what was registered rather than changing what was
registered.

The distinction matters enough to state the limit of it: **only the walk is corrected.** Document
scoring still sums the PPR mass of a document's entities, unnormalised by entity count, even
though that carries a measured 1.45x bias toward entity-rich documents. Summing is what HippoRAG
does, so changing it would be a method change made after seeing results. It stays, and the bias is
reported.

## 3. How it was found, and what the evidence was

Not by a test — by asking why the arm scored 21.3 Recall@2 when HippoRAG's *deliberately
weakened* REBEL ablation scores 43.9 against a BM25 of ~55. Six lines of evidence, recorded in
`results/003/graph-arm-defect-report.json`:

1. Full PPR (0.2132) scored **below a baseline that did no propagation at all** (0.2307).
2. Seeded with an oracle — the anchor gold document's own entities — the walk found the partner
   document in the top 10 only 24.7% of the time, though 67.7% of those pairs share an entity.
3. Top-1 results averaged 13.42 entities against a corpus mean of 9.25, while gold documents
   averaged 8.29.
4. Against `networkx.pagerank` on a toy graph with a known answer, the walk put 2.5x less mass on
   the informative rare edge and ranked the only correct target outside the top 4.

The corrected walk now agrees with `networkx.pagerank` to 2e-6.

## 4. The test that should have existed

Every existing test passed against the broken walk, because each checked a property the broken
version also satisfied: mass reaches a bridged document, an unconnected document scores zero, the
seed retains most mass. **None compared against an independent implementation of the thing the
docstring claimed to be.**

`tests/test_graph_retriever.py` now does, and `networkx` is a test-only dependency for that
purpose alone. This is the same lesson 002 learned from a stub that could tie — a test that
cannot fail is not evidence — arrived at from a new direction: a suite of tests that are all
*consistent with* the bug is equally not evidence.

## 5. What is published

Both runs. The defective artifacts stay at `results/003/pool/graph-defective/` with a README
saying what they are. Retracting them silently would be a worse fault than the bug: a reader must
be able to see what was published, what replaced it, and by how much it moved.

## 6. What is NOT yet known at the time of writing

Whether the correction changes any conclusion. The corrected arm may still lose to BM25, and may
still show the registered per-class differential, or may not. This section is written **before the
corrected numbers exist**, so that the framing cannot be fitted to them.

The registered clauses A and B will be recomputed on the corrected run and reported beside the
defective ones. If the differential does not survive, that is the result.
