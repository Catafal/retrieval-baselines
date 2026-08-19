# Amendment 1 to protocol-002 — the dense and hybrid arms

**Status: frozen on tagging as `protocol-002-amendment-1`.** Written and committed before any
embedding is computed. `protocol-002` covers the lexical factorial and is unchanged by this
document; this amendment adds the two arms that were always gated behind pinning an encoder.

## 1. Why this is an amendment rather than a new protocol

`protocol-002` section 5 specified a dense rung and a hybrid rung but deliberately did not pin
the encoder, on the grounds that pinning a version nobody had run would be a guess dressed as
rigour. That gate is now closed here, before anything is embedded.

## 2. What Experiment 002 asks, restated

Where does dense retrieval start beating lexical search, on which queries, and at what cost?

Reported against **two** lexical baselines, not one:

- the coordination matcher from 001, the naive floor, already measured;
- full BM25, the competent baseline, already measured as the all-on corner of the factorial.

Reporting both is the point. The commonest flaw in this comparison is a baseline nobody has
shown to be competently built, and the mechanism attribution already measured under
`protocol-002` is the evidence that this one is. A reader sees the floor, the real baseline and
dense, and locates the gap themselves.

## 3. Encoder, pinned

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Licence: Apache-2.0
- Library: `sentence-transformers==6.0.0`, pinned in `requirements.txt` before any embedding runs

Chosen for reproducibility rather than for performance. It is small enough that a stranger can
rerun this on a laptop, which is the property this repository optimises for, and it is among the
most widely used sentence encoders in existence, so the number is comparable to what most people
would actually get. A larger encoder would flatter the dense arm and make the result harder to
check. That trade is made deliberately and stated, and the entry must not present this arm as
the best dense retrieval available — only as what a typical, cheap, popular choice delivers.

## 4. Dense arm

- Exact cosine over float32 vectors. **No approximate index.** An ANN structure trades recall
  for speed and that loss would be reported as a property of the model.
- Truncation at the model's maximum sequence length, 256 word pieces for this model. Documents
  beyond it are truncated, which is a property of the measurement and is recorded, not hidden.
- Mean pooling, embeddings L2-normalised, as this model specifies.
- Batch size recorded in the manifest, and the manifest must report what was actually used.

## 5. Hybrid arm

Reciprocal rank fusion of full BM25 and dense, `k = 60`, fixed here before any hybrid number
exists. No weight search. A weighted fusion is a different experiment about fusion.

## 6. Corpora

All three: SciFact, Quora, HotpotQA. Same queries, same subsample, same seed as 001 and the
factorial, so per-query pairing across entries is valid.

HotpotQA is included rather than bounded out. Projected from this repository's own measured
runs, embedding it is roughly 109 minutes and 8.0 GB of vectors, which is one unattended run.
It is the multi-hop regime and the one where the literature says lexical is hardest to beat,
which makes it the most informative of the three, not the most expendable.

## 7. The query-property analysis, fixed in advance

For every scored query: which arm won, and does that track these four properties and no others.

1. Query length in tokens, using the scorer's own tokenizer.
2. Maximum IDF across the query's terms, and mean IDF, as measures of term rarity.
3. Lexical overlap between query and gold documents: Jaccard over token sets, averaged across
   that query's gold documents.
4. Number of gold documents, which separates single-hop from multi-hop.

**Four properties, and only these four.** Analysing whichever property turns out to correlate
with winning is how a fishing expedition produces a finding. Anything else that looks
interesting after the fact is reported as a suggestion for a later experiment, never as a
result of this one.

Reported as win rate for dense over full BM25 within bins of each property, with intervals,
alongside the paired per-query difference. Bins with too few queries are reported as sparse and
never dropped.

## 8. Statistics

- Paired bootstrap on per-query differences between arms, 2,000 rounds, seed 20260818, matching
  the existing convention. Comparisons: dense against full BM25, dense against coordination,
  hybrid against whichever of its two components scored higher.
- Holm correction across those comparisons within each corpus.
- Effect sizes in nDCG points. No ratio of two metrics where only one carries an interval.

## 9. Controls

Every one halts the run.

- **Embedding shuffle.** Permuting the document embeddings must collapse nDCG@10 to at most
  0.15. Catches a broken index or misaligned ids before the number is believed.
- **Self-retrieval.** A document embedded and used as its own query must retrieve itself at
  rank 1. Catches an encoder wired up wrongly in a way the shuffle would not, for example query
  and document encoders transposed.
- **Cross-process determinism.** Every arm must produce identical rankings in separate
  processes. This is a control rather than a test because the defect has already occurred here:
  BM25's outer sum iterated a set whose order depends on `PYTHONHASHSEED`, and two runs
  disagreed on 63 of 300 SciFact queries.
- **BM25 closure**, gold presence, empty query and the one-doc-per-line invariant, inherited
  unchanged.

## 10. Stopping rule, committed in advance

The entry publishes whatever comes out.

- If dense loses to full BM25 on a corpus, that is the result and it is reported as plainly as
  the reverse would be.
- If the hybrid fails to beat its better component, that is the result.
- If none of the four query properties separates the arms, the entry reports that the four
  pre-registered properties do not predict a winner, which is useful to anyone who assumed they
  would.
- If a control fails, the entry reports the failure rather than the run.

## 11. Out of scope

The knowledge-graph arm, which is 003. Tuning `k1`, `b`, the fusion weights or the encoder.
Approximate nearest neighbour indexes. Rerankers and cross-encoders. Any claim about agent
memory, production systems or vendors. Any claim that generalises beyond these three corpora and
these four query properties.

## 12. Known threats to validity

- One encoder, chosen for reproducibility. Results do not generalise to larger or
  domain-adapted encoders, and the entry must say so rather than let the reader assume.
- Truncation at 256 word pieces bites hardest on SciFact, whose documents average 225 tokens,
  and is a real limitation of the dense arm as configured rather than a property of dense
  retrieval.
- BEIR judgements are incomplete, and dense retrieval may surface relevant-but-unjudged
  documents at a different rate than lexical, which would penalise it unequally.
- The four query properties are correlated with each other and with corpus identity; the
  analysis is descriptive, not causal.
