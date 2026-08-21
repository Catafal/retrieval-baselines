# Amendment 2 to protocol-003 — the dense arm

**Status: frozen on tagging as `protocol-003-amendment-2`. Written and committed before a single
vector is computed.** `protocol-003` pins no encoder and never registers dense as an arm: §3 says
"every arm is scored against this identical candidate set" and names none. Running dense on the
pool is therefore an addition, and it is registered here rather than slipped in.

## 1. Why it is being added

The sequence's public promise is grep versus vectors versus graphs, compared the same way on the
same corpus. 003 has BM25 and the graph arm on the 66,581-passage pool. Without dense, the
instalment that is supposed to complete the three-way comparison delivers two arms.

## 2. What it may claim, and what it may not

**May:** report dense's Recall@2/@5 and nDCG@10 on the pool beside BM25 and the graph arm, and
describe how the three families rank on this corpus.

**May NOT** enter the registered analysis. §7's clauses A and B are already resolved, and their
p-values are Holm-corrected across a family declared before the run: {A, B} × {R@2, R@5, nDCG@10}
× {primary, sensitivity}. Adding members to that family now would change the correction applied to
claims that have already resolved, which is retrospective re-scoring of a settled result. Dense is
reported outside the family, and any per-class figure computed for it is labelled exploratory.

**May NOT** be used to revise the graph arm's conclusion in either direction.

## 3. Both encoders, pinned

002's own finding is the reason this is two runs and not one: swapping MiniLM for BGE **flipped**
SciFact's corpus-level verdict from "no difference" to "dense wins by 0.080", while the per-query
gradient survived. An entry whose lesson is that corpus-level verdicts are encoder-dependent
should not then report one encoder and call it "the dense arm".

| | model | revision | dims | note |
|---|---|---|---|---|
| dense-minilm | `sentence-transformers/all-MiniLM-L6-v2` | as pinned in 002 amendment 1 | 384 | the encoder 002's HotpotQA number was measured with |
| dense-bge | `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab7` | 768 | 002 amendment 2; query prefix applied to queries only |

Both are affordable here precisely because the pool is 66,581 passages rather than 5,233,329 —
which is why 002 amendment 2 excluded HotpotQA from the BGE run in the first place.

Exact cosine, no approximate index, same as 002. All inherited controls apply, including the
embedding-shuffle control and self-retrieval.

## 4. Predicted before running

Registered so this is a test rather than a description:

1. **Both dense encoders lose to full BM25 on this pool.** 002 measured dense losing to BM25 on
   HotpotQA by 0.130 nDCG@10 at full-corpus scale, and the paper this experiment builds on calls
   HotpotQA lexically easy. Restricting to distractors does not change the query side.
2. **Both dense encoders beat the graph arm**, whose Recall@2 is 0.213 against BM25's 0.549.
3. **BGE scores above MiniLM** on this corpus, as it did on SciFact and Quora.

A failure of 3 without 1 or 2 means the encoder was not actually stronger here, which invalidates
the comparison rather than the argument, and the entry will say so — the same wording 002
amendment 2 used, because the same failure mode applies.

## 5. Stopping rule

Published whatever comes out. If dense beats BM25 on this pool, that contradicts prediction 1 and
is the most interesting outcome available, because it would mean the distractor restriction
changes the lexical/dense verdict that 002 measured at full-corpus scale — and that would be a
finding about the corpus construction rather than about either arm.
