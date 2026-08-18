# Pre-registration — Experiment 002, the retrieval ladder

**Status: not yet tagged.** This document is written and committed before the first scored
run, as 001's was. It becomes frozen when tagged `protocol-002`; any change after that tag is
an amendment, made in its own commit with a stated reason. Tagging is a separate, human-run
step and has not happened as of this commit.

---

## 1. Question

> What does each step of retrieval sophistication, from counting matched terms up to a
> hybrid of BM25 and dense retrieval, actually buy in nDCG@10 — and at what cost — and is the
> gap between the cheap steps and the expensive ones real or noise?

001 showed an unweighted coordination matcher reaches roughly two thirds to three quarters of
BM25's nDCG@10 and refused to say which of BM25's three mechanisms bought the difference,
because no ablation was run. This entry runs that ablation and extends it up through dense and
hybrid retrieval.

## 2. Datasets, query subsample, seed

Same three BEIR datasets as 001 — SciFact, Quora, HotpotQA — same checksums
(`manifests/datasets.json`), same query subsample and same seed (`20260818`,
`rb.run.select_queries`, reused directly rather than reimplemented) so per-query pairing
across the two entries is valid. The corpus is never subsampled, for the reason 001 gives:
shrinking the haystack inflates recall.

**Corpus bounding differs from 001 for the dense and hybrid rungs only:**

| Dataset | Lexical rungs | Dense + hybrid |
|---|---|---|
| SciFact | yes | yes |
| Quora | yes | yes |
| HotpotQA | yes | **no — see below** |

HotpotQA is lexical-only. Embedding 5.2M documents exceeds the compute budget for this entry.
That is a stated budget limit, not a quiet omission — no HotpotQA dense or hybrid number will
appear, and its absence should read as this line, not as an oversight.

## 3. The seam

One interface, `rb.retriever.Retriever`:

```python
class Retriever(Protocol):
    name: str
    def retrieve(corpus: dict[str, str], queries: dict[str, str], top_k: int) -> dict[str, dict[str, float]]: ...
```

Every rung below is one implementation of this interface. The runner (`rb.retriever.run_rung`),
the scorer (`rb.metrics`), the controls (`rb.controls`) and the bootstrap/Holm/Shapley code
(`rb.stats`) are written once against it and are not specialised per rung.

## 4. Rungs

**Rung 0 — coordination (`rb.experiments.ladder.retrievers.coordination.CoordinationRetriever`).**
001's matcher, refit behind the interface, behaviour unchanged. Its regression test is that it
reproduces `results/001/scifact/summary.json`'s ranked metrics exactly.

**Rungs 1–8 — the lexical factorial (`rb.experiments.ladder.retrievers.lexical.LexicalRetriever`).**
One scoring function, three independent boolean switches — idf, tf_saturation (`k1`),
length_norm (`b`) — run as the full eight-cell factorial, not a single ladder ordering:
a ladder hands shared credit for interacting mechanisms to whichever one went first.

- `k1 = 1.2`, `b = 0.75`, fixed. Never tuned — tuning against this evaluation set would be
  fitting on the test data.
- All-off corner: raw term-frequency sum.
- All-on corner: full BM25 (`lexical.full_bm25()`), the target of the closure control below.

**Rung 9 — dense (`rb.experiments.ladder.retrievers.dense.DenseRetriever`).**

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision pinned to the commit hash
  recorded in the run's `retriever_manifest.revision` field at scoring time. (Not yet run, so
  no revision hash exists to commit here — the pinning happens in the manifest of the actual
  run, and this document commits to *that being pinned*, not to a specific hash chosen without
  having run anything.)
- Exact cosine over full-precision (float32) vectors. No approximate index.
- Batch size 32, mean pooling, max sequence length 512 (documents beyond this are truncated;
  truncation is a property of the measurement and is recorded, not hidden).
- Query and document encoding use whatever asymmetric convention the pinned model specifies;
  the convention used is recorded in the manifest.

**Rung 10 — hybrid (`rb.experiments.ladder.retrievers.hybrid.HybridRetriever`).**
Reciprocal rank fusion of the full-BM25 lexical rung and the dense rung. `k = 60`, fixed here,
before any hybrid number exists. No weight search.

## 5. Statistics

- Per-rung nDCG@10, Recall@10, Recall@100, as in 001, via the same `pytrec_eval` code path
  (`rb.metrics`).
- **Paired bootstrap** (`rb.stats.paired_bootstrap`) on the per-query difference between
  adjacent rungs: 2000 resamples, seed `20260818`. Reports the mean difference, a 95%
  percentile interval, and a two-sided bootstrap p-value. This is the difference the reader
  gets an interval on — 001 only reported per-arm intervals, which cannot say whether an
  adjacent gap is real.
- **Holm correction** (`rb.stats.holm_correction`) across the adjacent-rung comparisons made
  within each corpus, family-wise alpha 0.05.
- **Shapley values** (`rb.stats.shapley_values`, via `lexical.shapley_from_ndcg`) per lexical
  mechanism, computed from the eight-cell factorial's measured nDCG@10 values. This is the
  order-independent attribution; the ladder ordering is reported alongside it for narrative
  only, never as the attribution itself.
- Effect sizes reported in nDCG points, not as a ratio of one metric to another, per 001's own
  flag that an unintervalled ratio is misleading.

## 6. Cost accounting

Wall clock per rung (`rb.retriever.run_rung`'s `cost.total_seconds`), plus for the dense rung:
embedding wall clock and dollar cost, index build time (there is no index — exact search — so
this is reported as zero with a note why), and peak memory during embedding.

## 7. Controls

Every one fails loudly and halts the run, same convention as 001.

- **Gold-presence** and **empty-query** (`rb.controls`), inherited unchanged, run for every
  rung via `run_rung`.
- **BM25 closure** (`rb.controls.bm25_closure`). The lexical all-on corner
  (`lexical.full_bm25()`) must agree with the **published** BM25 figure for the dataset
  (Thakur et al. 2021, Table 2, carried in `results/001/<dataset>/bm25_control.json` as
  `published_bm25_ndcg_cut_10`) within **0.10 absolute** nDCG@10, the same tolerance and the
  same external reference 001 used. It runs before any factorial artifact is written, and it
  raises. This is the single most important control in this experiment: if it fails, nothing
  from 002 ships until the lexical implementation is fixed.

  This is not what an earlier draft of this protocol said, and the correction is recorded
  rather than quietly applied. That draft gated against 001's in-repo `bm25s` anchor at 0.02,
  reasoning that two numbers from the same repository should differ only by rounding. Measured
  on SciFact before tagging, that reasoning was wrong:

  | | nDCG@10 | tokenisation |
  |---|---|---|
  | this repo's full BM25 | 0.6605 | no stopword removal, no stemming |
  | 001's `bm25s` anchor | 0.6863 | stopwords + Snowball stemming |
  | published (Anserini) | 0.6650 | — |

  The two in-repo numbers differ by 0.0258 because they tokenise differently, which is a real
  difference in what is measured rather than noise, and the 0.02 gate would therefore have
  failed a correct implementation. Ours sits 0.0045 from the published figure, closer to it
  than `bm25s` is. So the gate is against the external reference, and the in-repo anchor is
  reported alongside it as information rather than used to pass or fail the run.
- **Embedding shuffle** (`rb.controls.embedding_shuffle`). Permuting the document embedding
  matrix (`DenseRetriever(..., shuffle_seed=<seed>)`) before scoring must collapse nDCG@10 to
  at most **0.15** (the stated chance ceiling). If it does not, the vector index or its id
  bookkeeping is broken and the dense number is void.

## 8. What gets published

Committed here before any number exists, same stopping rule as 001:

- If the three lexical mechanisms account for most of the climb to full BM25, the entry says
  so and names it as uncomfortable for the default assumption that embeddings are needed.
- If the dense and hybrid rungs dominate, the entry says so and quantifies the premium they
  command over the lexical ceiling.
- If a control fails, the entry reports the failure rather than the run.
- `status: measured` requires the result artifacts committed in this repository and
  reproducible by the relevant Makefile target. Absent the artifact, the entry does not get
  the label whatever the prose says.

## 9. Out of scope

Parameter tuning of `k1`, `b`, or the fusion weight. Approximate nearest-neighbour indexes and
their speed/recall tradeoff. Reranking, cross-encoders, query expansion or rewriting. Any
claim about which retrieval method anyone should adopt, or about production memory systems,
agent frameworks, or specific vendors.

## 10. Known threats to validity

- BEIR relevance judgements are incomplete, as noted in 001; this penalises every rung, not
  necessarily equally, and may penalise dense retrieval differently than lexical if dense
  retrieves semantically relevant but unjudged documents more often.
- The lexical ablation's behaviour when `tf_saturation` is off and `length_norm` is on is a
  modelling choice this repository makes (documented in the code and in the accompanying
  implementation notes), not a value handed down by a single canonical "BM25 minus one
  mechanism" definition — no such single definition exists in the literature.
- HotpotQA's missing dense/hybrid cells mean the ladder's top two rungs are compared on two
  corpora, not three, which is a narrower base than the lexical factorial's three.
