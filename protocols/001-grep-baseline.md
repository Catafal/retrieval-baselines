# Pre-registration — Experiment 001, grep baseline

**Status: frozen.** Written and committed before the first scored run. Tagged
`protocol-001`. Any change after that tag is an amendment, made in its own commit with a
stated reason, so the diff between what was planned and what shipped is public.

This document exists because the entry it precedes replaced a retracted one. The retracted
entry asked to be trusted. This one does not.

---

## 1. Question

> On a public labelled retrieval benchmark, what fraction of the gold documents does
> literal substring search retrieve, and what does it cost?

Not "is grep good". Not "do you need a vector database". Both are questions this experiment
cannot answer and the published entry must refuse.

## 2. Datasets

Three BEIR datasets, publicly redistributable subset, fixed before any run. Corpora are
downloaded and never redistributed; `manifests/datasets.json` records the SHA-256 of every
zip consumed.

| Dataset | Query regime | Expectation |
|---|---|---|
| SciFact | term-dense scientific claims | grep's best case |
| Quora | duplicate-question paraphrase | grep's worst case, vocabulary mismatch |
| HotpotQA | multi-hop, 2 gold documents per query | needs two documents, not one |

Adding, removing or swapping a dataset after seeing results is a protocol amendment.

## 3. System under test — grep-baseline-v1

1. **Query to terms.** Lowercase, split on non-alphanumerics, drop the frozen 33-word
   stopword list in `src/rb/stopwords.py`, deduplicate preserving order. No stemming, no
   expansion, no rewriting.
2. **Matching.** `rg -i -w -F` — case-insensitive, word-bounded, literal. Word-bounded
   rather than raw substring because unbounded matching makes *insulin* match *insulinoma*,
   inflating the candidate set without adding signal. The unbounded variant is run as a
   sensitivity check and reported as such.
3. **Ranking.** Grep returns an unordered set; metrics require an order. The rule is
   coordination-level matching: count of distinct query terms present, tie-broken by total
   match count, then by document id. Deterministic. Deliberately the dumbest set-to-order
   rule available. **It is not BM25 and no number from it may be presented as if it were.**

## 4. Metrics

**Ranked**, comparable to published work: nDCG@10, Recall@10, Recall@100, via `pytrec_eval`.

**Set**, the honest grep-alone figure: recall over the entire unranked output, median and
mean output size, and the count of queries returning nothing. Grep hands back a pile; the
size of the pile is the cost, and it is the number nobody publishes.

**Cost**: mean wall-clock seconds per query, total seconds, and spend in USD.

## 5. Controls

Every one fails loudly and halts the run.

- **Gold-presence.** Every judged document must be in the indexed corpus. A missing gold
  document depresses recall for reasons unrelated to the retriever and would read as a finding.
- **Empty query.** A query with no surviving terms retrieves nothing and scores zero. Catches
  a scorer that credits documents it never retrieved.
- **One-doc-per-line invariant.** Corpus line count must equal document count, or every
  line-number-to-document-id lookup after the first multi-line document is silently wrong.
- **BM25 anchor.** External calibration against BM25 nDCG@10 published in Thakur et al. 2021,
  Table 2: SciFact 0.665, Quora 0.789, HotpotQA 0.603. Tolerance 0.10 absolute — loose on
  purpose, since our implementation and tokenisation differ from Anserini and claiming exact
  agreement would be dishonest. This is an instrument check reported in methods. It is not
  the vector comparison and it is not a headline.

## 6. Subsampling

Queries with no relevance judgement are excluded before anything is scored.

Where a dataset has more than 500 judged queries, a random 500 are drawn with seed
`20260818`, from the sorted query-id list so dictionary order cannot affect the draw.

**Queries may be subsampled. The corpus may not.** Shrinking the haystack raises recall and
would manufacture a flattering result.

## 7. What gets published

The entry publishes whatever comes out. Committed here before any number exists:

- If grep scores well, the entry says so, and says why that is uncomfortable for the field.
- If grep scores badly, the entry says so, and does not soften it into a setup for entry 002.
- If a control fails, the entry reports the failure rather than the run.

`status: measured` requires the result artifact committed in this repository and reproducible
by `make reproduce`. Absent the artifact, the entry does not get the label whatever the prose says.

## 8. Out of scope

Nothing about vector databases. Nothing about production memory systems. Nothing about agents.
Nothing about whether anyone should use grep. One number, one corpus, one method.

## 9. Known threats to validity

- Grep is not a ranker; the coordination-level order is an imposed construct and a different
  imposed order would give different ranked numbers. The set metrics are the ones free of this.
- BEIR relevance judgements are incomplete: an unjudged retrieved document counts as wrong even
  when it answers the question. This penalises every method, but not necessarily equally.
- Word-boundary matching is a choice, reported alongside its unbounded sensitivity check.
- Three datasets are not the field. The regimes were chosen to separate methods, not to represent
  all retrieval.
