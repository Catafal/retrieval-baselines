# 003 — add a second corpus, as a registered crossover

## Why this is legitimate, since it is the obvious objection

Adding 2WikiMultiHopQA to an experiment whose HotpotQA results are already known looks like corpus
shopping. Three things make it not that:

1. **The protocol named 2Wiki in advance.** §2 of `protocol-003` registered, before any scored run,
   that the same HippoRAG graph hurts on HotpotQA (64.7 → 60.5) and lifts 2Wiki (59.2 → 70.7), and
   that the source paper calls HotpotQA "a much weaker test for multi-hop reasoning".
2. **Precedent in this experiment.** `003-amendment-2` added the entire dense arm after the tag —
   "an addition after the tag, registered rather than slipped in". A corpus is the same move.
3. **The 2Wiki data is unseen.** The prediction is registered and tagged BEFORE anything is scored
   on it. That is a forward prediction, not a retrospective fit.

The illegitimate version is: run it, see the result, then write the prediction. That is the one
thing this plan must not do, and the ordering below is the control against it.

## Order of work — the ordering IS the method

1. Write `protocols/003-amendment-6-second-corpus.md` with the predictions. **Tag it.**
2. Only then: build the pool, extract, score.

No arm is scored on 2Wiki before the tag exists. Verifiable from git history.

## Build

- `pool2wiki.py` — download `dev.parquet`, build corpus from `context` (title → passage), qrels
  from `supporting_facts`. Freeze expected counts as a §9-style control.
  Corpus is built from the dataset itself; there is no BEIR release for 2Wiki, so doc ids are
  minted from titles and the mapping is asserted rather than assumed.
- Reuse unchanged: `extractor`, `entity_types`, `build`, `retriever`, `coverage`, `measures`,
  `analysis`, `stats`. **One variable moves: the corpus.**
- `run.py --dataset {hotpotqa,2wiki} --arm {bm25,graph,dense-minilm,dense-bge}`.

## Scope

Four arms on 2Wiki. Same seed, same B, same margin, same Holm family shape, same two class
definitions.

## Risk, stated before running

The prediction may fail. A flat per-class profile on 2Wiki falsifies the mechanism claim outright
and the entry publishes that. That is the point of registering it.

Compute: extraction over ~90k passages, PPR over 12,576 queries, two dense arms to embed. Hours.
