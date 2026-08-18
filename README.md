# retrieval-baselines

The measurement harness behind the numbered experiments at
[notebook.jcatafal999.workers.dev](https://notebook.jcatafal999.workers.dev).

One repo, several arms. Each experiment swaps the retriever and keeps the corpus, the
relevance judgements and the metric code identical, because arms measured by different
code are not comparable no matter what the prose claims.

| Experiment | Retriever | Status |
|---|---|---|
| 001 | ripgrep, literal word-bounded matching | see `protocol.md` |
| 002 | dense embeddings | not started |
| 003 | knowledge graph | not started |

## Reproduce

```
make setup      # venv + dependencies
make reproduce  # downloads corpora, runs controls, scores every dataset
```

Corpora come from [BEIR](https://github.com/beir-cellar/beir). They are downloaded, never
redistributed. What this repo commits is the SHA-256 of every zip consumed
(`manifests/datasets.json`) and every number produced (`results/`).

## How to check this rather than trust it

- `protocol.md` was committed and tagged `protocol-001` **before the first scored run**.
  Diff it against `HEAD` to see whether the experiment that ran is the experiment that was
  planned.
- `results/001/<dataset>/per_query.jsonl` holds the retrieved document ids in rank order
  for every query. Every aggregate in the published entry recomputes from it without
  rerunning anything.
- `results/001/<dataset>/bm25_control.json` compares a BM25 run against the figures
  published with BEIR. If that disagrees, the harness is broken and the results are void.

## Licence

MIT for this code. The corpora carry their own licences and are not included here.
