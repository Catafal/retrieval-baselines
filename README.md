# retrieval-baselines

The measurement harness behind the numbered experiments at
[notebook.jcatafal999.workers.dev](https://notebook.jcatafal999.workers.dev).

One repo, several arms. Each experiment swaps the retriever and keeps the corpus, the
relevance judgements and the metric code identical, because arms measured by different
code are not comparable no matter what the prose claims.

| Experiment | Retriever | Status |
|---|---|---|
| 001 | ripgrep, literal word-bounded matching | see `protocols/001-grep-baseline.md` |
| 002 | lexical factorial, dense, hybrid (the ladder) | harness implemented, see `protocols/002-ladder.md`; not yet scored |
| 003 | knowledge graph | not started |

## Shared instrument, per-experiment arms

One corpus loader, one metric implementation, one control library (`src/rb/datasets.py`,
`metrics.py`, `controls.py`, `stats.py`) is shared by every experiment and never forked. Only
the retriever under test — and that experiment's protocol and results — is per-experiment. If
arms were measured by copies of the same code that had drifted apart, a difference between them
could be an artifact of the drift rather than of the retriever, and no reader could tell which.
Sharing the instrument is what keeps a cross-experiment comparison (e.g. "how much of 001's gap
to BM25 does 002 explain") meaningful instead of merely adjacent.

## Reproduce

```
make setup                 # venv + dependencies
make reproduce              # Experiment 001: downloads corpora, runs controls, scores every dataset
make reproduce-002-lexical  # Experiment 002: coordination + the eight-cell lexical factorial
```

**`make reproduce` means Experiment 001, exactly, and always will.** It is a published
instruction — the entry at notebook.jcatafal999.workers.dev tells a reader to run it — so its
recipe does not change and does not grow to cover new experiments. Every later experiment gets
its own target instead (`reproduce-002-lexical` above), so checking 001 never silently becomes
a longer job than the one that was promised.

`reproduce-002-lexical` only runs the cheap rungs (coordination and the lexical factorial,
minutes per dataset). Experiment 002's dense and hybrid rungs are **not** wired into any
Makefile target: they need a real pinned embedding model, and running them is a separate,
explicitly gated step described in `protocols/002-ladder.md`, not something a stranger should
trigger by accident while trying to reproduce the cheap part. As of this commit neither rung has
been run — there is no `results/002/*/dense` or `.../hybrid`, and none of this repo's docs
should be read as claiming otherwise.

Corpora come from [BEIR](https://github.com/beir-cellar/beir). They are downloaded, never
redistributed. What this repo commits is the SHA-256 of every zip consumed
(`manifests/datasets.json`) and every number produced (`results/`).

## The `Retriever` seam

Every experiment from 002 onward plugs into one interface (`src/rb/retriever.py`):

```python
class Retriever(Protocol):
    name: str
    def retrieve(self, corpus: dict[str, str], queries: dict[str, str], top_k: int) -> dict[str, dict[str, float]]:
        ...
```

`rb.retriever.run_rung()` is written once against this interface — scoring, the per-query
artifact, the environment manifest, the controls — and never specialised per rung. Adding a new
rung means adding a new class that implements `retrieve()`; `run_rung()` does not change. This
is deliberately the highest seam available: the point where an experiment's only genuine
variable, the retrieval function, enters the system.

A new rung's `retrieve()` must, on any corpus and query set:

- return at most `top_k` documents per query,
- return only document ids present in the corpus,
- return **strictly decreasing** scores per query — ties must be broken deterministically (this
  repo's convention is by document id), because a tied score handed to `trec_eval` gets broken
  by `trec_eval`'s own internal rule instead of the pre-registered one,
- be deterministic across two invocations on the same input.

`tests/helpers.py::assert_retriever_contract` checks all four properties against a tiny
in-memory corpus and is run against every rung in `tests/test_retriever_contract.py`. Run it
against a new rung before anything else.

## How to check this rather than trust it

- `protocols/001-grep-baseline.md` was committed and tagged `protocol-001` **before the first
  scored run**. Diff it against `HEAD` to see whether the experiment that ran is the experiment
  that was planned. (`protocol.md` at the old path still resolves, as a stub pointing here.)
  `protocols/002-ladder.md` is the same document for experiment 002 — as of this commit it is
  written and committed but **not yet tagged**, because 002 has not been scored yet.
- `results/001/<dataset>/per_query.jsonl` holds the retrieved document ids in rank order
  for every query. Every aggregate in the published entry recomputes from it without
  rerunning anything. Experiment 002 writes the same shape of artifact per rung
  (`results/002/<dataset>/<rung>/per_query.jsonl`) via the same `run_rung()` code path.
- `results/001/<dataset>/bm25_control.json` compares a BM25 run against the figures
  published with BEIR. If that disagrees, the harness is broken and the results are void.
  Experiment 002's lexical factorial runs an analogous closure control
  (`rb.controls.bm25_closure`) before writing anything: the factorial's all-on corner is full
  BM25 by construction, so it is checked against the same published figure before any Shapley
  attribution is trusted.

## Licence

MIT for this code. The corpora carry their own licences and are not included here.
