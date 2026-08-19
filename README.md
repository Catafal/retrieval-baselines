# retrieval-baselines

The measurement harness behind the numbered experiments at
[notebook.jcatafal999.workers.dev](https://notebook.jcatafal999.workers.dev).

One repo, several arms. Each experiment swaps the retriever and keeps the corpus, the
relevance judgements and the metric code identical, because arms measured by different
code are not comparable no matter what the prose claims.

| Experiment | Retriever | Status |
|---|---|---|
| 001 | ripgrep, literal word-bounded matching | see `protocols/001-grep-baseline.md` |
| 002 | lexical factorial, dense, hybrid (the ladder) | lexical factorial measured on all three datasets; dense and hybrid measured on SciFact only, see `protocols/002-ladder.md` and `protocols/002-amendment-1-dense.md` |
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
Makefile target: they need a real pinned embedding model (now pinned per
`protocols/002-amendment-1-dense.md`), and running them (`python -m rb.experiments.ladder.run
--dataset <name> --rung dense`, `--rung hybrid`) is a separate, explicitly invoked step, not
something a stranger should trigger by accident while trying to reproduce the cheap part. As of
this commit both rungs have been measured on SciFact only — `results/002/scifact/dense` and
`.../hybrid` — per the amendment's stopping point; Quora and HotpotQA are unattended runs the
amendment gates separately and neither has been launched.

The lexical rungs (`LexicalRetriever`) score against a sparse inverted index
(`lexical.build_index`) rather than scanning the corpus per query — the naive scan does not
finish on HotpotQA (projected 18.6 hours and 18.3 GB, on a 26 GB machine) because it rebuilds
its term-frequency bookkeeping from scratch, once per query, once per one of the eight
factorial configs. The index is built once per dataset and shared read-only across all eight
configs. Measured on HotpotQA (5.2M documents): index build ~93–100s, ~51 minutes wall clock for
all eight configs including that build, ~13.3 GB peak memory (the transient cost of the Python
lists that stage postings before `scipy.sparse.csc_matrix` compacts them; the built matrix
itself is ~2.1 GB).

The lexical factorial has since been run for real, on all three datasets:
`results/002/<dataset>/lexical_factorial.json`, one file per dataset, each
`<dataset>/lexical(...)/` holding that cell's own `per_query.jsonl` and `summary.json`. Dense
and hybrid have not — see the paragraph above.

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

**Determinism is verified across processes, not only within one.** The fourth bullet above
looks like a property a single test run could confirm and it is not: two rungs can return
identical rankings every time inside one Python process and still disagree between two
separate `python` invocations, if anything in the retrieval path depends on iteration order
that Python does not guarantee stable across processes. That happened here — see "The
cross-process nondeterminism" below — and the fix is `tests/test_lexical.py`'s
`test_retrieval_is_identical_across_processes` and `tests/test_stats.py`'s
`test_shapley_bootstrap_deterministic_across_processes`, which both run the code under test in
a fresh subprocess (2-3 times) and assert byte-identical output, rather than asserting against
a second call in the same process the way a same-process test would.

## The cross-process nondeterminism

`LexicalRetriever`'s BM25 sum previously iterated `set(_tokenize(query))` — the distinct query
terms — in whatever order Python's set gave it. Set iteration order depends on `PYTHONHASHSEED`,
which is randomised once per process by default, and float addition is not associative, so
summing the same per-term contributions in a different order can produce totals that differ in
the last bits. That is normally invisible, until two documents' scores are close enough for the
difference to flip which one sorts first. Measured directly: two separate processes scoring the
same 300 SciFact queries disagreed on the ranking for 63 of them. Query terms are now sorted
before scoring (`sorted(set(...))`), in both the fast and reference code paths, which removes
the dependency on hash order entirely. This is why the contract's "deterministic across two
invocations" bullet above is now checked with a subprocess test rather than a same-process one:
a same-process test cannot see this class of bug, because it never leaves the process whose
hash seed made the bug invisible.

## Shapley attribution: intervals, ordering, ties

`lexical_factorial.json`'s `shapley_ndcg_cut_10` is a point estimate per mechanism (idf,
tf_saturation, length_norm) computed once, from one query subsample. On its own it invites
exactly the misread this repo made once already: SciFact's `idf` (0.2354) and `tf_saturation`
(0.2427) differ by 0.0073, and reading that gap as "tf_saturation matters more" treats a single
sample's noise as a finding. Three fields sit next to it to stop that:

- **`shapley_ci95`** — a 95% bootstrap interval per mechanism (`rb.stats.shapley_bootstrap`,
  2000 rounds, seed `20260818`), from resampling queries, not cells: the eight factorial cells
  are a complete enumeration of the three switches, not a sample, so there is nothing to
  resample there, but the query subsample each cell's mean was computed over is exactly the
  kind of sample a bootstrap interval is for. Every one of the eight cells is recomputed from
  the SAME drawn queries within a round before Shapley runs on that round — see
  `shapley_bootstrap`'s docstring in `src/rb/stats.py` for why that pairing, not the interval
  itself, is the property most likely to be gotten wrong by a future edit.
- **`shapley_pairwise_ordering`** — for every pair of mechanisms, the fraction of bootstrap
  rounds where one outranked the other, e.g. `"idf>tf_saturation": 0.336`. **A fraction near 0.5
  means the ordering is unresolved, flipping depending on which queries happened to be drawn,
  not that the loser "narrowly" lost.** SciFact's `idf`/`tf_saturation` pair above is exactly
  this case: 0.336 is not 1.0 or 0.0, so the 0.0073 point-estimate gap is not a reliable
  ordering. Compare `idf>length_norm` and `tf_saturation>length_norm` on the same dataset,
  both 1.0: that ordering does not flip under resampling and can be reported as one.
- **`shapley_pairwise_ties`** — the fraction of rounds where the pair came out exactly equal.
  Reported separately because the ordering fraction alone cannot distinguish "b always wins" from
  "a and b always tie": both give `a>b: 0.0`, and a reader scanning only the ordering table would
  wrongly read the second case as b dominating.

## How to check this rather than trust it

- `protocols/001-grep-baseline.md` was committed and tagged `protocol-001` **before the first
  scored run**. Diff it against `HEAD` to see whether the experiment that ran is the experiment
  that was planned. (`protocol.md` at the old path still resolves, as a stub pointing here.)
  `protocols/002-ladder.md` is the same document for experiment 002, tagged `protocol-002`.
  Diff it against `HEAD` the same way; as of this commit the only change since the tag is a
  correction to that document's own stale "not yet tagged" status line, recorded in the
  document as an amendment. Everything the tag actually pre-registered — including the Shapley
  values section the intervals and fixes below add to — is unchanged.
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
