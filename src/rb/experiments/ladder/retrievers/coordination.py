"""
Rung 0 — 001's coordination matcher, refit behind the Retriever interface.

This is the migration's critical regression: the refit must not change 001's
behaviour at all. It is a thin wrapper, not a reimplementation — it calls
grep_baseline.materialise() and grep_baseline.run_query() exactly as rb.run.run()
does, just through retrieve(corpus, queries, top_k) instead of a pre-materialised
corpus_path and a per-query loop the caller controls. tests/test_coordination_regression.py
is the acceptance test: this retriever must reproduce results/001/scifact/summary.json's
ranked metrics exactly.
"""

import tempfile
from pathlib import Path

from rb.grep_baseline import materialise, run_query


class CoordinationRetriever:
    def __init__(self, word_bounded: bool = True):
        self.word_bounded = word_bounded
        self.name = "coordination" if word_bounded else "coordination(substring)"

    def retrieve(
        self, corpus: dict[str, str], queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        # A fresh temp file per call rather than the on-disk cache rb.run.run()
        # uses (data/<dataset>/rg_corpus.txt): that cache exists to avoid
        # re-materialising a multi-gigabyte corpus across repeated CLI
        # invocations of `rb.run`, but it makes retrieve() impure (behaviour
        # depends on what happens to be on disk). Materialising fresh keeps this
        # retriever's contract test — "deterministic across two invocations" —
        # true without relying on filesystem state between them.
        with tempfile.TemporaryDirectory() as tmp:
            corpus_path = Path(tmp) / "rg_corpus.txt"
            doc_ids = materialise(corpus, corpus_path)
            run: dict[str, dict[str, float]] = {}
            for qid in sorted(queries):
                results, _hits, _elapsed, _terms = run_query(
                    queries[qid], corpus_path, doc_ids, top_k=top_k, word_bounded=self.word_bounded
                )
                run[qid] = dict(results)
        return run
