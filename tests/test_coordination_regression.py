"""
The migration's acceptance test.

001's coordination matcher was refit behind the Retriever interface
(rb.experiments.ladder.retrievers.coordination.CoordinationRetriever) without
touching grep_baseline's tokenize/search/rank functions at all — it only changes
how the corpus is handed in and how results are collected. This test proves that
refit did not change behaviour: run the new retriever on the real, already-downloaded
SciFact corpus and confirm its ranked metrics match results/001/scifact/summary.json
exactly.

Runs on real data (SciFact, ~20s) rather than a fixture, because the property under
test — "identical output to the committed 001 artifact" — is only meaningful against
the corpus 001 was actually measured on.
"""

import json
from pathlib import Path

import pytest

from rb import datasets, metrics
from rb.experiments.ladder.retrievers.coordination import CoordinationRetriever
from rb.run import select_queries

ROOT = Path(__file__).resolve().parents[1]
SCIFACT_SUMMARY = ROOT / "results" / "001" / "scifact" / "summary.json"

pytestmark = pytest.mark.skipif(
    not (Path(datasets.DATA_DIR) / "scifact").exists() or not SCIFACT_SUMMARY.exists(),
    reason="requires the downloaded SciFact corpus and committed 001 results",
)


def test_coordination_retriever_reproduces_001_scifact_summary():
    committed = json.loads(SCIFACT_SUMMARY.read_text())
    assert committed["word_bounded"] is True, "regression target assumes 001's default word-bounded run"

    corpus, queries, qrels = datasets.load("scifact")
    qids, sampled = select_queries(queries)
    assert sampled is False, "SciFact has fewer than 500 judged queries, same as 001's run"
    subsampled_queries = {q: queries[q] for q in qids}

    run_dict = CoordinationRetriever(word_bounded=True).retrieve(corpus, subsampled_queries, top_k=100)
    per_query = metrics.score_ranked({q: qrels[q] for q in qids}, run_dict)

    ranked = {
        m: round(metrics.mean([per_query[q][m] for q in qids]), 4) for m in sorted(metrics.MEASURES)
    }
    assert ranked == committed["ranked"], (
        f"refit coordination retriever diverged from committed 001 numbers: {ranked} != {committed['ranked']}"
    )
