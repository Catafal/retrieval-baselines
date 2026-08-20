"""
Experiment 003's metric set, and the guarantee that adding it moved nothing —
protocols/003-graph-arm.md section 5.

The risk this file exists for is subtle: adding measures cannot change a measured VALUE
(pytrec_eval evaluates each independently) but it can change the SHAPE of an artifact,
and 001's artifact shape is a published promise. These tests pin both.
"""

import json
from pathlib import Path

import pytest

from rb import datasets, metrics
from rb.experiments.graph.measures import (
    GRAPH_MEASURES,
    PRIMARY_MEASURE,
    SECONDARY_MEASURES,
)

ROOT = Path(__file__).resolve().parents[1]

_QRELS = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1, "d4": 1}}
_RUN = {
    "q1": {"d1": 5.0, "d9": 4.0, "d2": 3.0, "d8": 2.0, "d7": 1.0},
    "q2": {"d9": 5.0, "d8": 4.0, "d3": 3.0, "d7": 2.0, "d4": 1.0},
}


def test_the_published_measure_set_is_unchanged():
    """001 and 002 published against exactly these three. If this test ever needs
    editing, an already-published artifact is about to change shape."""
    assert metrics.MEASURES == {"ndcg_cut_10", "recall_10", "recall_100"}


def test_graph_measures_is_a_superset_that_adds_only_the_two_cutoffs():
    assert GRAPH_MEASURES > metrics.MEASURES
    assert GRAPH_MEASURES - metrics.MEASURES == {"recall_2", "recall_5"}


def test_score_ranked_defaults_to_the_published_set():
    """Every existing caller passes no measures and must keep getting exactly three."""
    scored = metrics.score_ranked(_QRELS, _RUN)
    assert set(scored["q1"]) == metrics.MEASURES


def test_adding_measures_adds_keys_without_moving_any_existing_value():
    """The whole safety argument for section 5 in one assertion."""
    base = metrics.score_ranked(_QRELS, _RUN)
    extended = metrics.score_ranked(_QRELS, _RUN, GRAPH_MEASURES)
    for qid in base:
        for measure, value in base[qid].items():
            assert extended[qid][measure] == value, f"{qid}/{measure} moved"
        assert set(extended[qid]) - set(base[qid]) == {"recall_2", "recall_5"}


def test_recall_cutoffs_actually_discriminate_on_two_gold_documents():
    """
    The reason section 5 changes the primary metric. q1 has a gold document at rank 1
    and q2 has none until rank 3, which R@2 separates and R@10 cannot see at all.
    """
    scored = metrics.score_ranked(_QRELS, _RUN, GRAPH_MEASURES)
    assert scored["q1"]["recall_2"] == 0.5
    assert scored["q2"]["recall_2"] == 0.0
    assert scored["q1"]["recall_10"] == scored["q2"]["recall_10"] == 1.0, (
        "at cutoff 10 both queries look identical — this is what R@2 exists to expose"
    )


def test_primary_and_secondary_measures_are_all_in_the_set():
    """Section 7 fixes which metric carries the prediction, so the analysis cannot
    promote whichever one happens to reach significance."""
    assert PRIMARY_MEASURE in GRAPH_MEASURES
    assert set(SECONDARY_MEASURES) <= GRAPH_MEASURES
    assert PRIMARY_MEASURE not in SECONDARY_MEASURES


@pytest.mark.parametrize("path", sorted(
    p for p in (ROOT / "results").rglob("summary.json")
))
def test_every_committed_summary_still_has_exactly_the_published_measures(path):
    """
    The regression guarantee, checked against every artifact 001 and 002 actually
    committed rather than against a representative one. If the measure set had been
    grown globally, every one of these would now be missing two keys relative to a
    re-run — no value wrong, every shape wrong.
    """
    ranked = json.loads(path.read_text())["ranked"]
    assert set(ranked) == metrics.MEASURES, f"{path.relative_to(ROOT)} changed shape"


# --- the run_rung path, which is where the measure set reaches an artifact ----------

class _Stub:
    """Minimal Retriever: ranks by descending doc id so scores are strictly decreasing."""
    name = "stub"

    def retrieve(self, corpus, queries, top_k):
        return {
            qid: {d: float(len(corpus) - i) for i, d in enumerate(sorted(corpus)[:top_k])}
            for qid in queries
        }


def _tiny():
    corpus = {f"d{i}": f"document {i}" for i in range(1, 6)}
    queries = {"q1": "anything"}
    qrels = {"q1": {"d1": 1, "d2": 1}}
    return corpus, queries, qrels


def test_run_rung_writes_the_published_measures_by_default(tmp_path):
    from rb.retriever import run_rung
    corpus, queries, qrels = _tiny()
    summary = run_rung(_Stub(), "tiny", corpus, queries, qrels, tmp_path / "a")
    assert set(summary["ranked"]) == metrics.MEASURES


def test_run_rung_honours_an_experiment_specific_measure_set(tmp_path):
    """003 asks for five; the artifact must actually carry five, not silently three."""
    from rb.retriever import run_rung
    corpus, queries, qrels = _tiny()
    summary = run_rung(_Stub(), "tiny", corpus, queries, qrels, tmp_path / "b",
                       measures=GRAPH_MEASURES)
    assert set(summary["ranked"]) == GRAPH_MEASURES
    written = json.loads((tmp_path / "b" / "summary.json").read_text())
    assert set(written["ranked"]) == GRAPH_MEASURES, "the artifact on disk, not just the return value"
