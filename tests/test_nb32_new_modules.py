"""
Coverage for the modules added late in 003, where a pre-publication review seat mutated the source
and the suite stayed green every time.

It zeroed every figure in `arms_summary`, broke `pool2wiki`'s corpus-defining filter, and swapped
graph for BM25 in `per_class_profile` -- 375/375 passed on each. Those last are the exact per-class
numbers the entry cites. A test suite that survives its own headline figures being zeroed is not
evidence of anything.
"""

import json

import pytest

from rb.experiments.graph import analysis, arms_summary, pool2wiki


# ---------------------------------------------------------------- arms_summary

def _fake_arm(tmp_path, name, r2):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "ranked": {"recall_2": r2, "recall_5": r2 + 0.1, "ndcg_cut_10": r2 + 0.2,
                   "recall_10": r2 + 0.3, "recall_100": r2 + 0.4},
        "queries_scored": 10}))
    return d


def test_arm_row_reads_the_published_figures(tmp_path):
    """KILLS: zeroing or hardcoding the figures in `arm_row`.

    This is the table the entry's headline comparison is built from. The reviewer replaced every
    value with 0.0 and nothing failed.
    """
    row = arms_summary.arm_row(_fake_arm(tmp_path, "bm25", 0.5490))
    assert row["recall_2"] == 0.5490
    assert row["ndcg_cut_10"] == pytest.approx(0.7490)
    assert row["queries_scored"] == 10


def test_arm_row_returns_none_for_an_arm_that_was_not_run(tmp_path):
    """An arm missing on one corpus must be absent, not zero. Zero is a measurement."""
    assert arms_summary.arm_row(tmp_path / "never-run") is None


def test_the_defective_run_is_never_a_fifth_arm():
    """KILLS: folding `graph-defective` into the comparison table.

    Its R@2 of 0.2132 is RETRACTED (amendment 3). It is kept on disk so a reader can see what was
    withdrawn, and it must appear only under `superseded_runs`.
    """
    built = arms_summary.build()
    for corpus in built["corpora"].values():
        assert "graph-defective" not in corpus["arms"]
    assert "graph-defective" in built["superseded_runs"]


# ---------------------------------------------------------------- pool2wiki

def test_only_two_gold_questions_are_scored():
    """KILLS: changing GOLD_PER_QUESTION, which silently redefines the corpus.

    2Wiki's 2,751 four-gold questions cap R@2 at 0.50, so including them would make the primary
    measure mean something different than on HotpotQA and break the cross-corpus comparison. The
    reviewer set this to 999 and the suite stayed green.
    """
    assert pool2wiki.GOLD_PER_QUESTION == 2
    rows = pool2wiki.load_rows()
    assert len(rows) == pool2wiki.EXPECTED_QUESTIONS
    assert all(len(r["gold"]) == 2 for r in rows)


def test_the_frozen_counts_gate_the_run():
    """The §9-style control must fail on a corpus that does not match what was registered."""
    bad = pool2wiki.control({"questions": 1, "passages": 1, "title_slots": 1,
                             "variant_titles": 0, "gold_titles": 1,
                             "gold_titles_matched": 0, "gold_queries": 1})
    assert bad["passed"] is False
    assert "questions" in bad["mismatched"]


# ---------------------------------------------------------------- per-class profile

def _profile_fixture():
    graph = {f"a{i}": {"recall_2": 0.20} for i in range(4)}
    graph.update({f"c{i}": {"recall_2": 0.45} for i in range(4)})
    bm25 = {f"a{i}": {"recall_2": 0.53} for i in range(4)}
    bm25.update({f"c{i}": {"recall_2": 0.66} for i in range(4)})
    classes = {f"a{i}": 1 for i in range(4)}
    classes.update({f"c{i}": 2 for i in range(4)})
    return graph, bm25, sorted(graph), classes


def test_per_class_profile_does_not_swap_the_arms():
    """KILLS: swapping graph_recall_2 and bm25_recall_2.

    These are the 0.1952 / 0.2272 / 0.2100 figures the entry quotes. The reviewer swapped them and
    all 14 relevant tests passed.
    """
    graph, bm25, shared, classes = _profile_fixture()
    out = analysis.per_class_profile(graph, bm25, shared, classes, empty_qids=set())
    assert out["coverage_1"]["graph_recall_2"] == 0.20
    assert out["coverage_1"]["bm25_recall_2"] == 0.53
    assert out["coverage_2"]["graph_recall_2"] == 0.45


def test_per_class_profile_counts_empty_retrievals():
    """KILLS: gutting `empty_query_ids` to return an empty set.

    The empty rate is the mechanism the entry's central claim rests on. The reviewer made it always
    return nothing and no test noticed.
    """
    graph, bm25, shared, classes = _profile_fixture()
    out = analysis.per_class_profile(graph, bm25, shared, classes,
                                     empty_qids={"a0", "a1", "c0"})
    assert out["coverage_1"]["graph_retrieved_nothing"] == 2
    assert out["coverage_1"]["graph_empty_rate"] == 0.5
    assert out["coverage_2"]["graph_empty_rate"] == 0.25


def test_empty_query_ids_finds_the_queries_that_retrieved_nothing():
    """The real reader, against the committed artifact."""
    ids = analysis.empty_query_ids("graph")
    assert len(ids) == 1309, "the published HotpotQA no-seed count"
