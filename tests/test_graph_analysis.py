"""
003's registered analysis and its headroom control (NB-25 D6, sixth defect).

Named test_graph_analysis to avoid colliding with test_analysis.py, which covers the
unrelated rb.experiments.ladder.analysis.

MUTATION-CHECKED. NB-24 shipped three tests that passed whether or not the defect was present:
one raised the error it asserted, one used a topology where the term under test normalised
away, one asserted a sum the code's own guard restored even when the invariant was violated.
A test that passes is not evidence; only a test that FAILS against the defect is. Each test
below names the mutation it kills. See NB-25 for the audit these close.
"""

import pytest

from rb.experiments.graph import analysis as an


# ---------------------------------------------- D6: the headroom control's missing producer

def test_headroom_normalises_by_remaining_room():
    """KILLS: reporting the raw deficit as if it were the normalised one.

    The whole point of the control is that a class where BM25 already scores well has less
    room left, so an identical raw deficit means something different there. Two classes with
    the SAME raw deficit and DIFFERENT baselines must get different normalised figures.
    """
    graph = {"a": {"recall_2": 0.4}, "b": {"recall_2": 0.7}}
    bm25 = {"a": {"recall_2": 0.5}, "b": {"recall_2": 0.8}}
    out = an.headroom(graph, bm25, ["a", "b"], {"a": 1, "b": 2})
    absent, comparison = out["per_class"]["bridge_absent"], out["per_class"]["comparison"]
    assert absent["raw_deficit"] == pytest.approx(comparison["raw_deficit"])
    # Same raw deficit, but 0.5 headroom vs 0.2 — the normalised figures must diverge.
    assert absent["headroom_normalised_deficit"] != comparison["headroom_normalised_deficit"]
    assert absent["headroom_normalised_deficit"] == pytest.approx(-0.2)
    assert comparison["headroom_normalised_deficit"] == pytest.approx(-0.5)


def test_headroom_splits_classes_on_coverage_two():
    """KILLS: an off-by-one in the class boundary.

    Coverage 0 and 1 are bridge-absent; only coverage 2 is the comparison control.
    """
    graph = {q: {"recall_2": 0.5} for q in "abc"}
    bm25 = {q: {"recall_2": 0.5} for q in "abc"}
    out = an.headroom(graph, bm25, list("abc"), {"a": 0, "b": 1, "c": 2})
    assert out["per_class"]["bridge_absent"]["n"] == 2
    assert out["per_class"]["comparison"]["n"] == 1


def test_headroom_differential_is_the_gap_between_the_two_classes():
    """KILLS: subtracting in the wrong direction, which would flip the reported sign."""
    graph = {"a": {"recall_2": 0.4}, "b": {"recall_2": 0.1}}
    bm25 = {"a": {"recall_2": 0.5}, "b": {"recall_2": 0.5}}
    out = an.headroom(graph, bm25, ["a", "b"], {"a": 1, "b": 2})
    assert out["raw_differential"] == pytest.approx(
        out["per_class"]["bridge_absent"]["raw_deficit"]
        - out["per_class"]["comparison"]["raw_deficit"])
    assert out["raw_differential"] > 0  # graph does relatively better where the bridge is absent


def test_headroom_omits_the_normalised_differential_at_ceiling():
    """KILLS: differencing a None normalised deficit against a float.

    A class whose BM25 baseline is at ceiling has zero headroom, so its normalised deficit is
    undefined rather than zero. The raw differential is still reported; the normalised one is
    absent rather than crashing. Found by the ponytail pass, not by the council.
    """
    graph = {"a": {"recall_2": 1.0}, "b": {"recall_2": 0.5}}
    bm25 = {"a": {"recall_2": 1.0}, "b": {"recall_2": 0.6}}
    out = an.headroom(graph, bm25, ["a", "b"], {"a": 1, "b": 2})
    assert "raw_differential" in out
    assert "normalised_differential" not in out


def test_unclassified_queries_join_neither_arm():
    """KILLS: a default that sweeps unclassified queries into a class.

    This file previously spelled the same lookup three ways — .get(q, 2), .get(q) and
    .get(q, -1). Two of those would have assigned an unclassified query to an arm.
    """
    scores = {"x": {"recall_2": 0.5}}
    assert an.headroom(scores, scores, ["x"], {})["per_class"] == {}

    # The comparison class must be NON-EMPTY, or `_contrast`'s own "a class is empty" guard
    # returns the same result with or without the defect and the test proves nothing. That is
    # precisely how NB-24 shipped three vacuous tests, and the mutation sweep caught this one
    # the same way: the unclassified query must have somewhere wrong to go.
    # BOTH arms must be non-empty, or `_contrast`'s own "a class is empty" guard short-circuits
    # and the test returns the same result with or without the defect. My first attempt at this
    # test did exactly that and the mutation sweep caught it — the same shape as NB-24's three
    # vacuous tests. The unclassified query needs a populated class to be wrongly swept into.
    diffs = {"absent": 1.0, "named": 0.0, "unclassified": 1.0}
    contrast = an._contrast(diffs, {"absent": 1, "named": 2})
    assert "error" not in contrast
    assert contrast["n_bridge_absent"] == 1, "an unclassified query was swept into a class"
    assert contrast["n_comparison"] == 1


def test_run_returns_the_three_artifacts_separately_and_leaks_none(monkeypatch, tmp_path):
    """KILLS: smuggling the headroom control -- or the post-hoc decomposition -- through the
    results dict.

    The only path that wires headroom()'s real inputs into run(), and the only path that keeps
    it OUT of the published analysis.json, was exercised by nothing but a live invocation. It
    previously travelled under a leading-underscore key that __main__ had to remember to pop;
    a caller who forgot would have leaked it into the artifact.

    Drives run() against two fabricated arms rather than the real 66,581-passage pool.
    """
    qrels = {f"q{i}": {"d1": 1, "d2": 1} for i in range(4)}
    graph = {q: {"recall_2": 0.5, "recall_5": 0.5, "ndcg_cut_10": 0.5,
                 "recall_10": 0.5, "recall_100": 0.5} for q in qrels}
    bm25 = {q: {"recall_2": 0.6, "recall_5": 0.6, "ndcg_cut_10": 0.6,
                "recall_10": 0.6, "recall_100": 0.6} for q in qrels}
    classes = {"q0": 0, "q1": 1, "q2": 2, "q3": 2}

    monkeypatch.setattr(an.datasets, "load_qrels", lambda *a, **k: qrels)
    monkeypatch.setattr(an, "_per_query", lambda arm, qr: graph if arm == "graph" else bm25)
    monkeypatch.setattr(an, "_classes", lambda definition: classes)
    monkeypatch.setattr(an, "B", 200)  # keep the bootstrap quick

    results, head, decomposition = an.run()

    # The triple is explicit, and no artifact carries another.
    assert "_headroom" not in results
    # AMENDMENT 5. The decomposition is post-hoc. It must not appear inside `contrasts` and must
    # not inflate `family_size`, or Holm would be correcting over a family section 7 never
    # registered and a reader would read an after-the-fact statistic as a pre-registered one.
    assert "decomposition" not in results
    assert results["family_size"] == len(results["contrasts"]) == 12
    assert all(not k.startswith(("A|", "B|")) for k in decomposition)
    assert set(decomposition) and all("|" in k for k in decomposition)
    assert "per_class" not in results
    assert "contrasts" not in head
    assert results["queries"] == 4
    assert head["per_class"]["comparison"]["n"] == 2
    assert head["per_class"]["bridge_absent"]["n"] == 2
    # Every contrast must carry a Holm-adjusted value, and none may be an impossible zero.
    assert all("p_holm" in v for v in results["contrasts"].values())
    assert all(v["p_value"] > 0 for v in results["contrasts"].values())
