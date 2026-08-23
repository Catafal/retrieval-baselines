"""
Coverage for the four producer modules added while closing the orphaned artifacts.

A re-verification seat mutated all four and every mutation survived all 383 tests, because no test
file imported any of them. That included re-introducing `oracle.py`'s whitelist-label bug -- the one
`corrections.md` records as having been caught by a reviewer reading the code, never by any output.
Fixing a producer without testing it leaves exactly the hole the producer was written to close.
"""

import json

import pytest

from rb.experiments.graph import ablation, closure, diagnostics, oracle
from rb.experiments.graph import retriever as rmod
from rb.experiments.graph.extraction_score import normalise


# ---------------------------------------------------------------- closure (the §8.1 GATE)

def _closure_result(r2, r5):
    return {"ours": {"recall_2": r2, "recall_5": r5},
            "deltas": {m: round(abs(v - closure.PUBLISHED_BM25[m]), 4)
                       for m, v in (("recall_2", r2), ("recall_5", r5))}}


def test_the_gate_fails_when_our_bm25_is_too_far_from_the_published_row():
    """KILLS: loosening the gate's comparison, or dropping abs() from the deltas.

    A delta computed without abs() is negative when we score BELOW the reference, and every
    negative number passes a `<= tolerance` test. The gate would then only ever fire when our BM25
    was too HIGH, which is the direction that does not indicate a broken harness.
    """
    d = _closure_result(0.554 - 0.2, 0.722)          # 0.2 below the published R@2
    assert d["deltas"]["recall_2"] == pytest.approx(0.2)
    assert not all(v <= closure.TOLERANCE for v in d["deltas"].values())


def test_the_gate_passes_at_the_boundary_but_not_past_it():
    """KILLS: flipping `<=` to `<` (or `<` to `<=`) on the tolerance comparison."""
    at = _closure_result(0.554 + closure.TOLERANCE, 0.722)
    past = _closure_result(0.554 + closure.TOLERANCE + 0.0001, 0.722)
    assert all(v <= closure.TOLERANCE for v in at["deltas"].values())
    assert not all(v <= closure.TOLERANCE for v in past["deltas"].values())


def test_the_published_reference_is_hippo_rags_bm25_row_not_ours():
    """The gate is only a gate against an EXTERNAL number. If this ever recomputes itself from our
    own run it stops checking anything."""
    assert closure.PUBLISHED_BM25 == {"recall_2": 0.554, "recall_5": 0.722}


def test_the_subset_rule_is_stated_and_deterministic():
    """The original draw was unrecoverable (9,811 committed; 9,769 and 9,755 under the two obvious
    rules). Whatever rule replaces it must at least be fixed and repeatable."""
    assert closure.SUBSET_QUESTIONS == 1000
    from rb.experiments.graph import pool
    ctx = pool.load_distractor_context()
    assert sorted(ctx)[:1000] == sorted(ctx)[:1000]


# ---------------------------------------------------------------- diagnostics

def _rows():
    return [{"doc_id": "d1", "title": "Marion County", "text": "irrelevant",
             "entities": ["Marion County", "Missouri"], "annotator": "llm-panel-3x-majority",
             "rule_card": "v1", "rater_jaccard": 1.0},
            {"doc_id": "d2", "title": "Hannibal", "text": "irrelevant",
             "entities": ["Hannibal"], "annotator": "llm-panel-3x-majority",
             "rule_card": "v1", "rater_jaccard": 0.5}]


def test_agreement_reports_the_panel_not_a_single_rater():
    """KILLS: reverting to the single-rater claim.

    The note this replaces said the reference was one rater with no agreement available, while a
    sibling artifact published 0.9356 across three. Derived, so it cannot drift again.
    """
    a = diagnostics.agreement(_rows())
    assert a["raters"] == 3
    assert a["annotator"] == "llm-panel-3x-majority"
    assert a["mean_pairwise_jaccard"] == pytest.approx(0.75)
    assert a["entities_kept"] == 3


def test_a_boundary_disagreement_counts_as_one():
    """KILLS: changing the boundary predicate from substring-containment to exact equality.

    A false positive that CONTAINS or is contained by a gold entity means the extractor found the
    right thing and disagreed about the span. Under exact equality that count collapses to zero and
    the reading -- that most errors are boundary, not hallucination -- silently inverts.
    """
    assert diagnostics.error_analysis  # imported
    gold, pred = {"marion county"}, {"marion county missouri"}
    assert any(p in g or g in p for g in gold for p in pred)
    assert not any(p == g for g in gold for p in pred)


def test_error_analysis_reproduces_the_published_counts():
    """The real sample, against the committed artifact."""
    err = diagnostics.error_analysis(diagnostics.load_sample())
    assert err["false_positives"] == 286
    assert err["false_negatives"] == 322
    assert err["fp_substring_or_superstring_of_a_gold_entity"] == 237


# ---------------------------------------------------------------- ablation

def test_the_mean_divisor_is_guarded_against_documents_with_no_entities():
    """KILLS: dividing by the raw per-document entity count.

    594 pooled documents have no entity at all, so the raw count contains zeros and the mean
    variant would divide by zero -- producing inf or nan scores that sort to the top of the
    ranking rather than an error anyone would notice.
    """
    import numpy as np
    per_doc = np.array([3.0, 0.0, 5.0])
    safe = np.where(per_doc > 0, per_doc, 1.0)
    assert not np.isinf(np.array([1.0, 1.0, 1.0]) / safe).any()
    with np.errstate(divide="ignore"):
        assert np.isinf(np.array([1.0, 1.0, 1.0]) / per_doc).any()


def test_the_registered_scoring_stays_summed():
    """The arm SUMS entity mass, which is what the prior art does. The ablation reports mean beside
    it, never in place of it -- swapping after seeing results would be a method change."""
    import inspect
    src = inspect.getsource(ablation.main)
    assert '("sum", base)' in src and '("mean", base / safe)' in src


# ---------------------------------------------------------------- oracle

def test_oracle_query_entities_survive_the_whitelist():
    """KILLS: labelling oracle entities with a type outside the whitelist.

    `_seed` routes query entities through `node_strings`, which drops any label not in WHITELIST.
    Labelled "ORACLE" every seed vector was zero and the arm retrieved nothing for all 7,405
    queries -- and because an unseeded query legitimately returns an empty list, a totally broken
    oracle is indistinguishable from a very bad one by its output alone.
    """
    from rb.experiments.graph.entity_types import WHITELIST
    from rb.experiments.graph.extractor import node_strings
    assert node_strings([("Missouri", "ORACLE")]) == []
    assert node_strings([("Missouri", "ORG")]) == ["Missouri"]
    assert "ORG" in WHITELIST


def test_oracle_matches_titles_as_whole_word_runs():
    corpus = {"d1": "Missouri is a state. Kansas City sits in Missouri.",
              "d2": "Kansas City has a team."}
    titles = {"d1": "Missouri", "d2": "Kansas City"}
    ents = oracle.oracle_entities(corpus, titles)
    assert set(ents["d1"]) == {"Missouri", "Kansas City"}
    assert set(ents["d2"]) == {"Kansas City"}


def test_oracle_restores_the_monkeypatched_query_extractor():
    """KILLS: removing the try/finally around the module-scope patch.

    `_seed` calls `_query_entities` by module-level name, so an unrestored patch is live for the
    rest of the process and would silently change any later use of the real arm.
    """
    import inspect
    src = inspect.getsource(oracle.main)
    assert "finally:" in src
    assert "rmod._query_entities = original" in src
    assert callable(rmod._query_entities)


# ---------------------------------------------------------------- decomposition exposition

def test_the_two_terms_ADD_to_the_differential():
    """KILLS: publishing terms that do not reconcile with the stated formula.

    The docstring and the registered amendment printed `differential = graph_term - bm25_term`,
    while the code publishes `bm25_term` as the NEGATED baseline swing. A reader hand-verifying
    0.0065 and 0.0651 against 0.0716 by subtraction got -0.0586 and a sign flip. The relation is
    additive and is now stated as such; this pins it so the exposition cannot drift from the code
    again.
    """
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cells = json.loads((root / "results/003/decomposition.json").read_text())["cells"]
    for key, c in cells.items():
        g, b, t = c["graph_term"]["point"], c["bm25_term"]["point"], c["differential"]
        # 2e-4, not 1e-4: each field is independently rounded to four places, so their sum can
        # differ from the rounded differential in the last digit. It does, on recall_5|stripped:
        # -0.0155 + 0.1338 = 0.1183 against a published 0.1182. That is disclosed in the
        # amendment rather than hidden by a looser tolerance here.
        assert round(g + b, 4) == pytest.approx(t, abs=2e-4), f"{key} terms must ADD"


def test_a_negative_control_distinguishes_inconclusive_from_confirmed():
    """KILLS: collapsing 'the interval straddles zero' into 'confirmed no advantage'.

    Section 7's registered flag treats both alike, so it stays as registered. But absence of
    evidence is not evidence of absence, and experiment 004 reuses this path.
    """
    from rb.experiments.graph.analysis import advantage
    import rb.experiments.graph.analysis as an

    entirely_negative = advantage([-0.5] * 30)
    assert entirely_negative["verdict"] == "confirmed_no_advantage"
    assert entirely_negative["no_advantage"] is True

    straddling = advantage([-0.5, 0.5] * 15)
    assert straddling["verdict"] == "inconclusive"
    assert straddling["no_advantage"] is True, "the registered flag is unchanged"
