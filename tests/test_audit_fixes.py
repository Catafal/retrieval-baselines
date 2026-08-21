"""
NB-25 — the four defects the second audit found, and the fifth the council found.

EVERY TEST HERE IS MUTATION-CHECKED. NB-24 wrote three tests that passed whether or not the
defect was present: one raised the error it asserted, one used a graph topology where the term
under test normalised away, and one asserted a sum the code's own guard restored even when the
invariant was violated. A test that passes is not evidence; only a test that FAILS against the
defect is. Each test below names the mutation it kills.
"""

import json
import numpy as np
import pytest

from rb import stats
from rb.experiments.graph import build as kg
from rb.experiments.graph import entity_types as et
from rb.experiments.graph import retriever as gr
from rb.experiments.graph import run_controls as rc


# ---------------------------------------------------------------- D2: the p-value estimator

def test_p_value_cannot_be_zero_when_no_resample_crosses():
    """KILLS: reverting to `2 * min(c_le, c_ge) / B`, which returns exactly 0.0 here.

    Every draw strictly positive, so the light tail has a count of zero — the case that
    produced all twelve published zeros.
    """
    p = stats.bootstrap_p_value([1.0] * 10_000, 10_000)
    assert p > 0.0
    assert p == pytest.approx(2 / 10_001)


def test_p_value_floor_is_not_one_over_b():
    """KILLS: `max(p, 1/B)`, the fix originally specified.

    1/B = 0.0001 is a value the two-sided statistic cannot emit — its counts are integers and
    the `2 *` doubles them. A floor there would claim a resolution finer than the procedure
    has. This pins the actual smallest value ABOVE 1/B.
    """
    p = stats.bootstrap_p_value([1.0] * 10_000, 10_000)
    assert p > 1 / 10_000


def test_p_value_is_not_uniformly_the_minimum():
    """KILLS: an implementation that always returns the smallest value.

    A distribution straddling zero must report a large p. Without this, "always return 2/(B+1)"
    would pass the two tests above.
    """
    draws = [-1.0] * 5_000 + [1.0] * 5_000
    assert stats.bootstrap_p_value(draws, 10_000) > 0.9


def test_p_value_is_two_sided_and_capped():
    """A perfectly symmetric distribution cannot report p > 1."""
    assert stats.bootstrap_p_value([-1.0, 1.0], 2) <= 1.0


# --------------------------------------------------------------------------- D5: one Holm

def _holm_adjusted_reference(p_values):
    """Frozen copy of the pre-refactor algorithm, for differential testing.

    Kept deliberately: the spec's hand-picked cases only cover edge cases someone thought of,
    and a differential test against the original covers the ones nobody did.
    """
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m, out, running = len(ordered), [0.0] * len(p_values), 0.0
    for i, idx in enumerate(ordered):
        running = min(1.0, max(running, (m - i) * p_values[idx]))
        out[idx] = running
    return out


def test_holm_adjusted_applies_the_running_maximum():
    """KILLS: `adj = min(1.0, (m - rank) * p)` without `max(running, ...)`.

    Scaled values here are [0.08, 0.045] — non-monotone. Correct output is [0.08, 0.08];
    the mutant returns [0.08, 0.045]. Adjusted p-values must be non-decreasing in rank.
    """
    assert stats.holm_adjusted([0.04, 0.045]) == pytest.approx([0.08, 0.08])


def test_holm_adjusted_returns_input_order_not_sorted_order():
    """KILLS: returning the values in ascending-p order.

    002's ladder passes deliberately-unsorted p-values and zips the result against its own
    labels, so sorted output would silently mislabel every published comparison.
    """
    assert stats.holm_adjusted([0.5, 0.01])[1] < stats.holm_adjusted([0.5, 0.01])[0]


def test_holm_adjusted_handles_ties():
    """Tied p-values must receive identical adjusted values, not order-dependent ones."""
    adj = stats.holm_adjusted([0.02, 0.02, 0.02])
    assert adj[0] == adj[1] == adj[2]


def test_holm_adjusted_matches_the_frozen_reference_implementation():
    """DIFFERENTIAL: the refactor must not change any value, on any input shape."""
    rng = np.random.default_rng(20260821)
    for _ in range(2_000):
        m = int(rng.integers(1, 8))
        # Coarse rounding deliberately manufactures ties, the case most likely to diverge.
        ps = [round(float(x), 2) for x in rng.random(m)]
        assert stats.holm_adjusted(ps) == pytest.approx(_holm_adjusted_reference(ps))


def test_holm_correction_still_matches_the_step_down_decisions():
    """KILLS: any change to holm_correction's decisions. 002 is PUBLISHED on this function.

    Reimplements the original break-based step-down and requires identical output.
    """
    def original(p_values, alpha=0.05):
        m = len(p_values)
        order = sorted(range(m), key=lambda i: p_values[i])
        sig = [False] * m
        for rank, idx in enumerate(order):
            if p_values[idx] <= alpha / (m - rank):
                sig[idx] = True
            else:
                break
        return sig

    rng = np.random.default_rng(4242)
    for _ in range(2_000):
        m = int(rng.integers(1, 8))
        ps = [round(float(x), 3) for x in rng.random(m) * 0.2]
        assert stats.holm_correction(ps) == original(ps)


def test_holm_correction_on_002s_committed_p_values_is_unchanged():
    """The published artifact, not a synthetic fixture: every stored decision must still hold."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for path in sorted((root / "results" / "002").glob("*/*.json")):
        payload = json.loads(path.read_text())

        def walk(node):
            nonlocal checked
            if isinstance(node, dict):
                for value in node.values():
                    if (isinstance(value, list) and value and isinstance(value[0], dict)
                            and "p_value" in value[0] and "holm_significant" in value[0]):
                        ps = [c["p_value"] for c in value]
                        stored = [c["holm_significant"] for c in value]
                        assert stats.holm_correction(ps) == stored, path
                        checked += 1
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(payload)
    assert checked > 0, "no 002 Holm families found — this test would be vacuous"


# ------------------------------------------------------------------- D3: the partition check

def test_partition_rejects_a_type_that_is_both_kept_and_excluded(monkeypatch):
    """KILLS: deleting the disjointness branch.

    Monkeypatches the module global and calls the REAL function, so the production logic is
    what runs. NB-24's partition test constructed and raised the error itself, which made it
    satisfied by its own setup rather than by the code under test.

    The `match=` is REQUIRED and must differ from the coverage test's: two tests asserting a
    bare RuntimeError would both pass while only the disjointness branch existed, which is
    exactly the shipped bug.
    """
    monkeypatch.setattr(et, "WHITELIST", et.WHITELIST | {"DATE"})
    with pytest.raises(RuntimeError, match="cannot be both kept and excluded"):
        et.assert_partition()


def test_partition_rejects_a_model_label_in_neither_set():
    """KILLS: deleting the coverage branch — i.e. reverting to the shipped bug.

    Distinct `match=` from the disjointness test, so this cannot pass by accidentally
    exercising the other branch. Passes labels directly: no monkeypatching needed, because the
    check takes its inventory as an argument.
    """
    with pytest.raises(RuntimeError, match="classifies neither way"):
        et.assert_partition(set(et.WHITELIST | et.EXCLUDED) | {"BRAND_NEW_TYPE"})


def test_partition_rejects_a_declared_type_the_model_never_emits():
    """KILLS: checking only one direction of the set difference."""
    with pytest.raises(RuntimeError, match="never emits"):
        et.assert_partition(set(et.WHITELIST | et.EXCLUDED) - {"PERSON"})


def test_partition_accepts_the_real_inventory():
    """The declared sets must exactly cover the pinned model's labels — the live invariant."""
    et.assert_partition(set(et.WHITELIST | et.EXCLUDED))


def test_partition_without_labels_cannot_check_coverage():
    """Pins the documented contract: the no-argument form is disjointness ONLY.

    Stops a future 'improvement' from adding a hand-written label constant here, which is the
    self-defeating design this fix rejected.
    """
    et.assert_partition()  # must not raise despite no inventory being supplied


# ------------------------------------------------------------------------ D4: the seed dedup

def _fitted_retriever(docs):
    """A real GraphRetriever with real fitted state, WITHOUT the _StubbedGraph subclass.

    _StubbedGraph overrides _seed wholesale with a flat 1.0 and never calls normalise, so a
    test written against it passes with or without this fix — its own docstring records that
    this stub is why the whitelist bug stayed invisible. The seam used instead is
    `_query_entities`, which exists precisely so the walk can run without a model.
    """
    r = gr.GraphRetriever()
    nodes, doc_ids, inc = kg.build(docs)
    r._fitted = {
        "nodes": nodes, "node_index": {n: i for i, n in enumerate(nodes)},
        "doc_ids": doc_ids, "doc_id_set": frozenset(doc_ids), "incidence": inc,
        "specificity": kg.node_specificity(inc), "degrees": kg.degrees(inc), "entities": docs,
    }
    return r


def test_two_surface_forms_of_one_node_seed_it_once(monkeypatch):
    """KILLS: `seed[i] += ...` over surface forms without deduplicating on the normalised key.

    "U.S." and "U.S" are distinct surface forms — node_strings keeps both — that normalise to
    one node. The document side counts that node once (build dedupes after normalise), so the
    query side must too.

    The fixture ALSO contains a node reached by a single surface form ("paris"), and the test
    asserts the two are equal. Without that second node, a mutant that simply halved every
    seed weight would pass.
    """
    docs = {"d1": ["U.S.", "Paris"], "d2": ["U.S.", "Paris"]}
    r = _fitted_retriever(docs)
    monkeypatch.setattr(gr, "_query_entities",
                        lambda q: [("U.S.", "GPE"), ("U.S", "GPE"), ("Paris", "GPE")])
    seed = r._seed("irrelevant")

    us = r._fitted["node_index"]["u s"]
    paris = r._fitted["node_index"]["paris"]
    # Both entities appear in both documents, so their specificity is identical. The duplicated
    # surface form must not buy the node extra weight.
    assert seed[us] == pytest.approx(seed[paris])


def test_a_repeated_surface_form_does_not_compound(monkeypatch):
    """KILLS: the same defect via exact repetition rather than near-variants."""
    docs = {"d1": ["Paris"], "d2": ["Paris"]}
    r = _fitted_retriever(docs)
    monkeypatch.setattr(gr, "_query_entities", lambda q: [("Paris", "GPE")])
    once = r._seed("q")[r._fitted["node_index"]["paris"]]
    monkeypatch.setattr(gr, "_query_entities",
                        lambda q: [("Paris", "GPE"), ("PARIS", "GPE"), ("paris", "GPE")])
    thrice = r._seed("q")[r._fitted["node_index"]["paris"]]
    assert once == pytest.approx(thrice)


def test_distinct_nodes_still_accumulate_separately(monkeypatch):
    """KILLS: a mutant that collapses ALL seeds into one node.

    Deduplication must be per node, not across nodes.
    """
    docs = {"d1": ["Paris", "Berlin"], "d2": ["Paris", "Berlin"]}
    r = _fitted_retriever(docs)
    monkeypatch.setattr(gr, "_query_entities",
                        lambda q: [("Paris", "GPE"), ("Berlin", "GPE")])
    seed = r._seed("q")
    assert (seed > 0).sum() == 2


# --------------------------------------------------------- D1: the graph block's lost producer

# Hard-coded, NOT read from the committed artifact. Once main() rewrites those files, "matches
# the committed file" means "matches whatever main() last wrote" — a tautology that would let a
# renamed or dropped key sail through forever.
_GRAPH_KEYS = {"documents", "documents_with_an_entity",
               "distinct_surface_entities", "mean_entities_per_document"}


def test_graph_summary_schema_is_pinned_to_a_literal():
    """KILLS: renaming or dropping a field of the reconstructed block."""
    assert set(rc.graph_summary({"d1": ["A"]})) == _GRAPH_KEYS


def test_graph_summary_counts_surface_forms_not_normalised_nodes():
    """KILLS: normalising before counting.

    The published 291,837 exceeds the graph's 285,013 nodes precisely because it counted
    surface forms. "U.S." and "U.S" are two here and one in the graph.
    """
    out = rc.graph_summary({"d1": ["U.S.", "U.S"], "d2": ["U.S."]})
    assert out["distinct_surface_entities"] == 2


def test_graph_summary_reports_documents_with_no_entities():
    """KILLS: counting all documents as populated — the 594 empty ones are the interesting tail."""
    out = rc.graph_summary({"d1": ["A"], "d2": [], "d3": []})
    assert out["documents"] == 3
    assert out["documents_with_an_entity"] == 1
    assert out["mean_entities_per_document"] == pytest.approx(1 / 3, abs=0.01)


def test_graph_summary_is_defined_on_an_empty_corpus():
    """No ZeroDivisionError on the degenerate input."""
    assert rc.graph_summary({})["documents"] == 0
