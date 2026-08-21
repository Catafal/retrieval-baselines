"""
Query-side seeding: the graph is weighted by one rule on both sides (NB-25 D4).

MUTATION-CHECKED. NB-24 shipped three tests that passed whether or not the defect was present:
one raised the error it asserted, one used a topology where the term under test normalised
away, one asserted a sum the code's own guard restored even when the invariant was violated.
A test that passes is not evidence; only a test that FAILS against the defect is. Each test
below names the mutation it kills. See NB-25 for the audit these close.
"""

import pytest

from rb.experiments.graph import build as kg
from rb.experiments.graph import retriever as gr


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
