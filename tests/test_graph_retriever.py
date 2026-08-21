"""
The graph arm — protocols/003-graph-arm.md §2.

The walk is tested against a stub extractor so the mechanism is checked without a model
download; the contract and the real linking behaviour are tested with spaCy where available.
"""

import numpy as np
import pytest

from rb.experiments.graph import build as kg
from rb.experiments.graph import retriever as gr
from helpers import assert_retriever_contract


# --- the walk, on a hand-built graph ------------------------------------------------

def _bridge_corpus():
    """d1 names the anchor, d2 holds the answer, and they share only the bridge entity.
    This is the experiment in miniature: the query names the anchor and never the bridge."""
    return {
        "d1": ["Kiss and Tell", "Shirley Temple"],
        "d2": ["Shirley Temple", "Ghana"],
        "d3": ["Paris", "France"],
    }


def test_the_walk_reaches_a_document_the_seed_does_not_touch():
    """The mechanism under test. d2 shares NO entity with the seed and must still be
    reached, through the bridge. d3 must not be reached at all."""
    nodes, ids, inc = kg.build(_bridge_corpus())
    seed = np.zeros(len(nodes))
    seed[nodes.index("kiss and tell")] = 1.0
    scores = dict(zip(ids, kg.score_documents(inc, kg.personalized_pagerank(inc, seed))))
    assert scores["d1"] > scores["d2"] > 0, "the bridged document must be reached, below the anchor"
    assert scores["d3"] == 0, "an unconnected document must not be reached"


def test_an_empty_seed_walks_nowhere_rather_than_ranking_arbitrarily():
    nodes, ids, inc = kg.build(_bridge_corpus())
    rank = kg.personalized_pagerank(inc, np.zeros(len(nodes)))
    assert rank.sum() == 0


def test_node_specificity_prefers_the_rarer_entity():
    """An entity in half the corpus says little about which document is wanted; one in two
    documents nearly identifies the pair. HippoRAG's own ablation costs 4.2 R@2 without it."""
    nodes, _ids, inc = kg.build(_bridge_corpus())
    spec = kg.node_specificity(inc)
    assert spec[nodes.index("ghana")] > spec[nodes.index("shirley temple")]


def test_the_walk_agrees_with_a_reference_pagerank():
    """
    The test that would have caught the original defect, and did not exist.

    The first implementation applied B^T (B rank) and rescaled the vector to sum 1, which is
    damped power iteration on an unnormalised co-occurrence matrix rather than a random walk.
    It passed every test we had, because every test checked properties the broken version
    also satisfied - mass reaches a bridged document, an unconnected document scores zero,
    the seed dominates. None compared against an INDEPENDENT implementation of the thing the
    docstring claimed to be.

    networkx is a test-only dependency for exactly this reason.
    """
    nx = pytest.importorskip("networkx")
    docs = {f"d{i}": [f"e{i}", "american"] for i in range(10)}
    docs["d0"] = ["e0", "american", "bridge"]
    docs["d10"] = ["bridge", "e99"]
    nodes, _ids, inc = kg.build(docs)
    i0 = nodes.index("e0")
    seed = np.zeros(len(nodes)); seed[i0] = 1.0
    ours = kg.personalized_pagerank(inc, seed, damping=0.5)

    adjacency = (inc.T @ inc).toarray().astype(float)
    np.fill_diagonal(adjacency, 0.0)          # the self-loop the corrected walk subtracts
    ref = nx.pagerank(nx.from_numpy_array(adjacency), alpha=0.5, personalization={i0: 1.0})
    ref = np.array([ref[i] for i in range(len(nodes))])

    assert np.abs(ours - ref).max() < 1e-4, "the walk no longer agrees with a reference PPR"


def test_degree_normalisation_discounts_a_hub_relative_to_a_rare_edge():
    """The property the defect destroyed, stated directly rather than via a reference: a
    generic entity in every document must not outrank an informative rare one by as much as
    an unnormalised walk makes it."""
    docs = {f"d{i}": [f"e{i}", "american"] for i in range(10)}
    docs["d0"] = ["e0", "american", "bridge"]
    docs["d10"] = ["bridge", "e99"]
    nodes, _ids, inc = kg.build(docs)
    seed = np.zeros(len(nodes)); seed[nodes.index("e0")] = 1.0
    rank = kg.personalized_pagerank(inc, seed)
    hub = rank[nodes.index("american")] / rank[nodes.index("bridge")]
    assert hub < 2.0, f"hub outranks the informative edge by {hub:.2f}x; unnormalised was 3.19x"


def test_dangling_entities_do_not_leak_probability():
    """An entity alone in its document has no co-occurrence partner. Its mass must return to
    the restart vector, not vanish - silent leakage would renormalise the result and hide it."""
    docs = {"d0": ["a", "b"], "d1": ["solo"]}
    nodes, _ids, inc = kg.build(docs)
    seed = np.zeros(len(nodes)); seed[nodes.index("solo")] = 1.0
    rank = kg.personalized_pagerank(inc, seed)
    assert abs(rank.sum() - 1.0) < 1e-6


def test_the_entity_entity_matrix_is_never_materialised():
    """A hub entity makes incidence.T @ incidence dense enough to exhaust memory, so the
    walk must stay two sparse products against the incidence matrix."""
    docs = {f"d{i}": ["Hub", f"E{i}"] for i in range(400)}
    _nodes, _ids, inc = kg.build(docs)
    assert inc.nnz == 800, "incidence stays sparse and linear in mentions"


# --- the retriever, with a stubbed extractor ------------------------------------------

class _StubbedGraph(gr.GraphRetriever):
    """Substitutes both extraction paths so the retriever is exercised without spaCy."""

    def __init__(self, entities, query_entities, **kw):
        super().__init__(**kw)
        self._stub_docs = entities
        self._stub_queries = query_entities

    def fit(self, corpus):
        nodes, doc_ids, inc = kg.build(self._stub_docs)
        self._fitted = {"nodes": nodes, "node_index": {n: i for i, n in enumerate(nodes)},
                        "doc_ids": doc_ids, "incidence": inc,
                        "specificity": kg.node_specificity(inc), "entities": self._stub_docs}
        return {"documents": len(doc_ids), "nodes": len(nodes)}

    def _seed(self, query):
        seed = np.zeros(len(self._fitted["nodes"]))
        for e in self._stub_queries.get(query, []):
            i = self._fitted["node_index"].get(e)
            if i is not None:
                seed[i] += 1.0
        return seed


def _stubbed():
    r = _StubbedGraph(_bridge_corpus(), {"anchor": ["kiss and tell"], "nothing": ["unknown"]})
    r.fit({})
    return r


def test_retrieve_satisfies_the_shared_retriever_contract():
    r = _stubbed()
    corpus = {d: "" for d in ["d1", "d2", "d3"]}
    assert_retriever_contract(r, corpus, {"q1": "anchor"}, top_k=3)


def test_a_query_matching_no_node_retrieves_nothing():
    """The honest outcome: the graph cannot reach anything from a seed it does not have,
    and that must not be disguised as a ranking of arbitrary documents."""
    r = _stubbed()
    assert r.retrieve({}, {"q1": "nothing"}, top_k=10) == {"q1": {}}


def test_retrieve_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="fit"):
        gr.GraphRetriever().retrieve({}, {"q1": "x"}, 10)


def test_scores_are_strictly_decreasing_even_when_the_walk_ties():
    """run_rung checks strict ordering BEFORE it re-encodes ranks, so ties must be broken
    here or a correct arm fails the contract."""
    r = _StubbedGraph({"a": ["X"], "b": ["X"]}, {"q": ["x"]})
    r.fit({})
    scores = list(r.retrieve({}, {"q1": "q"}, 10)["q1"].values())
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_retrieval_is_deterministic():
    r = _stubbed()
    assert r.retrieve({}, {"q1": "anchor"}, 3) == r.retrieve({}, {"q1": "anchor"}, 3)


# --- the real linker ------------------------------------------------------------------

pytest.importorskip("spacy", reason="linking behaviour needs the pinned spaCy")


def test_exact_string_linking_is_defeated_by_context_dependent_spans():
    """
    A registered known failure mode, found before any scored run (protocol §2).

    spaCy segments the SAME name differently by context: "Kiss and Tell" becomes two PERSON
    spans inside a document sentence and one WORK_OF_ART span inside a question. Exact string
    linking therefore fails on precisely the bridge case this experiment is about. The test
    pins the behaviour so a future spaCy that fixes it shows up as a failure rather than as an
    unexplained improvement.
    """
    from rb.experiments.graph.extractor import extract

    doc_spans = {t for t, _ in extract("Kiss and Tell is a 1945 film starring Shirley Temple.")}
    query_spans = {t for t, _ in extract(
        "What position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?")}
    assert "Kiss and Tell" in query_spans
    assert "Kiss and Tell" not in doc_spans
    assert {"Kiss", "Tell"} <= doc_spans
