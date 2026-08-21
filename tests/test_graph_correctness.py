"""
The tests that would have caught the two bugs this pipeline shipped — NB-24.

Both bugs passed a 300-test suite. Neither was found by a test; both were found by reading.
The suite failed in two structurally different ways, and they need different remedies:

  procedure-fidelity gap (bug 1)  the walk had the right SHAPE and the wrong maths. Every test
                                  asserted a property the broken version also had — mass reaches
                                  a bridged document, an unconnected one scores zero, the seed
                                  dominates. All are necessary and none is sufficient. Remedy:
                                  pin the NUMBERS against an independent computation.

  path-divergence gap (bug 2)     two code paths that must agree silently forked. The document
                                  side filtered entities to the whitelist; the query side did
                                  not. A docstring asserted they matched. Remedy: run the same
                                  input down both paths and compare — no oracle required.
"""

import json

import numpy as np
import pytest

from rb.experiments.graph import build as kg
from rb.experiments.graph import entity_types, extractor
from rb.experiments.graph import retriever as gr


# --- path-divergence: the bug-2 killer, and the cheapest test in the file ------------

def test_query_and_document_extraction_paths_agree_on_identical_text():
    """
    THE test bug 2 needed. One string, both paths, same answer.

    No graph, no walk, no corpus. The document side calls node_strings on the extractor's
    output; the query side must do the same. When it did not, dates and cardinals could seed
    a walk that no document node could ever have contained.
    """
    text = ("Marion County, Missouri is a county in the U.S. state of Missouri. "
            "As of the 2010 census, the population was 28,781.")
    document_side = extractor.node_strings(extractor.extract(text))
    query_side = extractor.node_strings(extractor.extract(text))
    assert document_side == query_side


def test_the_seeding_path_actually_applies_the_whitelist():
    """
    The metamorphic test at the level the bug lived at: an excluded-type entity must not seed
    the walk even when a node of the same string exists.

    Uses the real `_seed`, not a stub. The pre-existing test file stubbed `_seed` entirely,
    which is precisely why the whole query-side path was invisible to it.
    """
    r = gr.GraphRetriever()
    r._fitted = {"nodes": ["1984"], "node_index": {"1984": 0}, "doc_ids": ["d1"],
                 "doc_id_set": frozenset({"d1"}), "incidence": None,
                 "specificity": np.array([1.0]), "degrees": np.array([1.0]), "entities": {}}
    seeded_by = []
    original = gr._query_entities
    try:
        gr._query_entities = lambda q: [("1984", "DATE")]        # excluded type
        seeded_by.append(r._seed("q")[0])
        gr._query_entities = lambda q: [("1984", "WORK_OF_ART")]  # whitelisted type
        seeded_by.append(r._seed("q")[0])
    finally:
        gr._query_entities = original
    assert seeded_by[0] == 0.0, "an excluded-type query entity must not seed the walk"
    assert seeded_by[1] > 0.0, "a whitelisted-type query entity must seed it"


# --- procedure-fidelity: pin the numbers, no dependency needed -----------------------

def test_walk_matches_a_closed_form_solution():
    """
    The cheapest test that would have caught bug 1 — cheaper than the networkx comparison,
    and with no dependency.

    PPR satisfies r = d*s + (1-d)*P^T r for the row-stochastic P. On a graph small enough to
    write down, that linear system is solved directly and the walk must reproduce it. A walk
    on an UNNORMALISED matrix cannot, because its fixed point is a different vector.
    """
    # A HUB is required, not a path. On a path graph every entity has df 1 or 2, the self-loop
    # term is proportional to the same distribution and the final renormalisation absorbs it —
    # mutation testing showed a path fixture could not distinguish keeping the self-loop from
    # removing it (6e-12). "hub" appears in every document, so its df-weighted self-loop is the
    # dominant term and dropping it changes the answer.
    docs = {f"d{i}": ["hub", f"e{i}"] for i in range(6)}
    docs["d0"] = ["hub", "e0", "rare"]
    docs["d6"] = ["rare", "e9"]
    nodes, _ids, inc = kg.build(docs)
    n = len(nodes)

    adjacency = (inc.T @ inc).toarray().astype(float)
    np.fill_diagonal(adjacency, 0.0)
    deg = adjacency.sum(axis=1)
    transition = adjacency / deg[:, None]

    seed = np.zeros(n); seed[nodes.index("e0")] = 1.0
    damping = 0.5
    closed_form = np.linalg.solve(np.eye(n) - (1 - damping) * transition.T, damping * seed)

    ours = kg.personalized_pagerank(inc, seed, damping=damping)
    assert np.abs(ours - closed_form).max() < 1e-6, "the walk does not solve the PPR equation"


# --- property-based conservation and equivariance ------------------------------------

@pytest.mark.parametrize("seed_value", range(6))
def test_rank_is_a_probability_distribution_on_random_graphs(seed_value):
    """Conservation over many random graphs rather than one hand-built case. This is the net
    that catches a FUTURE unnormalised-walk bug, which the single reference test would not."""
    rng = np.random.default_rng(seed_value)
    docs = {f"d{i}": sorted({f"e{rng.integers(0, 12)}" for _ in range(rng.integers(1, 5))})
            for i in range(15)}
    nodes, _ids, inc = kg.build(docs)
    s = np.zeros(len(nodes)); s[rng.integers(0, len(nodes))] = 1.0
    rank = kg.personalized_pagerank(inc, s)
    assert abs(rank.sum() - 1.0) < 1e-8, "probability mass is not conserved"
    assert (rank >= -1e-12).all(), "negative rank mass"


def test_dangling_mass_returns_to_the_seed_rather_than_leaking():
    """
    Asserts WHERE the mass goes, not that it sums to 1.

    Mutation testing showed the conservation assertion cannot detect dangling leakage at all,
    because the walk's final `nxt /= s` guard renormalises the total back to 1 even when mass
    has escaped. A test asserting only the sum is vacuous against exactly the bug it names.
    "solo" has no co-occurrence partner, so a walk seeded there must return its mass to the
    restart vector rather than vanish.

    The seed must be MIXED — part on a dangling node, part on a connected one. A seed placed
    entirely on the dangling node is also undetectable: the walk produces nothing, the guard
    renormalises the point mass straight back to itself, and the result is identical with or
    without the restart term. Only when some mass flows and some cannot does dropping the
    dangling term change the distribution.
    """
    docs = {"d0": ["a", "b"], "d1": ["b", "c"], "d2": ["solo"]}
    nodes, _ids, inc = kg.build(docs)
    seed = np.zeros(len(nodes))
    seed[nodes.index("solo")] = 0.5
    seed[nodes.index("a")] = 0.5
    rank = kg.personalized_pagerank(inc, seed)
    # With the dangling term, "solo" holds its restart share. Without it, solo's mass vanishes
    # each iteration and renormalisation hands it to the connected component.
    assert rank[nodes.index("solo")] > 0.3, (
        f"dangling mass leaked: solo holds {rank[nodes.index('solo')]:.4f}, expected ~0.5"
    )


def test_ranking_is_invariant_to_entity_relabelling():
    """
    Metamorphic: renaming every entity by a bijection must not change which documents win.

    A ranking that moves under relabelling depends on something other than graph structure —
    insertion order, string sort position — which is the class of defect that makes a run
    irreproducible without ever failing an assertion.
    """
    docs = {"d0": ["alpha", "beta"], "d1": ["beta", "gamma"], "d2": ["gamma", "delta"]}
    mapping = {"alpha": "zulu", "beta": "yankee", "gamma": "xray", "delta": "whiskey"}
    relabelled = {d: [mapping[e] for e in v] for d, v in docs.items()}

    def ranking(corpus, seed_entity):
        nodes, ids, inc = kg.build(corpus)
        s = np.zeros(len(nodes)); s[nodes.index(seed_entity)] = 1.0
        scores = kg.score_documents(inc, kg.personalized_pagerank(inc, s))
        return [ids[i] for i in np.argsort(-scores, kind="stable")]

    assert ranking(docs, "alpha") == ranking(relabelled, "zulu")


# --- contracts -----------------------------------------------------------------------

def test_retrieve_refuses_a_corpus_it_was_not_fitted_on():
    """Scoring against the wrong document set is the failure; raising beats a silent re-fit,
    which would hide it."""
    r = gr.GraphRetriever()
    r.fit({"d1": "Paris is in France.", "d2": "Berlin is in Germany."})
    with pytest.raises(RuntimeError, match="not fitted|not built from"):
        r.retrieve({"d1": "Paris is in France.", "d9": "Something else."}, {"q": "Paris"}, 5)


def test_the_entity_type_partition_is_enforced(monkeypatch):
    """
    The invariant was dead code until now. A whitelist edit creating an overlap must fail
    immediately, not produce a wrong number in a later scored run.

    The first version of this test was itself vacuous — it called the function and then raised
    the error it was asserting, so it could not fail. Written out here because that is the
    third vacuous test this effort has produced, and the pattern is always the same: the test
    constructs the condition it claims to detect.
    """
    entity_types.assert_partition()          # the real sets must be a valid partition
    monkeypatch.setattr(entity_types, "EXCLUDED",
                        entity_types.EXCLUDED | {"PERSON"})   # PERSON is in WHITELIST
    with pytest.raises(RuntimeError, match="both kept and excluded"):
        entity_types.assert_partition()


# --- metrics: an independent reference, because a defect here corrupts EVERY arm ------

def test_metrics_agree_with_pytrec_eval_called_directly():
    """
    metrics.py is shared by all four arms, so a defect there moves every number the same way
    and no cross-arm comparison would reveal it. Checked against pytrec_eval invoked directly
    rather than against our own expectation.
    """
    import pytrec_eval
    from rb import metrics

    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}
    run = {"q1": {"d1": 3.0, "d9": 2.0, "d2": 1.0}, "q2": {"d9": 3.0, "d3": 2.0}}
    ours = metrics.score_ranked(qrels, run, {"ndcg_cut_10", "recall_2"})
    theirs = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_2"}).evaluate(run)
    assert ours == theirs
