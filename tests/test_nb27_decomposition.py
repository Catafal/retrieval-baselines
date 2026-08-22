"""
NB-27 / amendment 5 — the post-hoc decomposition of prediction A's differential.

The algebraic identity is the thing worth testing: the two terms must sum to exactly the
differential `_contrast` reports, or the decomposition is describing a different quantity than the
registered one and the amendment's whole claim collapses.
"""

import pytest

from rb.experiments.graph.analysis import _contrast, decompose


def _fixture():
    """Two classes with clearly different per-arm behaviour, so both terms are non-trivial."""
    graph = {f"a{i}": {"recall_2": 0.20 + 0.01 * (i % 3)} for i in range(12)}
    graph.update({f"c{i}": {"recall_2": 0.18 + 0.01 * (i % 3)} for i in range(8)})
    bm25 = {f"a{i}": {"recall_2": 0.50 + 0.02 * (i % 4)} for i in range(12)}
    bm25.update({f"c{i}": {"recall_2": 0.62 + 0.02 * (i % 4)} for i in range(8)})
    classes = {f"a{i}": (i % 2) for i in range(12)}          # coverage 0/1 -> bridge_absent
    classes.update({f"c{i}": 2 for i in range(8)})            # coverage 2
    return graph, bm25, sorted(graph), classes


def test_the_two_terms_sum_to_the_registered_differential():
    """KILLS: any sign or grouping error in decompose().

    `_contrast` computes the differential the protocol registered. `decompose` claims to split
    THAT quantity. If the terms do not sum back to it, the amendment is decomposing something
    else and its conclusion does not transfer.
    """
    graph, bm25, shared, classes = _fixture()
    diffs = {q: graph[q]["recall_2"] - bm25[q]["recall_2"] for q in shared}

    registered = _contrast(diffs, classes)["difference"]
    d = decompose(graph, bm25, shared, classes)

    assert d["graph_term"]["point"] + d["bm25_term"]["point"] == pytest.approx(registered, abs=1e-4)
    assert d["differential"] == pytest.approx(registered, abs=1e-4)


def test_a_graph_flat_across_classes_puts_the_whole_differential_on_bm25():
    """The exact situation 003 found, constructed deliberately.

    When the graph scores identically in both classes, every bit of the differential must be
    attributed to the baseline — a share of 1.0 and a graph term of exactly zero. This is what
    makes the amendment's headline claim checkable rather than asserted.
    """
    graph = {f"a{i}": {"recall_2": 0.21} for i in range(10)}
    graph.update({f"c{i}": {"recall_2": 0.21} for i in range(10)})
    bm25 = {f"a{i}": {"recall_2": 0.53} for i in range(10)}
    bm25.update({f"c{i}": {"recall_2": 0.60} for i in range(10)})
    classes = {f"a{i}": 1 for i in range(10)}
    classes.update({f"c{i}": 2 for i in range(10)})

    d = decompose(graph, bm25, sorted(graph), classes)
    assert d["graph_term"]["point"] == 0.0
    assert d["bm25_share_of_differential"]["point"] == pytest.approx(1.0)
    assert d["graph_term"]["excludes_zero"] is False


def test_a_graph_carrying_the_effect_attributes_it_to_the_graph():
    """The converse, so the test above is not passing for a degenerate reason. If the mechanism
    HAD worked, the decomposition must say so."""
    graph = {f"a{i}": {"recall_2": 0.40} for i in range(10)}
    graph.update({f"c{i}": {"recall_2": 0.21} for i in range(10)})
    bm25 = {f"a{i}": {"recall_2": 0.55} for i in range(10)}
    bm25.update({f"c{i}": {"recall_2": 0.55} for i in range(10)})
    classes = {f"a{i}": 1 for i in range(10)}
    classes.update({f"c{i}": 2 for i in range(10)})

    d = decompose(graph, bm25, sorted(graph), classes)
    assert d["graph_term"]["point"] == pytest.approx(0.19, abs=1e-4)
    assert d["bm25_term"]["point"] == 0.0
    assert d["bm25_share_of_differential"]["point"] == 0.0


def test_decomposition_is_absent_from_the_registered_family():
    """The post-hoc statistic must not leak into `contrasts` or inflate `family_size`, or Holm
    would be correcting over a family the protocol did not register."""
    from rb.experiments.graph import analysis

    src = analysis.run.__doc__ or ""
    # Structural check: run() returns the decomposition as its own element, never merged in.
    import inspect
    body = inspect.getsource(analysis.run)
    assert "decomposition" in body
    assert 'results["decomposition"]' not in body and "results.update(decomposition" not in body
