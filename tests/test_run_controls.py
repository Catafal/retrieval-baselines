"""
The graph summary block, whose producer had vanished (NB-25 D1, fifth defect).

MUTATION-CHECKED. NB-24 shipped three tests that passed whether or not the defect was present:
one raised the error it asserted, one used a topology where the term under test normalised
away, one asserted a sum the code's own guard restored even when the invariant was violated.
A test that passes is not evidence; only a test that FAILS against the defect is. Each test
below names the mutation it kills. See NB-25 for the audit these close.
"""

import pytest

from rb.experiments.graph import run_controls as rc


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
