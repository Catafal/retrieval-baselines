"""
Typed identity — protocols/005-identity.md sections 4 and 5.

The tests that matter here are the ones that would fail if the linker silently did nothing, or
did something on one side only. Both are failure modes that look like a clean result: a linker
that never fires reproduces 004 exactly, and a linker applied to documents but not queries
lowers the seed rate for a reason unrelated to identity.
"""

import pytest

from rb.experiments.graph import build as kg
from rb.experiments.graph.extraction_score import normalise
from rb.experiments.graph.linker import build_registry, link, linker
from rb.experiments.graph.retriever import GraphRetriever

# One canonical, several aliases, one of which also names another pool document.
REDIRECTS = {
    "Cleveland State University": ["Cleveland State", "Cleveland St.", "Fenn College"],
    "Barack Obama": ["Obama", "President Obama"],
}
POOL = {"Cleveland State University", "Barack Obama", "Fenn College", "Michelle Obama"}


def test_empty_registry_is_exactly_normalise():
    """The string arm and the typed arm must be one code path, or the comparison is two programs."""
    for text in ["U.S. Grant!", "  spaced   out ", "Cleveland State", "", "Ünïcode"]:
        assert link(text, {}) == normalise(text)


def test_alias_resolves_to_its_canonical():
    registry, _ = build_registry(REDIRECTS, POOL)
    assert link("Cleveland State", registry) == normalise("Cleveland State University")
    assert link("President Obama", registry) == normalise("Barack Obama")


def test_r5_drops_an_alias_that_is_itself_a_pool_document():
    """Merging it away would erase a document the graph is expected to retrieve."""
    registry, counts = build_registry(REDIRECTS, POOL)
    assert normalise("Fenn College") not in registry
    assert counts["dropped_self_title"] == 1


def test_r6_drops_an_alias_with_two_canonicals():
    """The precision failure NB-23 names in advance: a shared name inventing a path."""
    ambiguous = dict(REDIRECTS)
    ambiguous["Cleveland, Ohio"] = ["Cleveland State"]
    registry, counts = build_registry(ambiguous, POOL | {"Cleveland, Ohio"})
    assert normalise("Cleveland State") not in registry
    assert counts["dropped_ambiguous"] == 1


def test_identity_mappings_are_not_counted_as_merges():
    registry, counts = build_registry({"Aristotle": ["aristotle", "ARISTOTLE"]}, {"Aristotle"})
    assert registry == {}
    assert counts["dropped_identity"] == 1  # both aliases normalise to one key
    assert counts["kept"] == 0


def test_unknown_entity_still_gets_a_node():
    """An entity with no alias is still an entity. A miss must not drop it."""
    registry, _ = build_registry(REDIRECTS, POOL)
    assert link("Some Unrelated Thing", registry) == normalise("Some Unrelated Thing")


# --- the seam -----------------------------------------------------------------

DOCS = {"d1": ["Cleveland State"], "d2": ["Cleveland State University"]}


def test_build_without_a_linker_keeps_the_two_forms_apart():
    """003 and 004's behaviour, which must not move."""
    nodes, _, _ = kg.build(DOCS)
    assert len(nodes) == 2


def test_build_with_a_linker_merges_them():
    registry, _ = build_registry(REDIRECTS, POOL)
    nodes, _, incidence = kg.build(DOCS, link=linker(registry))
    assert len(nodes) == 1, "the linker did not fire; a no-op linker reproduces 004 exactly"
    assert incidence.shape == (2, 1)  # both documents now share the node


def test_retriever_builds_and_seeds_under_the_same_identity():
    """
    The failure this guards: a graph keyed by canonical identity, seeded by raw string. Every
    seed misses, the arm collapses, and it looks like graphs not working.
    """
    registry, _ = build_registry(REDIRECTS, POOL)
    r = GraphRetriever(
        extract_docs=lambda corpus: {d: [("Cleveland State University", "ORG")] for d in corpus},
        extract_query=lambda q: [("Cleveland State", "ORG")],
        link=linker(registry),
    )
    r.fit({"d1": "text", "d2": "text"})
    assert r._seed("who plays at Cleveland State").sum() > 0, "query did not link into the graph"


def test_retriever_default_is_unchanged():
    """An unqualified GraphRetriever is still the published arm."""
    r = GraphRetriever(
        extract_docs=lambda corpus: {d: [("Cleveland State University", "ORG")] for d in corpus},
        extract_query=lambda q: [("Cleveland State", "ORG")],
    )
    r.fit({"d1": "text", "d2": "text"})
    assert r._seed("who plays at Cleveland State").sum() == 0, (
        "exact-string identity must NOT link these two forms; if it does, the 005 comparison "
        "has no baseline to move away from"
    )


def test_no_alias_ever_resolves_to_an_empty_key():
    """
    An alias whose canonical normalises to nothing would map real entities onto an empty node,
    silently pooling unrelated documents. Punctuation-only titles exist on Wikipedia.
    """
    registry, _ = build_registry({"!!!": ["Chk Chk Chk"], "?": ["Question mark band"]},
                                 {"!!!", "?"})
    assert "" not in registry.values()
    assert all(k and v for k, v in registry.items())
