"""
The pinned spaCy extractor — protocols/003-graph-arm.md §2 and §8.

Skipped when spaCy is absent, the same convention the dense rung uses for its encoder: the
rest of the suite must not require a model download.
"""

import pytest

from rb.experiments.graph import entity_types

spacy = pytest.importorskip("spacy", reason="extractor tests need the pinned spaCy")

from rb.experiments.graph import extractor  # noqa: E402


MARION = ("Marion County, Missouri Marion County is a county in the U.S. state of Missouri. "
          "As of the 2010 census, the population was 28,781. Its county seat is Palmyra.")


def test_the_pin_is_enforced_rather_than_assumed():
    """A drifted environment must fail loudly here rather than quietly produce different
    entities — the same reason the dataset manifest refuses a changed hash."""
    assert spacy.__version__ == extractor.SPACY_VERSION
    assert extractor._nlp().meta["version"] == extractor.MODEL_VERSION


def test_manifest_records_what_separates_an_environment_from_a_finding():
    m = extractor.manifest()
    assert m["spacy_version"] == extractor.SPACY_VERSION
    assert m["model_version"] == extractor.MODEL_VERSION
    assert m["whitelist"] == sorted(entity_types.WHITELIST)


def test_extract_returns_labels_unfiltered():
    """Unfiltered on purpose: the whitelist is applied in ONE place, to both sides. A DATE
    must survive this call so that filtering stays symmetric downstream."""
    labels = {label for _, label in extractor.extract(MARION)}
    assert "DATE" in labels or "CARDINAL" in labels
    assert "GPE" in labels


def test_node_strings_drops_excluded_types_and_dedupes():
    nodes = extractor.node_strings(extractor.extract(MARION))
    assert "2010" not in nodes and "28,781" not in nodes
    assert "Missouri" in nodes
    assert len(nodes) == len(set(nodes))


def test_node_strings_keeps_a_correct_span_under_a_wrong_type():
    """spaCy labels "Palmyra" PERSON here, which is wrong — it is a town. The span is right,
    and the graph keys nodes by string without consulting the type, so this counts. That is
    why §8.2's scoring is type-blind after whitelist filtering."""
    assert "Palmyra" in extractor.node_strings(extractor.extract(MARION))


def test_extract_many_cannot_misalign_documents_and_entities():
    """The failure that would attach each document's entities to its neighbour and still look
    entirely plausible."""
    texts = {f"d{i}": f"{city} is a city." for i, city in
             enumerate(["Paris", "Berlin", "Madrid", "Lisbon", "Vienna"])}
    got = extractor.extract_many(texts)
    assert set(got) == set(texts)
    for doc_id, ents in got.items():
        city = texts[doc_id].split()[0]
        assert any(city == t for t, _ in ents), f"{doc_id} got another document's entities"


def test_extraction_is_deterministic():
    assert extractor.extract(MARION) == extractor.extract(MARION)


def test_empty_text_yields_no_entities():
    assert extractor.extract("") == []
    assert extractor.node_strings([]) == []
