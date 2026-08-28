"""
Stage 0's coverage measure — protocols/005-identity.md section 6 as amended by amendment 1.

The test that earns its place here is the one that distinguishes the corrected definition of
"alias-affected" from the narrow one it replaced. The narrow reading gave 49 affected queries on
2Wiki where the corrected reading gives 1,085, and reporting the narrow figure would have
concluded the mechanism was untestable on a measurement artefact.
"""

import json

import pytest

from rb.experiments.graph import identity_coverage as ic

# The query names the CANONICAL. A document names an ALIAS. No query entity is a registry key,
# so the narrow definition sees nothing — but the merge is exactly what makes d2 reachable.
DOCS = {"d1": ["Cleveland State University"], "d2": ["Cleveland State"]}
QENTS = {"q1": ["Cleveland State University"]}
REGISTRY = {"cleveland state": "cleveland state university"}


@pytest.fixture
def measured(monkeypatch):
    monkeypatch.setattr(ic, "_load", lambda corpus: ({}, {"q1": "text"}, set()))
    monkeypatch.setattr(ic, "_entities", lambda *a: (DOCS, QENTS))
    return ic.measure("hotpotqa", "spacy", REGISTRY, {"kept": 1})


def test_a_query_naming_the_canonical_is_still_alias_affected(measured):
    """Amendment 1's whole point. The narrow definition returns 0 here."""
    assert measured["queries"]["alias_affected"] == 1
    assert measured["queries"]["alias_affected_narrow"] == 0


def test_the_two_documents_merge_into_one_node(measured):
    assert measured["nodes"]["before"] == 2
    assert measured["nodes"]["after"] == 1
    assert measured["nodes"]["reduction"] == 1


def test_an_untouched_query_is_not_affected(monkeypatch):
    monkeypatch.setattr(ic, "_load", lambda corpus: ({}, {"q1": "text"}, set()))
    monkeypatch.setattr(ic, "_entities", lambda *a: (DOCS, {"q1": ["Something Else"]}))
    m = ic.measure("hotpotqa", "spacy", REGISTRY, {})
    assert m["queries"]["alias_affected"] == 0


def test_an_empty_registry_affects_nothing(monkeypatch):
    """The string arm must measure as zero coverage, or the comparison has no origin."""
    monkeypatch.setattr(ic, "_load", lambda corpus: ({}, {"q1": "text"}, set()))
    monkeypatch.setattr(ic, "_entities", lambda *a: (DOCS, QENTS))
    m = ic.measure("hotpotqa", "spacy", {}, {})
    assert m["queries"]["alias_affected"] == 0
    assert m["nodes"]["reduction"] == 0
    assert m["mde_at_80_power_on_affected"] is None


def test_combine_refuses_a_partial_set(monkeypatch, tmp_path):
    """A coverage figure reported as if it were complete is the failure this refuses.

    Both manifests are present, so the only thing missing is a cell — otherwise the manifest
    read would raise first and the test would pass without exercising the guard.
    """
    monkeypatch.setattr(ic, "OUT", tmp_path)
    for corpus in ic.CORPORA:
        (tmp_path / f"redirects-{corpus}-manifest.json").write_text(json.dumps(
            {k: 0 for k in ("fetched_utc", "sha256", "titles_requested",
                            "titles_with_redirects", "aliases_total", "failed_batches")}))
    (tmp_path / "identity-coverage-hotpotqa-spacy.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="all four cells or not at all"):
        ic.combine()


def test_committed_artifact_shares_recompute():
    """Every share in the published artifact must follow from the counts beside it."""
    d = json.loads((ic.OUT / "identity-coverage.json").read_text())
    for corpus in d["corpora"].values():
        for arm in corpus["arms"].values():
            q, n = arm["queries"], arm["nodes"]
            assert q["share_alias_affected"] == round(q["alias_affected"] / q["n"], 4)
            assert n["share_resolving"] == round(n["resolving_through_alias"] / n["before"], 4)
            assert n["reduction"] == n["before"] - n["after"]
