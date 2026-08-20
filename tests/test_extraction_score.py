"""
Experiment 003's extraction diagnostic and connectivity gate — protocol §8.2.

Written and committed BEFORE the gold set exists (every sample record still reads
`annotated: false`), so the measurement procedure is fixed before the data it will measure.
"""

import json
from pathlib import Path

import pytest

from rb.experiments.graph import annotate, entity_types
from rb.experiments.graph import extraction_score as es

ROOT = Path(__file__).resolve().parents[1]


# --- the whitelist -----------------------------------------------------------------

def test_whitelist_and_exclusions_partition_spacys_labels():
    entity_types.assert_partition()
    assert len(entity_types.WHITELIST | entity_types.EXCLUDED) == 18
    assert not (entity_types.WHITELIST & entity_types.EXCLUDED)


def test_excluded_types_are_the_numeric_ones():
    """Grounded in data, not taste: 2 of 13,783 gold titles are purely numeric."""
    assert entity_types.EXCLUDED == {
        "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}


def test_filter_is_applied_symmetrically_so_a_correct_date_is_not_a_false_positive():
    """The single easiest way to make this number wrong. Filtering only the gold side turns
    every correctly-extracted DATE into a false positive and collapses precision."""
    gold = {"d1": ["Missouri"]}
    pred = {"d1": [("Missouri", "GPE"), ("2010", "DATE"), ("28,781", "CARDINAL")]}
    assert es.score(gold, pred)["micro"]["precision"] == 1.0


# --- normalisation and set semantics ------------------------------------------------

def test_normalise_folds_case_and_punctuation_only():
    assert es.normalise("Marion County, Missouri") == "marion county missouri"
    assert es.normalise("U.S.") == "u s"


def test_normalisation_does_not_canonicalise_abbreviations():
    """"U.S." and "United States" must stay DIFFERENT nodes, because the graph will treat
    them as different nodes. The diagnostic measures what the graph will do."""
    assert es.normalise("U.S.") != es.normalise("United States")


def test_repeated_mentions_collapse_to_one():
    """A set, matching how the graph deduplicates. This is also what removes the prepended
    title echo — median 2 occurrences per passage — from the arithmetic."""
    c = es.score_passage(["Missouri"], ["Missouri", "Missouri", "missouri"])
    assert c == {"tp": 1, "fp": 0, "fn": 0, "gold": 1, "pred": 1}


def test_partial_string_is_a_miss_in_both_directions():
    """Documented consequence of exact-string matching: gold "Paris" vs predicted
    "Paris, France" is one false negative AND one false positive, because the graph would
    key them as two different nodes."""
    c = es.score_passage(["Paris"], ["Paris, France"])
    assert c["tp"] == 0 and c["fp"] == 1 and c["fn"] == 1


def test_empty_annotation_is_not_a_crash():
    assert es.score_passage([], [])["gold"] == 0
    assert es.micro([es.score_passage([], [])])["precision"] == 0.0


# --- micro averaging -----------------------------------------------------------------

def test_micro_pools_counts_rather_than_averaging_per_passage_rates():
    counts = [{"tp": 1, "fp": 0, "fn": 0, "gold": 1, "pred": 1},
              {"tp": 1, "fp": 3, "fn": 0, "gold": 1, "pred": 4}]
    assert es.micro(counts)["precision"] == round(2 / 5, 4)


# --- the bootstrap -------------------------------------------------------------------

def _spread_counts():
    return [{"tp": i % 3, "fp": (i + 1) % 2, "fn": i % 2, "gold": 2, "pred": 2} for i in range(40)]


def test_bootstrap_resamples_passages_not_entities():
    """The sample was drawn as 100 passages and entities inside one share extraction
    outcomes; resampling entities would report an interval two to three times too narrow."""
    r = es.bootstrap_ci(_spread_counts(), "precision", b=500)
    assert r["unit"] == "passage"
    assert r["ci95"][0] <= r["point"] <= r["ci95"][1]
    # Non-zero WIDTH, not merely a bracket. Mutation testing caught that a bootstrap which
    # resamples nothing returns [point, point], and a containment-only assertion passes it.
    assert r["ci95"][1] - r["ci95"][0] > 0.01, "a degenerate bootstrap reports a zero-width CI"


def test_bootstrap_is_deterministic_under_its_seed():
    a = es.bootstrap_ci(_spread_counts(), "precision", b=300)
    assert a == es.bootstrap_ci(_spread_counts(), "precision", b=300)
    b = es.bootstrap_ci(_spread_counts(), "precision", b=300, seed=1)
    assert b["ci95"] != a["ci95"] or b["point"] == a["point"]


# --- the connectivity gate ------------------------------------------------------------

def test_connectivity_counts_entities_shared_across_documents():
    r = es.graph_connectivity({"d1": ["Paris", "Lyon"], "d2": ["Paris"], "d3": []})
    assert r["distinct_entities"] == 2
    assert r["entities_in_2plus_documents"] == 1
    assert r["documents_with_an_entity"] == 2


def test_gold_pairs_that_share_entities_beat_random_pairs():
    """Real bridging structure: each gold pair shares a private entity, random pairs do not."""
    docs, pairs = {}, []
    for i in range(30):
        docs[f"a{i}"] = [f"Bridge{i}", f"X{i}"]
        docs[f"b{i}"] = [f"Bridge{i}", f"Y{i}"]
        pairs.append((f"a{i}", f"b{i}"))
    assert es.bridge_reachability(docs, pairs, b=200)["passed"]


def test_gold_pairs_with_no_shared_entity_fail_the_gate():
    """If the two gold documents never share an entity, no propagation reaches the second
    one, and the arm cannot work for reasons unrelated to extraction accuracy."""
    docs = {f"a{i}": [f"X{i}"] for i in range(30)} | {f"b{i}": [f"Y{i}"] for i in range(30)}
    pairs = [(f"a{i}", f"b{i}") for i in range(30)]
    r = es.bridge_reachability(docs, pairs, b=200)
    assert r["observed_share_rate"] == 0.0 and not r["passed"]


def test_a_hub_shared_by_everything_does_not_pass_as_bridging_signal():
    """The case that killed the first design. One entity in every document makes every pair
    connected, gold or not, so the gold rate cannot exceed the random rate and the gate
    correctly refuses to call it structure."""
    docs = {f"d{i}": ["Hub", f"U{i}"] for i in range(30)}
    pairs = [(f"d{i}", f"d{i+1}") for i in range(0, 28, 2)]
    assert not es.bridge_reachability(docs, pairs, b=200)["passed"]


# --- the annotation tool --------------------------------------------------------------

def test_parse_entities_dedupes_and_preserves_first_appearance_order():
    assert annotate.parse_entities("Marion County, Missouri, Palmyra, Marion County") == [
        "Marion County", "Missouri", "Palmyra"]


def test_parse_entities_preserves_surface_form_exactly():
    """Rule card item 9: no case folding at annotation time. Normalisation happens once, at
    scoring time, under a rule fixed in code."""
    assert annotate.parse_entities("U.S., iPhone") == ["U.S.", "iPhone"]


def test_parse_entities_on_a_blank_line_is_an_empty_set_not_a_skip():
    assert annotate.parse_entities("   ") == []


def test_record_stamps_the_rule_card_version(tmp_path):
    """If the card is revised mid-session the cutover must be visible per passage rather
    than reconstructed afterwards."""
    row = annotate.record({"doc_id": "d1"}, ["Missouri"])
    assert row["annotated"] is True
    assert row["rule_card"] == annotate.RULE_CARD_VERSION
    assert row["annotated_at"].endswith("+00:00")


def test_save_is_atomic_and_round_trips(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [{"doc_id": "d1", "entities": ["A"], "annotated": True}]
    annotate.save_atomic(rows, p)
    assert annotate.load(p) == rows
    assert not list(tmp_path.glob("*.tmp")), "the temp file must not survive a successful write"


def test_next_index_resumes_at_the_first_unannotated_record():
    rows = [{"annotated": True}, {"annotated": False}, {"annotated": False}]
    assert annotate.next_index(rows) == 1
    assert annotate.next_index([{"annotated": True}]) is None


# --- the sample itself ----------------------------------------------------------------

def _sample_rows():
    path = ROOT / "results" / "003" / "extraction-sample.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_the_reference_set_is_complete_and_carries_its_provenance():
    """
    Replaces the pre-annotation guard, in the same commit that landed the annotations as
    that guard's docstring required.

    The scorer, the whitelist and the tool were all committed while every record was still
    empty — checkable in the history at 732be11. What is asserted now is the property that
    matters going forward: the reference set is complete, and every record says who
    produced it, because it is MODEL-annotated rather than hand-annotated and a reader must
    see that from the artifact rather than only from the protocol.
    """
    rows = _sample_rows()
    assert len(rows) == 100
    assert all(r["annotated"] is True for r in rows), "a partial reference set understates recall"
    assert all(r["rule_card"] == "v1" for r in rows)
    assert all(r["annotator"] == "llm-panel-3x-majority" for r in rows), (
        "provenance must travel with the record, not only with the protocol"
    )


def test_the_reference_set_is_not_silently_empty():
    """Three passages are legitimately empty (Aldosterone, Line of battle, Sacral nerve
    stimulator are concept articles with no named entities). Many more would mean the
    annotation failed rather than that the passages had nothing in them."""
    rows = _sample_rows()
    empty = [r for r in rows if not r["entities"]]
    assert len(empty) <= 5, f"{len(empty)} empty passages suggests a failed annotation pass"
    assert sum(len(r["entities"]) for r in rows) > 500


def test_agreement_artifact_matches_the_reference_set():
    """The published agreement number must describe the file it ships beside."""
    agreement = json.loads((ROOT / "results" / "003" / "annotation-agreement.json").read_text())
    rows = _sample_rows()
    assert agreement["passages"] == len(rows)
    assert agreement["entities_kept"] == sum(len(r["entities"]) for r in rows)
    assert agreement["raters"] == 3
