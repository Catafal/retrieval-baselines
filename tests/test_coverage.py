"""
Experiment 003's bridge-entity class — protocols/003-graph-arm.md section 4.

This is the most heavily tested unit in the experiment, because the protocol names the
class definition as the single largest threat to the entry: defined by judgement after
seeing results, the whole thing is unfalsifiable. Every test here is checked against a
deliberately broken implementation (see the mutation log in the run notes) — 002 shipped
a test it cited as covering encoder transposition that could not actually fail, and the
lesson generalises.
"""

import pytest

from rb import controls
from rb.experiments.graph import coverage as cov


# --- normalisation -----------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("Kiss and Tell (1945 film)", "Kiss and Tell"),
    ("Shirley Temple", "Shirley Temple"),
    ("Sunday Bloody Sunday (song)", "Sunday Bloody Sunday"),
    ("Alpha (beta) Gamma", "Alpha (beta) Gamma"),   # not trailing: part of the name
    ("Nested (a (b) c)", "Nested (a (b) c)"),       # unmatched: left alone, see below
    ("Trailing (x) (y)", "Trailing (x)"),           # one only, the outermost-last
])
def test_strip_disambiguator_only_removes_a_trailing_parenthetical(title, expected):
    """
    A nested trailing group is left UNCHANGED rather than partly removed. `[^)]*` cannot
    span an inner ")", so the pattern simply fails to match and the title survives whole.
    That is the safe direction: half-stripping "Nested (a (b) c)" to "Nested (a (b" would
    invent an entity name that never existed, and a title we failed to strip merely lands
    in the sensitivity definition's behaviour, which is registered and reported anyway.
    """
    assert cov.strip_disambiguator(title) == expected


def test_normalise_matches_the_scorer_tokenizer():
    """002's convention: properties use the tokenizer the lexical rung scores with, so
    the class describes the vocabulary the arms actually saw."""
    from rb.experiments.ladder.retrievers.lexical import _tokenize
    text = "Marion County, Missouri -- population 28,781!"
    assert cov.normalise(text) == _tokenize(text)


# --- the contiguity rule -----------------------------------------------------------

def test_title_must_appear_as_a_contiguous_run():
    """Scattered words are not the entity being named. A bag-of-words test would count
    almost any multi-word title against a long question."""
    q = cov.normalise("Chris said that Evans arrived")
    assert not cov.title_in_query("Chris Evans", q)
    assert cov.title_in_query("Chris Evans", cov.normalise("Who is Chris Evans anyway"))


def test_match_is_case_and_punctuation_insensitive():
    q = cov.normalise("who directed KISS AND TELL, really?")
    assert cov.title_in_query("Kiss and Tell (1945 film)", q)


def test_title_matches_at_the_very_start_and_very_end():
    """Off-by-one in the sliding window would drop boundary matches silently."""
    assert cov.title_in_query("Alpha Beta", cov.normalise("Alpha Beta is a thing"))
    assert cov.title_in_query("a thing", cov.normalise("Alpha Beta is a thing"))


def test_empty_or_whitespace_title_never_counts():
    """An untitled document cannot be 'named' in a query."""
    q = cov.normalise("anything at all")
    assert not cov.title_in_query("", q)
    assert not cov.title_in_query("   ", q)
    assert not cov.title_in_query("(1945 film)", q), "a title that is only a disambiguator"


def test_title_longer_than_the_query_cannot_match():
    assert not cov.title_in_query("A Very Long Entity Name Indeed", cov.normalise("short"))


def test_unknown_definition_is_refused():
    with pytest.raises(ValueError, match="unknown definition"):
        cov.title_in_query("Alpha", cov.normalise("Alpha"), definition="whatever")


# --- the two registered definitions ------------------------------------------------

def test_the_two_definitions_disagree_exactly_where_the_protocol_says():
    """The Kiss and Tell case: the entity IS named, but the title carries a Wikipedia
    disambiguator. This single behaviour moves 15.4% of the corpus."""
    q = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    golds = ["Kiss and Tell (1945 film)", "Shirley Temple"]
    assert cov.coverage(q, golds, cov.PRIMARY) == 1
    assert cov.coverage(q, golds, cov.SENSITIVITY) == 0


def test_coverage_counts_each_gold_title_once():
    q = "Were Scott Derrickson and Ed Wood of the same nationality?"
    assert cov.coverage(q, ["Ed Wood", "Scott Derrickson"], cov.PRIMARY) == 2
    assert cov.coverage(q, ["Nobody", "Nowhere"], cov.PRIMARY) == 0


def test_coverage_all_skips_queries_without_text_but_keeps_order_stable():
    queries = {"q1": "about Alpha", "q2": "about Beta"}
    golds = {"q1": ["Alpha"], "q2": ["Beta"], "q3": ["Gamma"]}
    assert cov.coverage_all(queries, golds) == {"q1": 1, "q2": 1}


def test_distribution_reports_empty_bins_as_zero():
    """A missing class must be visible as a zero, not as an absent key."""
    assert cov.distribution({"a": 1, "b": 1}) == {0: 0, 1: 2, 2: 0}


def test_reclassified_counts_only_disagreements():
    """
    The fixture is deliberately ASYMMETRIC — three queries, one disagreement. A
    two-query fixture with one agreement and one disagreement returns 1 whether the
    implementation counts matches or mismatches, and mutation testing showed the earlier
    version of this test could not tell those apart.
    """
    primary = {"a": 1, "b": 2, "c": 0}
    sensitivity = {"a": 0, "b": 2, "c": 0}
    assert cov.reclassified(primary, sensitivity) == 1


def test_reclassified_ignores_queries_missing_from_either_side():
    assert cov.reclassified({"a": 1, "b": 1}, {"a": 2}) == 1


def test_agreement_with_type_maps_coverage_le_1_to_bridge():
    """
    The fixture deliberately includes a comparison question at coverage 1 ("e"). Without
    it the mapping `comparison and coverage == 2` is indistinguishable from
    `comparison and coverage >= 1`, and mutation testing showed the earlier version of
    this test could not fail against that change. Two such queries exist in the real
    data, so the case is real rather than invented to satisfy a mutant.
    """
    classes = {"a": 0, "b": 1, "c": 2, "d": 2, "e": 1}
    types = {"a": "bridge", "b": "bridge", "c": "comparison", "d": "bridge",
             "e": "comparison"}
    r = cov.agreement_with_type(classes, types)
    assert r["agreed"] == 3, "a comparison question at coverage 1 is a DISagreement"
    assert r["queries"] == 5
    assert r["table"]["bridge|2"] == 1, "the type-bridge/coverage-2 case is kept visible"
    assert r["table"]["comparison|1"] == 1


# --- the control -------------------------------------------------------------------

def _ctl(**over):
    args = dict(measured=dict(cov.FROZEN_DISTRIBUTION[cov.PRIMARY]),
                definition=cov.PRIMARY, reclassified=cov.FROZEN_RECLASSIFIED)
    args.update(over)
    return controls.coverage_distribution(**args)


def test_control_passes_on_the_frozen_distribution():
    assert _ctl()["passed"]


def test_control_fails_when_a_single_query_moves_between_bins():
    moved = dict(cov.FROZEN_DISTRIBUTION[cov.PRIMARY])
    moved[0] += 1
    moved[1] -= 1
    assert not _ctl(measured=moved)["passed"], "totals still 7,405 but the split changed"


def test_control_fails_when_the_definitions_stop_disagreeing():
    """If the two definitions suddenly agree, the normalisation changed under us and
    15.4% of queries moved silently."""
    assert not _ctl(reclassified=0)["passed"]


def test_control_checks_the_sensitivity_definition_separately():
    assert _ctl(measured=dict(cov.FROZEN_DISTRIBUTION[cov.SENSITIVITY]),
                definition=cov.SENSITIVITY)["passed"]
    assert not _ctl(measured=dict(cov.FROZEN_DISTRIBUTION[cov.PRIMARY]),
                    definition=cov.SENSITIVITY)["passed"]
