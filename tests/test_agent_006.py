"""
Tests for experiment 006's scoring and registered estimands.

The interaction estimand (P3) is a double difference the CALLER builds; the bootstrap cannot
check it. 005 shipped within one commit of publishing four significant cells as nulls because
of an unchecked key name in exactly this position, so the double difference is tested against
a hand-computed case rather than trusted.
"""

import pytest

from rb.experiments.agent import analysis006 as a
from rb.experiments.agent import arms, score


# --- scoring -------------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold,expected", [
    ("Netherlands", "Netherlands", 1),
    ("the Netherlands", "Netherlands", 1),          # article stripped
    ("The answer is Netherlands", "Netherlands", 1),  # narration prefix stripped
    ("netherlands.", "Netherlands", 1),             # punctuation and case
    ("Belgium", "Netherlands", 0),
    ("", "Netherlands", 0),
])
def test_exact_match(pred, gold, expected):
    assert score.exact_match(pred, gold) == expected


def test_containment_is_length_bounded():
    """The lenient rule must not credit an answer that buries the gold in a paragraph."""
    padded = "Netherlands " + " ".join(["filler"] * 20)
    assert score.exact_match(padded, "Netherlands") == 0


def test_the_three_variants_disagree_where_the_arms_differ():
    """All three ship because they disagree on narration, and the arms narrate by different
    amounts by construction. The verbatim variant must NOT share the narration strip, or it is
    not an independent check -- which is what review found it doing."""
    pred, gold = "The answer is Netherlands", "Netherlands"
    assert score.exact_match(pred, gold) == 1           # primary strips stock openers
    assert score.exact_match_lenient(pred, gold) == 1
    assert score.exact_match_verbatim(pred, gold) == 0  # strictest sees the narration


@pytest.mark.parametrize("pred,gold", [
    ("Iron Man 3", "Iron Man"),                          # a different film
    ("New York City", "New York"),                       # a different entity
    ("North Dakota", "no"),                              # substring accident
    ("Alice Cooper", "Ali"),
    ("between 1962 and 1964", "196"),
    ("the answer is not France, it is Germany", "France"),  # an explicit negation
    ("I do not know, possibly Boston", "Boston"),        # a hedge
])
def test_containment_counterexamples_do_not_score(pred, gold):
    """Seven cases review found scoring CORRECT under the original containment rule. Every one
    needs the prediction to be longer than the gold, so the rule inflated whichever arms
    narrate most -- which are both arms in the primary contrast."""
    assert score.exact_match(pred, gold) == 0


def test_abstention_is_not_a_wrong_answer():
    assert score.is_abstention("UNKNOWN")
    assert not score.is_abstention("Netherlands")


# --- arm construction ----------------------------------------------------------------------

def test_every_arm_shares_the_answer_rule():
    """The strawman guard: arms may differ in context and tools, never in encouragement."""
    assert arms.ANSWER_RULE in arms.SYSTEM
    assert arms.ANSWER_RULE in arms.SYSTEM_GREP


def test_budget_never_splits_a_block_and_always_yields_one():
    big = "x" * 100_000
    inj = arms.fit_budget([big, "y" * 10], budget_tokens=400)
    assert inj.items == 1 and inj.text == big and inj.truncated


def test_budget_is_shared_by_every_injected_arm():
    docs = [("A", "a" * 800), ("B", "b" * 800), ("C", "c" * 800)]
    _, p = arms.passages("q", docs)
    _, g = arms.graph_facts("q", "a" * 800)
    assert p.tokens_est <= arms.DEFAULT_BUDGET_TOKENS
    assert g.tokens_est <= arms.DEFAULT_BUDGET_TOKENS


# --- the registered estimands --------------------------------------------------------------

def _row(arm, model, qid, em, outcome="ok"):
    return {"arm": arm, "model": model, "query_id": qid, "em": em, "em_strict": em,
            "f1": float(em), "abstained": 0, "outcome": outcome, "num_turns": 1,
            "input_tokens": 10, "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "cost_usd": 0.0, "duration_ms": 1}


def test_interaction_is_a_per_question_double_difference():
    """Hand-computed: on q1 the graph rescues haiku and does nothing for opus (d = +1);
    on q2 it does nothing for either (d = 0). The mean must be +0.5, not +1 or +0.25."""
    rows = [
        _row("graph-facts", "haiku", "q1", 1), _row("grep", "haiku", "q1", 0),
        _row("graph-facts", "opus", "q1", 1), _row("grep", "opus", "q1", 1),
        _row("graph-facts", "haiku", "q2", 1), _row("grep", "haiku", "q2", 1),
        _row("graph-facts", "opus", "q2", 1), _row("grep", "opus", "q2", 1),
    ]
    r = a.interaction(a.index(rows), "graph-facts", "grep", "haiku", "opus", ["q1", "q2"])
    assert r["n"] == 2
    assert r["mean_diff"] == pytest.approx(0.5)


def test_interaction_drops_questions_with_an_incomplete_block():
    """A double difference needs all four cells. A question missing one must not contribute
    a half-computed value; it must leave the sample."""
    rows = [
        _row("graph-facts", "haiku", "q1", 1), _row("grep", "haiku", "q1", 0),
        _row("graph-facts", "opus", "q1", 1), _row("grep", "opus", "q1", 1),
        _row("graph-facts", "haiku", "q2", 1), _row("grep", "haiku", "q2", 0),
        _row("graph-facts", "opus", "q2", 1),  # grep/opus/q2 missing
    ]
    r = a.interaction(a.index(rows), "graph-facts", "grep", "haiku", "opus", ["q1", "q2"])
    assert r["n"] == 1


def test_infrastructure_failures_leave_the_denominator_but_max_turns_does_not():
    """Protocol 006 section 8. A timeout is the harness's fault; running out of turns is the
    model's behaviour and must still be scored."""
    rows = [_row("grep", "haiku", "q1", 0, outcome="timeout"),
            _row("grep", "haiku", "q2", 0, outcome="max_turns"),
            _row("grep", "haiku", "q3", 1)]
    idx = a.index(rows)
    assert not a.usable(idx[("grep", "haiku", "q1")])
    assert a.usable(idx[("grep", "haiku", "q2")])
    t = a.arm_table(idx, ["q1", "q2", "q3"])[0]
    assert t["n"] == 2 and t["excluded"] == 1
    assert t["em"] == pytest.approx(0.5)
    assert t["max_turns_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_mde_matches_the_registered_table():
    """Protocol 006 section 6. Discordant-pair based, sqrt(2) heavier for the interaction --
    NOT a two-proportion formula, which is the estimand error 005-amendment-1 records."""
    assert a.mde(100, 0.30) == pytest.approx(0.1534, abs=1e-4)
    assert a.mde(100, 0.40) == pytest.approx(0.1772, abs=1e-4)
    assert a.mde(100, 0.30, interaction_test=True) == pytest.approx(0.2170, abs=1e-4)
    assert a.mde(100, 0.40, interaction_test=True) == pytest.approx(0.2506, abs=1e-4)


def test_underpowered_is_only_available_where_registered():
    """A null may be relabelled `underpowered` only for the prediction that registered it in
    advance. Otherwise any inconvenient null could be excused after the fact."""
    wide = {"resolved": True, "ci95": [-0.4, 0.4]}
    assert a.decide(wide, 0.2, underpowered_allowed=True) == "underpowered"
    assert a.decide(wide, 0.2, underpowered_allowed=False) == "no_advantage"


def test_discordance_drives_the_applicable_mde():
    assert a.discordance([1, 1, 0, 0], [1, 0, 0, 1]) == pytest.approx(0.5)
    assert a.discordance([1, 1], [1, 1]) == 0.0


def test_mde_survives_zero_discordance():
    """Reachable on a resistant-subset stratum. The original ternary divided by zero."""
    assert a.mde(50, 0.0) == 0.0
    assert a.mde(50, 0.0, interaction_test=True) == 0.0


def test_holm_gates_the_decision():
    """A cell whose interval clears zero but whose adjusted p does not must not read
    `supported`, or the registered family is decorative."""
    clear_but_unadjusted = {"resolved": True, "ci95": [0.05, 0.30], "p_holm": 0.20}
    clear_and_adjusted = {"resolved": True, "ci95": [0.05, 0.30], "p_holm": 0.01}
    assert a.decide(clear_but_unadjusted, 0.1, False) == "no_advantage"
    assert a.decide(clear_and_adjusted, 0.1, False) == "supported"
