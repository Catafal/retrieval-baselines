"""
The closure control's tolerance must be able to fail, and must not fire on correct code.

A tolerance is only meaningful relative to the distances it gates. At 0.10 this control sat
4x to 22x above every delta it was measuring, so no realistic defect could trip it — it was a
gate the size of the room. These tests pin the tolerance from BOTH sides so it cannot drift
back to decorative, and so a future corpus whose legitimate delta exceeds the gate fails
loudly rather than being absorbed.

The measurements the number comes from are in bm25_closure's docstring. The two classes the
control provably cannot see are asserted here too, so nobody re-tightens the tolerance hoping
to catch them and instead makes the control flaky on correct code.
"""
import pytest

from rb.controls import BM25_CLOSURE_TOLERANCE, bm25_closure

# Measured against the committed scorer, 2026-08-23. These are the real gaps between this
# repository's BM25 and Thakur et al. 2021's published Anserini figures.
LEGITIMATE_DELTAS = {"scifact": 0.0045, "quora": 0.0212, "hotpotqa": 0.0272}

# Measured the same way: what a wrong BM25 formula actually costs in nDCG@10.
WRONG_FORMULA_DELTAS = {"idf_off": 0.1085, "tf_saturation_off": 0.1075}


def test_the_tolerance_passes_every_corpus_we_actually_ship():
    """The control must not fire on correct code, or it gets loosened until it means nothing."""
    for corpus, delta in LEGITIMATE_DELTAS.items():
        out = bm25_closure(our_ndcg=0.5, published_ndcg=0.5 + delta)
        assert out["passed"], f"{corpus}: tolerance fires on a correct implementation"


def test_the_tolerance_catches_a_wrong_bm25_formula():
    """KILLS: a tolerance so loose that a scorer which is not BM25 still passes."""
    for defect, delta in WRONG_FORMULA_DELTAS.items():
        out = bm25_closure(our_ndcg=0.5, published_ndcg=0.5 + delta)
        assert not out["passed"], f"{defect}: a wrong formula passes the closure gate"


def test_the_gate_keeps_real_margin_on_both_sides():
    """
    The assertions above hold at 0.10 too — 0.1085 clears it, but by 1.08x, which is not a
    margin, it is luck. This pins the headroom itself so the gate cannot creep back toward the
    largest defect it is supposed to catch.
    """
    worst_legitimate = max(LEGITIMATE_DELTAS.values())
    smallest_defect = min(WRONG_FORMULA_DELTAS.values())

    assert BM25_CLOSURE_TOLERANCE >= worst_legitimate * 1.5, (
        "tolerance is too close to a legitimate difference; it will false-fail"
    )
    assert BM25_CLOSURE_TOLERANCE <= smallest_defect / 2, (
        "tolerance is too close to a real defect; it barely catches what it exists to catch"
    )


def test_the_control_records_what_it_cannot_detect():
    """
    A control that does not state its limits invites the reader to assume it has none.

    k1/b drift tops out at 0.0211 and length-norm-off at 0.0148, both under the largest
    legitimate difference of 0.0272 — so no tolerance separates them and no tightening ever
    will. That is a property of the measurement, not of the number chosen.
    """
    out = bm25_closure(our_ndcg=0.5, published_ndcg=0.5)

    assert "cannot_detect" in out, "the artifact must carry the control's blind spots"
    assert any("k1" in c for c in out["cannot_detect"])

    parameter_drift, length_norm_off = 0.0211, 0.0148
    worst_legitimate = max(LEGITIMATE_DELTAS.values())
    assert parameter_drift < worst_legitimate and length_norm_off < worst_legitimate, (
        "if these ever exceed the legitimate spread, the control COULD gate them and "
        "this test should be replaced by one that requires it to"
    )
