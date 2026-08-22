"""
NB-26 D1 — the section 9 pool control must be able to FAIL.

Every test here goes through `pool.construction_counts`, the real measurement path, rather than
through `controls.pool_construction` with hand-written literals. That distinction is the whole
point: the comparator was always correct given unequal inputs, so a test that feeds it a
contrived mismatch passes under both the defective and the fixed code and proves nothing. What
was broken was the PRODUCER. These tests exercise the producer.
"""

import pytest

from rb import controls
from rb.experiments.graph import pool


def _counts(corpus_titles, context, qrels):
    return pool.construction_counts(corpus_titles, context, qrels)


def test_gold_titles_matched_counts_queries_whose_gold_docs_are_all_in_the_pool():
    """KILLS: `gold_titles_matched=len(qrels)`.

    Under the defective code this field was `len(qrels)` regardless of the data, so it would
    report 1 here instead of 0 and the control would pass on a pool missing a gold document.
    """
    corpus_titles = {"d1": "A", "d2": "B", "d3": "C"}
    context = {"q1": ["A", "B"]}          # only A and B are pooled; C is not
    qrels = {"q1": {"d1": 1, "d3": 1}}    # q1's gold set reaches OUTSIDE the pool

    counts = _counts(corpus_titles, context, qrels)
    assert counts["gold_titles_matched"] == 0, "d3 is not in the pool, so q1 is not matched"
    assert counts["gold_queries"] == 1

    check = controls.pool_construction(
        questions=counts["questions"], passages=2, title_slots=counts["title_slots"],
        unresolved=counts["unresolved"], collisions=counts["collisions"],
        gold_titles_matched=counts["gold_titles_matched"], gold_queries=counts["gold_queries"])
    assert check["passed"] is False, "a pool missing a gold document must fail the control"
    assert "gold_titles_matched" in check["mismatched"]


def test_gold_titles_matched_passes_when_every_gold_document_is_pooled():
    """The positive case, so the test above is not passing for an unrelated reason."""
    counts = _counts({"d1": "A", "d2": "B"}, {"q1": ["A", "B"]}, {"q1": {"d1": 1, "d2": 1}})
    assert counts["gold_titles_matched"] == counts["gold_queries"] == 1


def test_unresolved_titles_are_measured_not_assumed_zero():
    """KILLS: `unresolved=0`.

    A pooled title absent from the corpus must be COUNTED. `pool.build` raises on this, which is
    exactly why the count was previously unobservable — the raise pre-empted any measurement, so
    relocating the literal would not have helped.
    """
    counts = _counts({"d1": "A"}, {"q1": ["A", "Z"]}, {})
    assert counts["unresolved"] == 1, "Z is pooled but absent from the corpus"
    with pytest.raises(RuntimeError, match="not a subset"):
        pool.build({"d1": "text"}, {"d1": "A"}, {"q1": ["A", "Z"]})


def test_title_collisions_are_measured_not_assumed_zero():
    """KILLS: `collisions=0`. Two documents sharing a pooled title must be counted, and
    `title_index` must still raise — the measurement does not replace the enforcement."""
    corpus_titles = {"d1": "A", "d2": "A"}
    counts = _counts(corpus_titles, {"q1": ["A"]}, {})
    assert counts["collisions"] == 1
    with pytest.raises(RuntimeError, match="duplicate titles"):
        pool.title_index(corpus_titles)


def test_counts_are_measured_before_build_would_raise():
    """The ordering that makes the control meaningful.

    `construction_counts` must not raise on data that `build` rejects, or it could never report a
    nonzero count and would be measuring its own control flow rather than the pool.
    """
    counts = _counts({"d1": "A"}, {"q1": ["A", "MISSING"]}, {})
    assert counts["unresolved"] == 1  # returned, not raised


def test_counts_are_restricted_to_pooled_titles():
    """KILLS: dropping the `if t in titles` filter in construction_counts.

    The control must describe the POOL, not the whole corpus. `build()` filters the corpus to
    pooled titles before indexing; `construction_counts` repeats that filter, and the two are kept
    in step by nothing but vigilance. Unfiltered, this would index all 5,233,329 BEIR titles rather
    than the 73,700 pooled slots, and report collisions from documents the pool never contains —
    failing a control the pool did not violate.

    Here "Shared" is duplicated in the corpus but is NOT pooled, so it must not be counted.
    """
    corpus_titles = {"d1": "A", "d2": "B", "d3": "Shared", "d4": "Shared"}
    counts = _counts(corpus_titles, {"q1": ["A", "B"]}, {})
    assert counts["collisions"] == 0, "a duplicate outside the pool is not a pool collision"


def test_passages_is_measured_from_the_resolved_documents():
    """KILLS: hardcoding `passages`.

    `passages` is what lets the control be evaluated BEFORE `build()` — it is the size of the
    resolved document set, which construction_counts already knows. Hardcoding it would reinstate
    the original defect (a published field that is a literal) on the one field whose measurement
    makes the ordering possible.
    """
    counts = _counts({"d1": "A", "d2": "B", "d3": "C"}, {"q1": ["A", "B"]}, {})
    assert counts["passages"] == 2, "only A and B are pooled; C is not a pool passage"

    counts3 = _counts({"d1": "A", "d2": "B", "d3": "C"}, {"q1": ["A", "B", "C"]}, {})
    assert counts3["passages"] == 3, "the count must track the pool, not a constant"
