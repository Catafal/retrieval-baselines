"""
Experiment 003's pool construction — protocols/003-graph-arm.md sections 3 and 9.

The pool's identity claim (an exactly-identified subset of BEIR's hotpotqa corpus, same
document ids, zero title ambiguity) is what every 003 number rests on. These tests run
against tiny in-memory fixtures rather than the 5.2M-document corpus, so a construction
bug surfaces in seconds instead of hours into a run.
"""

import pytest

from rb import controls
from rb.experiments.graph import pool


def _context():
    """Two questions sharing one distractor: 6 slots, 5 unique titles."""
    return {
        "q1": ["Alpha", "Beta", "Shared"],
        "q2": ["Gamma", "Delta", "Shared"],
    }


def _corpus_titles():
    return {"d1": "Alpha", "d2": "Beta", "d3": "Gamma", "d4": "Delta", "d5": "Shared", "d9": "Unpooled"}


def _corpus():
    return {d: f"{t} body text" for d, t in _corpus_titles().items()}


def test_pool_titles_reports_uniques_and_slots():
    """Both numbers, because only the pair is diagnostic: a loader that dropped a
    column would still produce a plausible unique count."""
    titles, slots = pool.pool_titles(_context())
    assert titles == {"Alpha", "Beta", "Gamma", "Delta", "Shared"}
    assert slots == 6, "the shared distractor occupies two slots but one unique title"


def test_build_keeps_corpus_ids_and_excludes_unpooled_documents():
    corpus, mapping = pool.build(_corpus(), _corpus_titles(), _context())
    assert set(corpus) == {"d1", "d2", "d3", "d4", "d5"}, "pool must be a subset, by corpus id"
    assert "d9" not in corpus, "a document nobody pooled must not enter the candidate set"
    assert corpus["d1"] == "Alpha body text", "pooled documents keep the corpus's own text"
    assert mapping["Shared"] == "d5"


def test_duplicate_title_is_refused_rather_than_resolved_by_iteration_order():
    """A duplicate title would make the pool depend on dict order, which is how a run
    stops being reproducible silently. Measured as zero on the real data, so it is
    checked rather than assumed."""
    titles = dict(_corpus_titles())
    titles["d6"] = "Alpha"
    with pytest.raises(RuntimeError, match="duplicate titles"):
        pool.build(_corpus() | {"d6": "Alpha other"}, titles, _context())


def test_pooled_title_absent_from_corpus_halts():
    """An unresolved title means the pool is not a subset of the corpus, which is the
    claim the whole construction rests on."""
    context = _context() | {"q3": ["Nowhere"]}
    with pytest.raises(RuntimeError, match="not a subset"):
        pool.build(_corpus(), _corpus_titles(), context)


def test_empty_titles_are_not_indexed_as_a_collision():
    """
    Untitled documents exist in BEIR; two of them are not an ambiguous title.

    Exercises title_index DIRECTLY rather than through build(). Mutation testing caught
    that going through build() could not fail: build() filters the corpus down to pooled
    titles before indexing, and "" is never a pooled title, so the empty-title guard was
    never reached and deleting it left the test green. That is the same defect 002 found
    in its own encoder-transposition test — a test cited as covering a case that could
    not actually fail — so it is fixed the same way, by testing the unit that owns the
    behaviour.
    """
    index = pool.title_index({"d1": "Alpha", "d7": "", "d8": ""})
    assert index == {"Alpha": "d1"}, "only titled documents are indexed"


def test_empty_titles_do_not_reach_the_pool_through_build():
    """The build path, which filters before indexing — kept alongside the unit test above
    so both routes are covered rather than one standing in for the other."""
    titles = _corpus_titles() | {"d7": "", "d8": ""}
    corpus, _ = pool.build(_corpus() | {"d7": "x", "d8": "y"}, titles, _context())
    assert set(corpus) == {"d1", "d2", "d3", "d4", "d5"}


def _control(**over):
    args = dict(
        questions=pool.EXPECTED_QUESTIONS,
        passages=pool.EXPECTED_PASSAGES,
        title_slots=pool.EXPECTED_TITLE_SLOTS,
        unresolved=0,
        collisions=0,
        gold_titles_matched=pool.EXPECTED_QUESTIONS,
        gold_queries=pool.EXPECTED_QUESTIONS,
    )
    args.update(over)
    return controls.pool_construction(**args)


def test_control_passes_on_the_measured_counts():
    assert _control()["passed"]


@pytest.mark.parametrize("field,value", [
    ("questions", 7404),
    ("passages", 66580),
    ("title_slots", 73699),
    ("gold_titles_matched", 7404),
])
def test_control_fails_when_any_frozen_count_moves(field, value):
    """Each count is falsifiable on its own. A control that only checks the total would
    let a compensating pair of errors through."""
    result = _control(**{field: value})
    assert not result["passed"]
    assert field in result["mismatched"]


@pytest.mark.parametrize("field", ["unresolved", "collisions"])
def test_control_fails_on_any_unresolved_title_or_collision(field):
    assert not _control(**{field: 1})["passed"]


# --- the extraction annotation sample, protocols/003-graph-arm.md section 8.2 ---

from pathlib import Path

from rb.experiments.graph import sample


def _big_pool():
    return {f"d{i}": f"doc {i} text" for i in range(500)}


def test_draw_is_deterministic_and_seed_sensitive():
    """The seed is frozen in the protocol before the draw, so a reader can check the
    annotated passages are the ones it selects."""
    a = sample.draw(_big_pool(), size=10)
    assert a == sample.draw(_big_pool(), size=10)
    assert a != sample.draw(_big_pool(), size=10, seed=sample.SAMPLE_SEED + 1)
    assert len(set(a)) == 10


def test_draw_does_not_depend_on_pool_insertion_order():
    """dict order is an accident of how the pool was built; if the draw tracked it, the
    sample would change whenever the loader changed."""
    forward = _big_pool()
    reversed_pool = {k: forward[k] for k in reversed(list(forward))}
    assert sample.draw(forward, size=10) == sample.draw(reversed_pool, size=10)


def test_template_is_written_unannotated(tmp_path: Path):
    """`entities` must ship empty. Pre-filling it with a model's guesses would make the
    control measure agreement rather than extraction quality."""
    import json

    out = sample.write_template(_big_pool(), {f"d{i}": f"T{i}" for i in range(500)},
                                tmp_path / "s.jsonl", size=10)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 10
    assert all(r["entities"] == [] and r["annotated"] is False for r in rows)
    assert all(r["doc_id"] in _big_pool() for r in rows)
