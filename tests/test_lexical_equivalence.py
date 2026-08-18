"""
Proof that the inverted-index fast path in LexicalRetriever.retrieve() is a pure
performance refactor: it must return byte-for-byte identical run dicts (within a
tight float tolerance) to _ReferenceLexicalRetriever, the untouched naive
implementation, on every corpus below.

TOLERANCE. The two implementations sum the same term contributions in a
different order — the reference sums sequentially per document; the fast path
concatenates per-term numpy arrays and reduces with np.add.at, so a document
touched by two query terms accumulates them in whichever order np.unique put
its row in. IEEE 754 float64 addition is not associative, so a summation-order
difference can move the last one or two bits of a float64 mantissa (~1e-15
relative). 1e-9 absolute is roughly six orders of magnitude looser than that
noise floor and roughly nine orders tighter than the smallest score difference
any of the behavioural tests in test_lexical.py rely on (1e-6), so it catches a
real algebraic divergence while never flagging summation-order noise.

Real-corpus equivalence uses SciFact specifically (~5.2k docs): small enough to
run both the O(n) reference and the fast path in a test, the same corpus the
BM25 closure control is anchored against, and already downloaded locally per
manifests/datasets.json — no network access and no results/ write happen here.
"""

import dataclasses

import pytest

from rb import datasets
from rb.experiments.ladder.retrievers.lexical import (
    ALL_CONFIGS,
    LexicalRetriever,
    _ReferenceLexicalRetriever,
    build_index,
)

TOLERANCE = 1e-9


def _assert_equivalent(fast_run: dict, ref_run: dict) -> None:
    assert set(fast_run) == set(ref_run), "same set of query ids"
    for qid in fast_run:
        fast_docs, ref_docs = fast_run[qid], ref_run[qid]
        # dict insertion order is the rank order both implementations produce
        # (see the shared tie-break epsilon in lexical.py) — compare as lists,
        # not sets, so a reordering would fail the test even if the score set
        # happened to match.
        assert list(fast_docs) == list(ref_docs), f"{qid}: doc order differs"
        for doc_id in fast_docs:
            assert fast_docs[doc_id] == pytest.approx(ref_docs[doc_id], abs=TOLERANCE), (
                f"{qid}/{doc_id}: fast={fast_docs[doc_id]!r} ref={ref_docs[doc_id]!r}"
            )


def _reference_for(cfg: LexicalRetriever) -> _ReferenceLexicalRetriever:
    return _ReferenceLexicalRetriever(idf=cfg.idf, tf_saturation=cfg.tf_saturation, length_norm=cfg.length_norm)


# --- Real SciFact corpus, several of the eight configs -----------------------


@pytest.fixture(scope="module")
def scifact():
    corpus, queries, _qrels = datasets.load("scifact")
    # A slice of queries, not all 300 — the point is proving the two code paths
    # agree, not re-timing the whole dataset (that happens in the throughput
    # measurement reported separately, not as a committed test).
    sliced_queries = {qid: queries[qid] for qid in sorted(queries)[:40]}
    return corpus, sliced_queries


@pytest.mark.parametrize(
    "cfg",
    [
        LexicalRetriever(idf=True, tf_saturation=True, length_norm=True),  # full BM25 — the closure anchor
        LexicalRetriever(idf=False, tf_saturation=False, length_norm=False),  # raw tf sum
        LexicalRetriever(idf=True, tf_saturation=False, length_norm=True),
        LexicalRetriever(idf=False, tf_saturation=True, length_norm=False),
    ],
    ids=lambda c: c.name,
)
def test_fast_path_matches_reference_on_real_scifact(scifact, cfg):
    corpus, queries = scifact
    index = build_index(corpus)
    fast = dataclasses.replace(cfg, index=index)
    fast_run = fast.retrieve(corpus, queries, top_k=50)
    ref_run = _reference_for(cfg).retrieve(corpus, queries, top_k=50)
    _assert_equivalent(fast_run, ref_run)


def test_fast_path_matches_reference_without_a_prebuilt_index(scifact):
    """LexicalRetriever.retrieve() must also build its own index correctly
    when none is injected — the shared-index path (run_lexical_factorial) is
    an optimisation, not the only supported way to call retrieve()."""
    corpus, queries = scifact
    cfg = LexicalRetriever(idf=True, tf_saturation=True, length_norm=True)
    fast_run = cfg.retrieve(corpus, queries, top_k=50)
    ref_run = _reference_for(cfg).retrieve(corpus, queries, top_k=50)
    _assert_equivalent(fast_run, ref_run)


# --- Hand-built edge cases, every one of the eight configs --------------------

EDGE_CASE_CORPORA = {
    "no_matches": (
        {"a": "apple banana cherry", "b": "date elderberry fig"},
        {"q": "grape honeydew"},
    ),
    "term_in_every_document": (
        {"a": "common word here", "b": "common word there", "c": "common everywhere word"},
        {"q": "common"},
    ),
    "single_document": (
        {"only": "the only document in this corpus has these words"},
        {"q": "document words"},
    ),
    "empty_query": (
        {"a": "some text", "b": "other text"},
        {"q": ""},
    ),
    "zero_length_documents": (
        {"empty1": "", "empty2": "", "normal": "some actual content words"},
        {"q": "content words"},
    ),
    "ties": (
        # Two documents, identical length, each matching the query term with
        # identical term frequency: a genuine score tie the epsilon tie-break
        # must resolve identically (by document id) in both implementations.
        {"tied_a": "match filler filler", "tied_b": "match filler filler"},
        {"q": "match"},
    ),
}


@pytest.mark.parametrize("case_name", sorted(EDGE_CASE_CORPORA))
@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.name)
def test_fast_path_matches_reference_on_edge_cases(case_name, cfg):
    corpus, queries = EDGE_CASE_CORPORA[case_name]
    index = build_index(corpus)
    fast = dataclasses.replace(cfg, index=index)
    fast_run = fast.retrieve(corpus, queries, top_k=10)
    ref_run = _reference_for(cfg).retrieve(corpus, queries, top_k=10)
    _assert_equivalent(fast_run, ref_run)


def test_empty_corpus_both_implementations_return_empty_runs():
    corpus: dict[str, str] = {}
    queries = {"q": "anything"}
    cfg = LexicalRetriever(idf=True, tf_saturation=True, length_norm=True)
    index = build_index(corpus)
    fast = dataclasses.replace(cfg, index=index)
    fast_run = fast.retrieve(corpus, queries, top_k=10)
    ref_run = _reference_for(cfg).retrieve(corpus, queries, top_k=10)
    assert fast_run == {"q": {}}
    assert ref_run == {"q": {}}
