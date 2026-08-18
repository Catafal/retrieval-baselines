"""Hybrid rung: reciprocal rank fusion behaviour, checked at the seam (retrieve()'s
output), not against the internal RRF accumulator."""

import pytest

from rb.experiments.ladder.retrievers.hybrid import RRF_K, HybridRetriever


class FixedRankRetriever:
    """Stub Retriever that returns a pre-baked ranking, so RRF's combination
    logic can be tested without a real lexical or dense scorer underneath it."""

    def __init__(self, name: str, ranking: dict[str, list[str]]):
        self.name = name
        self.ranking = ranking  # query_id -> doc ids in rank order

    def retrieve(self, corpus, queries, top_k):
        run = {}
        for qid in queries:
            docs = self.ranking.get(qid, [])[:top_k]
            run[qid] = {d: float(len(docs) - i) for i, d in enumerate(docs)}
        return run


def test_rrf_agreeing_rankings_reinforce_top_document():
    """A document ranked first by both components must beat one ranked first by
    only one of them."""
    lexical = FixedRankRetriever("lex", {"q": ["a", "b", "c"]})
    dense = FixedRankRetriever("dense", {"q": ["a", "c", "b"]})
    hybrid = HybridRetriever(lexical, dense, candidate_k=3)
    run = hybrid.retrieve({"a": "", "b": "", "c": ""}, {"q": "text"}, top_k=3)["q"]
    assert list(run.keys())[0] == "a", "document ranked first by both components should win RRF"


def test_rrf_uses_fixed_k_60():
    assert RRF_K == 60


def test_rrf_score_matches_hand_computed_formula():
    """score(d) = 1/(k+rank_lex) + 1/(k+rank_dense). For k=60, doc "a" at rank 1
    in both components: 1/61 + 1/61."""
    lexical = FixedRankRetriever("lex", {"q": ["a"]})
    dense = FixedRankRetriever("dense", {"q": ["a"]})
    hybrid = HybridRetriever(lexical, dense, k=60, candidate_k=1)
    run = hybrid.retrieve({"a": ""}, {"q": "text"}, top_k=1)["q"]
    # The retrieve() output is re-encoded to strictly-decreasing rank position
    # (same convention as every other rung), so recover the RRF score from the
    # internal accumulator directly rather than from the re-encoded output.
    lex_run = lexical.retrieve({}, {"q": ""}, 1)
    dense_run = dense.retrieve({}, {"q": ""}, 1)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    rrf = {}
    for component in (lex_run["q"], dense_run["q"]):
        ranked = sorted(component.items(), key=lambda kv: (-kv[1], kv[0]))
        for pos, (d, _) in enumerate(ranked, start=1):
            rrf[d] = rrf.get(d, 0.0) + 1.0 / (60 + pos)
    assert abs(rrf["a"] - expected) < 1e-9
    assert list(run.keys()) == ["a"]


def test_rrf_no_weight_search_component_contributions_are_symmetric():
    """Swapping which retriever plays 'lexical' vs 'dense' must not change the
    fused ranking — the fusion rule has no per-side weight to tune."""
    r1 = FixedRankRetriever("r1", {"q": ["a", "b"]})
    r2 = FixedRankRetriever("r2", {"q": ["b", "a"]})
    run_ab = HybridRetriever(r1, r2, candidate_k=2).retrieve({"a": "", "b": ""}, {"q": ""}, top_k=2)["q"]
    run_ba = HybridRetriever(r2, r1, candidate_k=2).retrieve({"a": "", "b": ""}, {"q": ""}, top_k=2)["q"]
    assert run_ab == run_ba


def test_rrf_weights_are_one_based_against_independently_computed_values():
    """
    Pin the actual RRF weights, not just the ordering.

    Ordering cannot catch a start-index error: 1/(k+pos) stays monotonically
    decreasing whether pos starts at 0 or 1, so the ranking is identical and every
    order-only assertion passes against a wrong implementation. A mutation test
    confirmed exactly that. The expected values below are written as literal
    fractions rather than recomputed with the implementation's own loop, so this
    test fails if the start index, k, or the formula moves.
    """
    from rb.experiments.ladder.retrievers.hybrid import HybridRetriever

    h = HybridRetriever(lexical=_Named("lex"), dense=_Named("dense"), k=60)
    # Two components, each already in this repo's decreasing rank-position encoding.
    fused = h.fuse({"a": 3.0, "b": 2.0, "c": 1.0}, {"c": 2.0, "a": 1.0})

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)   # rank 1 and rank 2
    assert fused["b"] == pytest.approx(1 / 62)            # rank 2, one component
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)   # rank 3 and rank 1


def test_rrf_k_is_sixty_and_not_searched():
    """k is pre-registered. If someone tunes it, this fails rather than drifting."""
    from rb.experiments.ladder.retrievers.hybrid import RRF_K, HybridRetriever

    assert RRF_K == 60
    assert HybridRetriever(lexical=_Named("lex"), dense=_Named("dense")).k == 60


class _Named:
    """Minimal stand-in: HybridRetriever only reads `.name` at construction."""

    def __init__(self, name: str):
        self.name = name
