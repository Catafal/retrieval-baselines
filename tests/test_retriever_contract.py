"""
The Retriever contract, checked once against every implementation on the same
tiny in-memory corpus — not against internals, against the seam every rung shares
(see rb.retriever.Retriever).
"""

import zlib

import numpy as np

from rb.experiments.ladder.retrievers.coordination import CoordinationRetriever
from rb.experiments.ladder.retrievers.dense import DenseRetriever
from rb.experiments.ladder.retrievers.hybrid import HybridRetriever
from rb.experiments.ladder.retrievers.lexical import ALL_CONFIGS
from helpers import assert_retriever_contract

CORPUS = {
    "a": "insulin lowers blood glucose in diabetic patients",
    "b": "photosynthesis converts light energy into chemical energy",
    "c": "insulin resistance is linked to type two diabetes",
    "d": "the mitochondria is the powerhouse of the cell",
}
QUERIES = {
    "q1": "insulin diabetes",
    "q2": "energy",
    "q3": "no matching terms here at all",
}


class StubEncoder:
    """Deterministic, dependency-free stand-in for a real sentence-transformers
    encoder — hashes each text into a fixed-size vector so cosine similarity is
    well-defined without downloading a model."""

    model_name = "stub-encoder"
    revision = "0000000000000000"
    precision = "float32"
    pooling = "mean"
    max_length = 128

    def _embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        dim = 16
        out = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            # crc32, not hash(): built-in hash() on str is salted per process
            # (PYTHONHASHSEED), so a stub built on it produces different vectors
            # every run — in the one file whose subject is determinism.
            seed = zlib.crc32(t.encode())
            out[i] = np.random.default_rng(seed).normal(size=dim)
        return out

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


def test_coordination_contract():
    assert_retriever_contract(CoordinationRetriever(), CORPUS, QUERIES, top_k=2)


def test_lexical_contract_all_eight_configs():
    for cfg in ALL_CONFIGS:
        assert_retriever_contract(cfg, CORPUS, QUERIES, top_k=2)


def test_dense_contract():
    assert_retriever_contract(DenseRetriever(StubEncoder()), CORPUS, QUERIES, top_k=2)


def test_hybrid_contract():
    lexical = ALL_CONFIGS[0]
    dense = DenseRetriever(StubEncoder())
    assert_retriever_contract(HybridRetriever(lexical, dense, candidate_k=4), CORPUS, QUERIES, top_k=2)


def test_scoring_uses_rank_order_not_score_magnitude(tmp_path):
    """
    A retriever whose scores differ by less than the scorer's resolution must still
    be scored in the order it asked for.

    pytrec_eval compares at reduced precision, so a 1e-9 separation is invisible to
    it and it falls back to its own tie rule. The lexical rung separates ties by
    exactly that much, so without re-encoding, the ranking scored is not the ranking
    protocols/002-ladder.md pre-registers.
    """
    from rb.retriever import run_rung

    class _Tiny:
        """Puts 'b' first, but by a margin far below what the scorer can see."""

        name = "tiny"

        def retrieve(self, corpus, queries, top_k):
            return {"q": {"b": 4.0, "a": 4.0 - 1e-9}}

    corpus = {"a": "alpha", "b": "beta"}
    queries = {"q": "alpha beta"}
    qrels = {"q": {"b": 1}}  # only b is relevant, so b-first must score 1.0

    summary = run_rung(_Tiny(), "scifact", corpus, queries, qrels, tmp_path, top_k=10)
    assert summary["ranked"]["ndcg_cut_10"] == 1.0

    class _TinyReversed(_Tiny):
        name = "tiny-reversed"

        def retrieve(self, corpus, queries, top_k):
            return {"q": {"a": 4.0, "b": 4.0 - 1e-9}}

    reversed_summary = run_rung(
        _TinyReversed(), "scifact", corpus, queries, qrels, tmp_path / "r", top_k=10
    )
    # Same epsilon, opposite intent: the two must NOT score the same.
    assert reversed_summary["ranked"]["ndcg_cut_10"] < 1.0
