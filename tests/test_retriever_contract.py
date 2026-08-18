"""
The Retriever contract, checked once against every implementation on the same
tiny in-memory corpus — not against internals, against the seam every rung shares
(see rb.retriever.Retriever).
"""

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
            seed = abs(hash(t)) % (2**32)
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
