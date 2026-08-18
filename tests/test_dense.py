"""
Dense retriever: ranking behaviour against a stub encoder (never a real model —
see rb.experiments.ladder.retrievers.dense's module docstring for why), plus the
embedding-shuffle control exercised end to end rather than only as a unit test on
hardcoded numbers.
"""

import numpy as np

from rb import controls, metrics
from rb.experiments.ladder.retrievers.dense import DenseRetriever


class OneHotEncoder:
    """Deterministic stub: encode_documents/encode_queries return one-hot rows in
    call order, so a query aligned with document i by construction gets a perfect
    cosine match — this is what makes "normal" retrieval below trivially correct,
    and what makes a shuffle's effect measurable."""

    model_name = "one-hot-stub"
    revision = "deadbeef00000000"
    precision = "float32"
    pooling = "none"
    max_length = 32

    def __init__(self, dim: int):
        self.dim = dim

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), self.dim)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), self.dim)


class NoisyAlignedEncoder:
    """Row i of both matrices is a dominant one-hot spike in dimension i plus
    small fixed Gaussian noise across every dimension. The noise exists only so
    cosine scores are continuous rather than exactly tied at zero — pure one-hot
    vectors make every non-matching pair score exactly 0, and this repo's tie-break
    (document id, ascending) would then rank a low-id document first for every
    query regardless of the embeddings, which would make a shuffled run look
    artificially better than shuffling actually made it. The noise is small enough
    that the dominant spike still wins whenever doc and query indices align."""

    model_name = "noisy-aligned-stub"
    revision = "deadbeef00000000"
    precision = "float32"
    pooling = "none"
    max_length = 32

    def __init__(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._vecs = np.eye(n, n) * 5.0 + rng.normal(scale=0.05, size=(n, n))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._vecs.copy()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._vecs.copy()


def _ndcg(run: dict, qrels: dict) -> float:
    per_query = metrics.score_ranked(qrels, run)
    return metrics.mean([per_query[q]["ndcg_cut_10"] for q in qrels])


def test_dense_retrieve_ranks_by_cosine_similarity():
    # Zero-padded ids so lexical sort order matches the numeric insertion order,
    # which the one-hot alignment below depends on.
    corpus = {f"d{i:02d}": f"document {i}" for i in range(5)}
    queries = {f"q{i:02d}": f"query {i}" for i in range(5)}
    encoder = OneHotEncoder(dim=5)
    retriever = DenseRetriever(encoder, normalize=False)
    run = retriever.retrieve(corpus, queries, top_k=1)
    for i in range(5):
        assert list(run[f"q{i:02d}"].keys()) == [f"d{i:02d}"], "top-1 should be the aligned one-hot document"


def test_embedding_shuffle_collapses_toward_chance():
    n = 30  # large enough that a random permutation leaves few queries aligned by chance
    corpus = {f"d{i:02d}": f"document {i}" for i in range(n)}
    queries = {f"q{i:02d}": f"query {i}" for i in range(n)}
    qrels = {f"q{i:02d}": {f"d{i:02d}": 1} for i in range(n)}
    encoder = NoisyAlignedEncoder(n=n, seed=1)

    normal = DenseRetriever(encoder, normalize=False)
    normal_ndcg = _ndcg(normal.retrieve(corpus, queries, top_k=10), qrels)
    assert normal_ndcg > 0.99, "the dominant aligned spike must win normal retrieval"

    shuffled = DenseRetriever(encoder, normalize=False, shuffle_seed=20260818)
    shuffled_ndcg = _ndcg(shuffled.retrieve(corpus, queries, top_k=10), qrels)

    result = controls.embedding_shuffle(normal_ndcg, shuffled_ndcg)
    assert shuffled_ndcg < normal_ndcg, "shuffling document embeddings must reduce nDCG"
    assert result["passed"], f"shuffled run did not collapse toward chance: {result}"
