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
    batch_size = 32

    def __init__(self, dim: int):
        self.dim = dim
        self.calls: list[int] = []  # batch sizes the caller "used", for the wiring test below

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
    batch_size = 32

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


def test_manifest_reports_batch_size_from_the_encoder_not_a_separate_field():
    """
    Regression for the defect this change fixes: a prior DenseRetriever had its
    own `batch_size` field that was reported in the manifest but never passed to
    the encoder, so the number in results/002/*/dense/summary.json could disagree
    with what actually reached sentence-transformers. batch_size is now owned
    solely by the encoder; the manifest must read it back from there.
    """
    encoder = OneHotEncoder(dim=3)
    encoder.batch_size = 7  # a value DenseRetriever never sets or sees directly
    retriever = DenseRetriever(encoder, normalize=False)
    assert retriever.manifest()["batch_size"] == 7


def test_self_retrieval_passes_when_query_and_document_paths_agree():
    n = 10
    corpus = {f"d{i:02d}": f"document {i}" for i in range(n)}
    encoder = NoisyAlignedEncoder(n=n, seed=2)
    retriever = DenseRetriever(encoder, normalize=False)
    result = controls.self_retrieval(retriever, corpus, sample_ids=sorted(corpus)[:5])
    assert result["passed"], result


def test_self_retrieval_fails_when_query_and_document_encoders_are_transposed():
    """
    The control this test exists for: query and document encoders swapped.
    Simulated with an encoder whose encode_queries returns a DIFFERENT (shifted)
    permutation than encode_documents, so a document's own text no longer
    retrieves itself at rank 1 even though nothing about the embedding-shuffle
    control (which only permutes doc-side vectors, not the query path) would
    catch it.
    """
    n = 10

    class TransposedEncoder:
        model_name, revision, precision, pooling, max_length, batch_size = (
            "transposed-stub", "deadbeef", "float32", "none", 32, 32,
        )

        def __init__(self, n: int):
            self._doc_vecs = np.eye(n, n)
            # Query vectors are the document matrix rolled by one row, so no
            # query aligns with its own document's vector anymore.
            self._query_vecs = np.roll(np.eye(n, n), shift=1, axis=0)

        def encode_documents(self, texts: list[str]) -> np.ndarray:
            return self._doc_vecs.copy()

        def encode_queries(self, texts: list[str]) -> np.ndarray:
            return self._query_vecs.copy()

    corpus = {f"d{i:02d}": f"document {i}" for i in range(n)}
    retriever = DenseRetriever(TransposedEncoder(n), normalize=False)
    result = controls.self_retrieval(retriever, corpus, sample_ids=sorted(corpus))
    assert not result["passed"], "a transposed query/document path must fail self-retrieval"
    assert result["failures"] == n


def test_document_embeddings_are_cached_to_disk_and_reused(tmp_path):
    """A second retrieve() call against the same corpus and revision must not
    call encode_documents again — that is the whole point of the cache."""
    n = 6
    corpus = {f"d{i:02d}": f"document {i}" for i in range(n)}
    queries = {f"q{i:02d}": f"query {i}" for i in range(n)}
    encoder = OneHotEncoder(dim=n)

    original_encode = encoder.encode_documents
    call_count = {"n": 0}

    def counting_encode(texts):
        call_count["n"] += 1
        return original_encode(texts)

    encoder.encode_documents = counting_encode

    retriever = DenseRetriever(encoder, normalize=False, cache_dir=tmp_path)
    retriever.retrieve(corpus, queries, top_k=1)
    retriever.retrieve(corpus, queries, top_k=1)
    assert call_count["n"] == 1, "second retrieve() on the same corpus should hit the cache, not re-embed"


def test_cache_is_keyed_by_revision_so_a_different_revision_cannot_reuse_vectors(tmp_path):
    """Two encoders with the same corpus but different `revision` must each
    embed independently — a different revision reusing another's cached
    vectors would silently report a result that was never actually produced
    by the revision the manifest claims."""
    n = 4
    corpus = {f"d{i:02d}": f"document {i}" for i in range(n)}
    queries = {f"q{i:02d}": f"query {i}" for i in range(n)}

    encoder_a = OneHotEncoder(dim=n)
    encoder_a.revision = "revision-aaaaaaaa"
    encoder_b = OneHotEncoder(dim=n)
    encoder_b.revision = "revision-bbbbbbbb"

    DenseRetriever(encoder_a, normalize=False, cache_dir=tmp_path).retrieve(corpus, queries, top_k=1)
    DenseRetriever(encoder_b, normalize=False, cache_dir=tmp_path).retrieve(corpus, queries, top_k=1)

    cached_dirs = sorted(p.name for p in tmp_path.iterdir())
    assert cached_dirs == ["revision-aaaaaaaa", "revision-bbbbbbbb"]
