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


def test_cache_key_includes_model_name_so_two_models_cannot_share_vectors(tmp_path):
    """
    A reviewer demonstrated two encoders with different names but the same revision
    string sharing a cache entry, so the second never encoded anything and silently
    inherited the first's vectors. Dormant today because revision hashes are unique
    per repository, and exactly the kind of thing that stops being dormant after an
    edit that updates one constant and not the other.
    """
    from rb.experiments.ladder.retrievers.dense import _corpus_fingerprint

    a = _corpus_fingerprint(["d1"], ["hello"], 256, "org/model-a")
    b = _corpus_fingerprint(["d1"], ["hello"], 256, "org/model-b")
    assert a != b, "two different models must not share a cache fingerprint"


def test_two_pinned_encoders_cannot_collide_in_the_cache(tmp_path):
    """
    protocol-002-amendment-2 section 3's "confirm the two encoders cannot
    collide" requirement, exercised with the actual pinned model names/revisions
    from rb.experiments.ladder.run rather than arbitrary stand-ins — MiniLM
    (amendment 1) and bge-base-en-v1.5 (amendment 2) must land in separate cache
    subdirectories and never share a fingerprint, so switching encoders on the
    same corpus can never silently reuse the other's vectors.
    """
    from rb.experiments.ladder.run import (
        BGE_MODEL_NAME, BGE_REVISION, ENCODER_MODEL_NAME, ENCODER_REVISION,
    )

    n = 4
    corpus = {f"d{i:02d}": f"document {i}" for i in range(n)}
    queries = {f"q{i:02d}": f"query {i}" for i in range(n)}

    minilm = OneHotEncoder(dim=n)
    minilm.model_name, minilm.revision = ENCODER_MODEL_NAME, ENCODER_REVISION
    bge = OneHotEncoder(dim=n)
    bge.model_name, bge.revision = BGE_MODEL_NAME, BGE_REVISION

    DenseRetriever(minilm, normalize=False, cache_dir=tmp_path).retrieve(corpus, queries, top_k=1)
    DenseRetriever(bge, normalize=False, cache_dir=tmp_path).retrieve(corpus, queries, top_k=1)

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([ENCODER_REVISION, BGE_REVISION])


def test_manifest_records_the_embedding_hash_after_retrieval(tmp_path):
    """
    The manifest must describe the run that happened.

    manifest() was being evaluated before retrieve(), so the embedding hash — which
    exists only once the encoder has run — was silently absent from every artifact.
    The point of that hash is that two runs producing different vectors say so, which
    is worth nothing if it is never written.
    """
    import json
    from rb.retriever import run_rung
    from rb.experiments.ladder.retrievers.dense import DenseRetriever

    class _Stub:
        model_name, revision, max_length, batch_size = "stub/model", "rev0", 8, 4
        precision, pooling, normalized = "float32", "mean", True
        verification = None

        def _vec(self, t):
            import numpy as np
            h = abs(hash(t)) % 7 + 1
            return np.array([h, 1.0], dtype="float32")

        def encode_documents(self, texts):
            import numpy as np
            return np.stack([self._vec(t) for t in texts])

        encode_queries = encode_documents

    r = DenseRetriever(_Stub(), cache_dir=tmp_path / "cache")
    out = tmp_path / "rung"
    run_rung(r, "scifact", {"a": "alpha", "b": "beta"}, {"q": "alpha"}, {"q": {"a": 1}},
             out, top_k=10, extra_manifest=r.manifest)
    written = json.loads((out / "summary.json").read_text())
    assert written["retriever_manifest"].get("embedding_sha256"), "embedding hash missing from the artifact"


class _Magnitudes:
    """Vectors that point the same way but differ in length, so cosine and dot
    product disagree about the ranking."""

    model_name, revision, max_length, batch_size = "stub/mag", "rev0", 8, 4
    precision, pooling, normalized = "float32", "mean", True
    verification = None

    _V = {
        "near": [1.0, 0.0],     # identical direction to the query, short
        "far": [8.0, 6.0],      # longer, but pointing away (cos = 0.8)
        "query": [1.0, 0.0],
    }

    def encode_documents(self, texts):
        import numpy as np
        return np.array([self._V[t] for t in texts], dtype="float32")

    def encode_queries(self, texts):
        import numpy as np
        return np.array([self._V[t] for t in texts], dtype="float32")


def test_ranking_is_cosine_not_dot_product():
    """
    The amendment says exact COSINE. Nothing tested it.

    Every existing dense test passed normalize=False, so a reviewer replaced
    _l2_normalize with a no-op — dot product instead of cosine — and the whole
    suite passed. The pinned model happens to emit unit-norm vectors, which is why
    it went unnoticed, but that is a property of this checkpoint and not of the code.

    "far" has the larger dot product (8.0 against 1.0) and the smaller cosine
    (0.8 against 1.0), so the two rules disagree and only one is correct here.
    """
    from rb.experiments.ladder.retrievers.dense import DenseRetriever

    r = DenseRetriever(_Magnitudes())  # default normalize=True, as production runs it
    run = r.retrieve({"near": "near", "far": "far"}, {"q": "query"}, top_k=2)
    assert list(run["q"])[0] == "near", "ranked by dot product, not cosine"


class _Asymmetric:
    """Encodes documents and queries differently, so using the wrong one is visible."""

    model_name, revision, max_length, batch_size = "stub/asym", "rev0", 8, 4
    precision, pooling, normalized = "float32", "mean", True
    verification = None

    def encode_documents(self, texts):
        import numpy as np
        return np.array([[1.0, 0.0] if t == "doc-a" else [0.0, 1.0] for t in texts], dtype="float32")

    def encode_queries(self, texts):
        import numpy as np
        # The query text is deliberately NOT "doc-a", so encode_documents would give
        # it [0,1] while encode_queries gives it [1,0]. The two disagree on this exact
        # input, which is what makes the swap observable. An earlier version of this
        # stub had both encoders agree here, so it proved nothing.
        return np.array([[1.0, 0.0] if t == "wants-a" else [0.0, 1.0] for t in texts], dtype="float32")


class _FakeSTModel:
    """Stands in for the real sentence_transformers model object that
    SentenceTransformerEncoder wraps: records exactly which texts .encode() was
    called with, so the query-prefix tests below can assert on them directly."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy):
        self.calls.append(list(texts))
        return np.zeros((len(texts), 2), dtype="float32")


def _bare_encoder(query_prefix):
    """A SentenceTransformerEncoder with its query_prefix/batch_size/_model set
    directly, bypassing __init__ (which loads a real transformer — see the
    module docstring on why tests never do that). This is the seam
    protocols/002-amendment-2-second-encoder.md section 3 needs covered: a test
    that fails if the prefix is dropped or applied to documents."""
    from rb.experiments.ladder.retrievers.dense import SentenceTransformerEncoder

    enc = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    enc.batch_size = 8
    enc.query_prefix = query_prefix
    enc._model = _FakeSTModel()
    return enc


def test_query_prefix_is_applied_to_queries_only():
    """Amendment 2 section 3: bge-base's asymmetric convention must reach
    encode_queries and must NOT reach encode_documents. This is the test the
    task calls for — it fails if the prefix is dropped from queries or leaks
    into documents."""
    prefix = "Represent this sentence for searching relevant passages: "
    enc = _bare_encoder(query_prefix=prefix)

    enc.encode_documents(["a document"])
    enc.encode_queries(["a query"])

    assert enc._model.calls[0] == ["a document"], "documents must be encoded plain, never prefixed"
    assert enc._model.calls[1] == [prefix + "a query"], "queries must carry the prefix"


def test_no_query_prefix_configured_leaves_queries_unchanged():
    """MiniLM (amendment 1) passes query_prefix=None; this must be a true no-op,
    not an accidental empty-string prefix that happens to look like one."""
    enc = _bare_encoder(query_prefix=None)
    enc.encode_queries(["a query"])
    assert enc._model.calls[0] == ["a query"]


def test_expected_dim_mismatch_raises_rather_than_silently_recording_wrong_dim():
    """Amendment 2 section 3: 768 dimensions is verified against the loaded
    model, not trusted. A model producing a different dimension than requested
    means the wrong revision loaded, which must fail loudly. Exercises the real
    __init__ code path's own verification function, not a reimplementation."""
    import pytest
    from rb.experiments.ladder.retrievers.dense import _verify_dim

    with pytest.raises(RuntimeError, match="384"):
        _verify_dim(actual_dim=384, expected_dim=768, model_name="fake/model", revision="deadbeef")


def test_expected_dim_none_skips_the_check():
    """MiniLM's caller (amendment 1) passes no expected_dim; that must not raise
    regardless of the model's actual dimension."""
    from rb.experiments.ladder.retrievers.dense import _verify_dim

    _verify_dim(actual_dim=384, expected_dim=None, model_name="fake/model", revision="deadbeef")  # no raise


def test_retrieve_uses_the_document_encoder_for_documents_and_the_query_encoder_for_queries():
    """
    Exercises retrieve()'s actual call sites.

    The existing transposition test hand-built its own encoder object rather than
    going through DenseRetriever.retrieve(), so a reviewer swapped the two calls
    inside retrieve() and the suite still passed, except incidentally via an
    unrelated cache call-count assertion. The real encoder's document and query
    paths are byte-identical today, so the self-retrieval control cannot catch this
    class of defect either.
    """
    from rb.experiments.ladder.retrievers.dense import DenseRetriever

    r = DenseRetriever(_Asymmetric())
    run = r.retrieve({"doc-a": "doc-a", "doc-b": "doc-b"}, {"q": "wants-a"}, top_k=2)
    # Through the query encoder "wants-a" is [1,0], matching doc-a. Through the
    # document encoder it would be [0,1], matching doc-b.
    assert list(run["q"])[0] == "doc-a", "encoders transposed at the retrieve() call sites"


def test_self_retrieval_threshold_tolerates_a_near_duplicate_but_not_misalignment():
    """
    The threshold from amendment 3, pinned so it cannot drift.

    98 of 100 passes, because a duplicate-question corpus can legitimately have a
    paraphrase edge out the document itself. 96 of 100 fails, because a genuine
    id-alignment failure breaks nearly every sample rather than one or two, and
    the threshold has to stay tight enough to catch that.
    """
    from rb import controls

    class _Rigged:
        """Returns itself for all but `misses` of the sampled documents."""

        model_name, revision, precision, pooling, max_length, batch_size = (
            "rigged", "rev0", "float32", "none", 8, 8,
        )
        verification = None

        def __init__(self, misses: int):
            self.misses = misses

        def retrieve(self, corpus, queries, top_k):
            out, missed = {}, 0
            for qid in sorted(queries):
                if missed < self.misses:
                    other = next(d for d in sorted(corpus) if d != qid)
                    out[qid] = {other: 1.0}
                    missed += 1
                else:
                    out[qid] = {qid: 1.0}
            return out

    corpus = {f"d{i:03d}": f"text {i}" for i in range(100)}
    ids = sorted(corpus)

    assert controls.self_retrieval(_Rigged(2), corpus, ids)["passed"], "2 misses must pass"
    assert not controls.self_retrieval(_Rigged(4), corpus, ids)["passed"], "4 misses must fail"
