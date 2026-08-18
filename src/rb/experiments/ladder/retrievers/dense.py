"""
The dense rung — exact cosine similarity over full-precision embeddings, no ANN index.

An approximate index trades recall for speed. Reporting a number produced through
one would attribute that lost recall to the embedding model instead of to the
index, so this retriever always scores every document in the corpus directly.

The encoder is injected as an `Encoder` object rather than hard-coded, so the
ranking logic in this file (cosine, tie-break, the shuffle hook the embedding
control uses) is testable with a tiny deterministic stub and never requires
sentence-transformers/torch to be installed just to run the unit tests.
SentenceTransformerEncoder below is the production implementation; it imports the
real library lazily, inside __init__, so importing this module alone stays cheap.
"""

from typing import Protocol

import numpy as np


class Encoder(Protocol):
    """Everything the manifest needs to pin an embedding to a specific artifact,
    not to a moving model name."""

    model_name: str
    revision: str
    precision: str
    pooling: str
    max_length: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class DenseRetriever:
    def __init__(
        self,
        encoder: Encoder,
        batch_size: int = 32,
        normalize: bool = True,
        shuffle_seed: int | None = None,
    ):
        self.encoder = encoder
        self.batch_size = batch_size
        self.normalize = normalize
        # Set only by the embedding-shuffle control (rb.controls.embedding_shuffle):
        # permutes the doc-id -> vector assignment before scoring, deterministically,
        # so a broken vector index collapses toward chance in a test instead of
        # silently reporting a real-looking number.
        self.shuffle_seed = shuffle_seed
        self.name = f"dense({encoder.model_name}@{encoder.revision[:8]})"

    def retrieve(
        self, corpus: dict[str, str], queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        doc_ids = list(corpus)
        doc_vecs = self.encoder.encode_documents([corpus[d] for d in doc_ids])
        if self.shuffle_seed is not None:
            doc_vecs = _shuffle_rows(doc_vecs, self.shuffle_seed)
        if self.normalize:
            doc_vecs = _l2_normalize(doc_vecs)

        q_ids = sorted(queries)
        q_vecs = self.encoder.encode_queries([queries[q] for q in q_ids])
        if self.normalize:
            q_vecs = _l2_normalize(q_vecs)

        sims = q_vecs @ doc_vecs.T  # exact cosine over the full corpus, no ANN structure

        run: dict[str, dict[str, float]] = {}
        for i, qid in enumerate(q_ids):
            order = sorted(range(len(doc_ids)), key=lambda j: (-sims[i, j], doc_ids[j]))[:top_k]
            # Same strictly-decreasing rank-position encoding as every other rung
            # (see coordination.py / lexical.py): cosine scores can tie, and
            # trec_eval must break ties by document id.
            run[qid] = {doc_ids[j]: float(len(order) - r) for r, j in enumerate(order)}
        return run

    def manifest(self) -> dict:
        """Deterministic settings a stranger needs to tell an environment
        difference from a finding: precision, batch size, normalisation, pooling
        and truncation length, since documents longer than max_length are
        truncated and that is a property of the measurement."""
        return {
            "model_name": self.encoder.model_name,
            "revision": self.encoder.revision,
            "precision": self.encoder.precision,
            "pooling": self.encoder.pooling,
            "max_length": self.encoder.max_length,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
        }


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector normalizes to itself rather than dividing by zero
    return mat / norms


def _shuffle_rows(mat: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic permutation of embedding rows, used only by the embedding-
    shuffle control to break the doc-id -> vector correspondence on purpose."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(mat.shape[0])
    return mat[order]


class SentenceTransformerEncoder:
    """
    Production encoder. Pinned by model name AND revision hash — a model name
    alone can move under you if the hub's "main" branch changes weights, so the
    result must be attached to a specific revision, not a name.

    Not exercised by the test suite: importing sentence-transformers happens
    inside __init__, not at module load, specifically so this class can exist and
    be reviewed without the dependency being installed. Experiment scope is code
    + tests only — no corpus is embedded with this class in this change.
    """

    precision = "float32"
    pooling = "mean"  # model-specific; recorded rather than assumed correct

    def __init__(self, model_name: str, revision: str, max_length: int = 512):
        from sentence_transformers import SentenceTransformer  # lazy: see class docstring

        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self._model = SentenceTransformer(model_name, revision=revision)
        self._model.max_seq_length = max_length

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        # Separate from encode_documents because some models use an asymmetric
        # query/document convention (e.g. an "query: " prefix); this is the seam
        # where that convention would be applied, kept visible rather than buried
        # inside a single shared encode().
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        )
