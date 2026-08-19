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

FIXED SINCE THE LAST REVIEW (protocol-002-amendment-1, section 4):
  - `doc_ids` is now `sorted(corpus)`, not `list(corpus)`. The old code trusted
    dict insertion order, which is a house-rule violation (no dict-order
    dependence) and made the on-disk embedding cache below unkeyable — the same
    corpus dict built in a different order would fingerprint differently.
  - `batch_size` was a field on DenseRetriever that was never passed to the
    encoder: `encode_documents`/`encode_queries` took no batch_size argument, so
    the manifest reported a batch size that never reached the model. Batch size
    is now owned by the encoder (the object that actually calls .encode()) and
    DenseRetriever.manifest() reads it back from there.
  - SentenceTransformerEncoder no longer force-overwrites max_seq_length with an
    unverified default. It reads the model's own configured value and only
    overrides when the caller explicitly asks for a different one, recording
    which happened — see its docstring.
"""

import hashlib
from pathlib import Path
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
    batch_size: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class DenseRetriever:
    def __init__(
        self,
        encoder: Encoder,
        normalize: bool = True,
        shuffle_seed: int | None = None,
        cache_dir: Path | None = None,
    ):
        self.encoder = encoder
        self.normalize = normalize
        # Set only by the embedding-shuffle control (rb.controls.embedding_shuffle):
        # permutes the doc-id -> vector assignment before scoring, deterministically,
        # so a broken vector index collapses toward chance in a test instead of
        # silently reporting a real-looking number.
        self.shuffle_seed = shuffle_seed
        # Document embeddings are the expensive half of a rerun; caching them
        # to disk means a second invocation on the same corpus + revision does
        # not re-embed. None (the default) disables caching entirely, which is
        # what every stub-encoder test below exercises. See _cached_doc_vectors.
        self.cache_dir = cache_dir
        self._embedding_digest: str | None = None
        self.name = f"dense({encoder.model_name}@{encoder.revision[:8]})"

    def retrieve(
        self, corpus: dict[str, str], queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        # Sorted, not `list(corpus)`: the old code trusted dict insertion order,
        # which two processes need not agree on if the corpus was ever built by
        # iterating something order-unstable upstream, and it made the embedding
        # cache below unkeyable — the same corpus in a different dict order would
        # fingerprint differently and never hit the cache. Same convention as
        # lexical.build_index's `doc_ids = tuple(sorted(corpus))`.
        doc_ids = sorted(corpus)
        texts = [corpus[d] for d in doc_ids]
        if self.cache_dir is not None:
            doc_vecs = self._cached_doc_vectors(doc_ids, texts)
        else:
            doc_vecs = self.encoder.encode_documents(texts)
        if self.shuffle_seed is not None:
            doc_vecs = _shuffle_rows(doc_vecs, self.shuffle_seed)
        if self.normalize:
            doc_vecs = _l2_normalize(doc_vecs)

        q_ids = sorted(queries)
        q_vecs = self.encoder.encode_queries([queries[q] for q in q_ids])
        if self.normalize:
            q_vecs = _l2_normalize(q_vecs)

        sims = q_vecs @ doc_vecs.T  # exact cosine over the full corpus, no ANN structure

        # doc ids as an array once, for the vectorised tie-break below.
        doc_id_arr = np.array(doc_ids)

        run: dict[str, dict[str, float]] = {}
        for i, qid in enumerate(q_ids):
            # np.lexsort, not sorted() with a Python key. The ordering is identical —
            # score descending, ties broken by document id ascending — but the key
            # function ran once per document per query, which is 260 million calls on
            # Quora and 2.6 billion on HotpotQA, where it simply does not finish. A
            # reviewer projected that before it was run for real. lexsort does the same
            # comparison in C, and takes the LAST key as primary.
            order = np.lexsort((doc_id_arr, -sims[i]))[:top_k]
            # Same strictly-decreasing rank-position encoding as every other rung
            # (see coordination.py / lexical.py): cosine scores can tie, and
            # trec_eval must break ties by document id.
            run[qid] = {doc_ids[j]: float(len(order) - r) for r, j in enumerate(order)}
        return run

    def _cached_doc_vectors(self, doc_ids: list[str], texts: list[str]) -> np.ndarray:
        """
        Load this corpus's document embeddings from disk if present, else
        compute and save them.

        Keyed by model revision AND a content fingerprint of (doc_ids, texts,
        max_length): the revision alone is not enough — the same revision run
        with a different max_length truncates differently and would silently
        reuse the wrong vectors if that were not part of the key. The revision
        additionally names the cache subdirectory (not just an input to the
        hash), so a stranger inspecting the cache directory can see which
        revision produced which files without recomputing any hash.
        """
        fingerprint = _corpus_fingerprint(
            doc_ids, texts, self.encoder.max_length, self.encoder.model_name
        )
        path = self.cache_dir / self.encoder.revision / f"{fingerprint}.npz"
        if path.exists():
            # allow_pickle=False by default: doc_ids is stored as a fixed-width
            # unicode array (np.array(doc_ids), no dtype=object), specifically so
            # loading a cache file never needs pickle, which would otherwise be a
            # code-execution surface for a cache directory shared with anyone else.
            cached = np.load(path, allow_pickle=False)
            # Defensive re-check rather than trusting the filename: a hash
            # collision or a hand-edited cache directory would otherwise return
            # vectors for the wrong document order silently.
            if list(cached["doc_ids"]) != doc_ids:
                # Everything else in this repo fails loudly where a wrong number could
                # otherwise be produced quietly, and a silent recompute here would turn a
                # hash collision or a hand-edited cache into a slow run rather than a
                # stated problem.
                raise RuntimeError(
                    f"cache file {path} matches the fingerprint but holds different "
                    "document ids. Delete it rather than trusting either."
                )
            vectors = cached["vectors"].astype(np.float32)
            self._embedding_digest = _matrix_digest(vectors)
            return vectors
        vecs = np.asarray(self.encoder.encode_documents(texts), dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, doc_ids=np.array(doc_ids), vectors=vecs)
        self._embedding_digest = _matrix_digest(vecs)
        return vecs

    def manifest(self) -> dict:
        """
        Deterministic settings a stranger needs to tell an environment
        difference from a finding.

        Every field here is read back from the encoder that actually ran, not
        from a value this retriever was merely configured with — see the
        module docstring's batch_size note. `precision`, `pooling` and
        `max_length` come from SentenceTransformerEncoder's own verification
        against the loaded model (self.encoder.verification), when available,
        so this reports what was ACTUALLY used rather than what was requested.
        """
        manifest = {
            "model_name": self.encoder.model_name,
            "revision": self.encoder.revision,
            "precision": self.encoder.precision,
            "pooling": self.encoder.pooling,
            "max_length": self.encoder.max_length,
            "batch_size": self.encoder.batch_size,
            "normalize": self.normalize,
        }
        verification = getattr(self.encoder, "verification", None)
        if verification is not None:
            manifest["verification"] = verification
        # Recorded so two runs can be compared for bit-identical embeddings after the
        # fact, since the real encoder cannot be gated on determinism per run.
        if getattr(self, "_embedding_digest", None):
            manifest["embedding_sha256"] = self._embedding_digest
        return manifest


def _matrix_digest(mat: np.ndarray) -> str:
    """
    Hash of the embedding matrix, recorded in the run manifest.

    The amendment lists cross-process determinism as a control, and for the lexical
    arms it is enforced by a test that runs retrieval in separate subprocesses. The
    real encoder cannot be gated that way without doubling the wall clock of every
    run, and torch's reductions on MPS carry no bit-identity guarantee across
    processes or hardware. So the guarantee is made checkable after the fact rather
    than asserted: two runs that produced the same vectors carry the same digest, and
    a reader comparing two manifests can see immediately whether the embeddings moved.
    A rerun on different hardware that quietly produced different vectors would
    otherwise be invisible.
    """
    return hashlib.sha256(np.ascontiguousarray(mat, dtype=np.float32).tobytes()).hexdigest()


def _corpus_fingerprint(doc_ids: list[str], texts: list[str], max_length: int, model_name: str) -> str:
    """Content hash of the exact (doc_id, text) pairs a cache entry was built
    from, plus max_length (truncation changes the resulting vectors). Streamed
    rather than joined into one giant string, so this does not double the peak
    memory of a large corpus just to fingerprint it."""
    h = hashlib.sha256()
    # model_name is in the key, not only the revision. A reviewer demonstrated two
    # encoders with different names but the same revision string sharing a cache
    # entry, so the second never encoded anything and silently inherited the first's
    # vectors. Revision hashes are effectively unique per repository today, which is
    # why this was dormant rather than live, but a cache that can serve another
    # model's vectors is the wrong thing to leave lying around in a repository whose
    # point is that numbers correspond to a pinned artifact.
    h.update(model_name.encode("utf8"))
    h.update(b"\x00")
    h.update(str(max_length).encode("utf8"))
    for doc_id, text in zip(doc_ids, texts):
        h.update(doc_id.encode("utf8"))
        h.update(b"\x1f")
        h.update(text.encode("utf8"))
        h.update(b"\x1e")
    return h.hexdigest()


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

    VERIFICATION, NOT ASSUMPTION. The amendment (protocols/002-amendment-1-dense.md
    section 4) states max_length=256, mean pooling, L2-normalised — but a prior
    version of this class hard-coded `pooling = "mean"` as a class attribute and
    unconditionally overwrote `max_seq_length`, which would have silently masked
    a mismatch between the pinned revision and what the amendment assumed about
    it. This class instead reads pooling mode and the model's native
    max_seq_length off the loaded SentenceTransformer object and records both the
    requested and actual values in `self.verification`, raising if the model's
    own L2-normalisation module is absent (the amendment's "exact cosine ...
    L2-normalised" is not true unless something actually normalises).
    """

    precision = "float32"  # this repo never loads a quantised checkpoint

    def __init__(
        self,
        model_name: str,
        revision: str,
        max_length: int | None = 256,  # amendment section 4; see class docstring re: verification
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer  # lazy: see class docstring

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, revision=revision)

        native_max_length = self._model.max_seq_length
        if max_length is not None and max_length != native_max_length:
            self._model.max_seq_length = max_length
        self.max_length = self._model.max_seq_length

        pooling_mode = _find_pooling_mode(self._model)
        self.pooling = pooling_mode or "unknown"
        model_normalizes = _has_normalize_module(self._model)
        if not model_normalizes:
            # The amendment's "L2-normalised, as this model specifies" is a claim
            # about THIS pinned revision, checked here rather than assumed. If a
            # future revision drops its Normalize module, DenseRetriever's own
            # `normalize=True` step still L2-normalises independently (see
            # retrieve() above), so results are not silently wrong — but the
            # amendment's premise about the model itself would be, and that is
            # worth failing loudly on rather than reporting quietly.
            raise RuntimeError(
                f"{model_name}@{revision} has no built-in L2-normalisation module; "
                "amendment section 4 assumes this model specifies one. Verify the "
                "revision before proceeding — DenseRetriever's own normalize=True "
                "step still runs, so this is a mismatch to report, not silent."
            )

        # Recorded rather than trusted: this is what run_rung's manifest reports,
        # so a reviewer sees what was actually measured against the model object,
        # not what protocols/002-ladder.md guessed before anything was run.
        self.verification = {
            "requested_max_length": max_length,
            "native_max_length": native_max_length,
            "actual_max_length": self.max_length,
            "pooling_mode_from_model": self.pooling,
            "model_has_normalize_module": model_normalizes,
        }

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        # Separate from encode_documents because some models use an asymmetric
        # query/document convention (e.g. an "query: " prefix); this is the seam
        # where that convention would be applied, kept visible rather than buried
        # inside a single shared encode(). all-MiniLM-L6-v2 uses no such prefix
        # (verified: it ships no prompt templates), so both methods call the
        # same underlying encode() today — recorded here rather than silently
        # assumed, so a future model swap has an obvious place to add one.
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        )


def _find_pooling_mode(model) -> str | None:
    """Read the pooling mode off the model's own Pooling module rather than
    trusting a hard-coded string. Returns None if the model has no such module
    (some sentence-transformers models pool inside a Dense layer instead), so
    the caller can decide how to report that rather than this function
    guessing."""
    for module in model._modules.values():
        get_config = getattr(module, "get_config_dict", None)
        if get_config is None:
            continue
        config = get_config()
        if "pooling_mode" in config:
            return config["pooling_mode"]
    return None


def _has_normalize_module(model) -> bool:
    """Whether the loaded model pipeline includes a Normalize module, i.e.
    whether `.encode()` output is already L2-normalised by the model itself
    rather than by DenseRetriever's own _l2_normalize step."""
    return any(type(module).__name__ == "Normalize" for module in model._modules.values())
