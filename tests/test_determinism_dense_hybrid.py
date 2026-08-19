"""
Cross-process determinism for the dense and hybrid rungs, amendment section 9.

Not a new source of nondeterminism (DenseRetriever already sorts doc_ids and
q_ids rather than trusting dict order — see dense.py's module docstring for
the corpus-order bug this change fixed), but the amendment lists cross-process
determinism as its own control, and 001/002's own history is that this class
of defect (BM25's set-iteration bug, 63/300 SciFact queries disagreeing
between processes) was invisible to every same-process test that existed at
the time. So this is checked the same way tests/test_lexical.py's analogous
test checks it: run the retriever in a fresh `python` subprocess, several
times, and diff byte-for-byte — a same-process test cannot see a bug whose
cause is "does this depend on something Python does not guarantee stable
across processes" by construction.

Uses stub encoders/retrievers, not the real sentence-transformers model: the
property under test is this repo's own ranking and fusion code, not whether a
transformer forward pass is bitwise reproducible across processes (a separate
and much more expensive question this repo does not need to answer, since the
manifest already pins model + revision + library version instead).
"""

import subprocess
import sys


def _run_three_times(script: str) -> list[str]:
    return [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        ).stdout
        for _ in range(3)
    ]


DENSE_SCRIPT = """
import json
import numpy as np
from rb.experiments.ladder.retrievers.dense import DenseRetriever

class Encoder:
    model_name, revision, precision, pooling, max_length, batch_size = (
        "stub", "deadbeef", "float32", "none", 32, 32,
    )
    def encode_documents(self, texts):
        # Ties by construction: every document gets an identical vector, so
        # ranking falls through entirely to the doc-id tie-break, which is
        # exactly the case a hash-order dependency would corrupt.
        return np.ones((len(texts), 4))
    def encode_queries(self, texts):
        return np.ones((len(texts), 4))

corpus = {f"d{i:02d}": f"document {i}" for i in range(25)}
queries = {"q": "query"}
run = DenseRetriever(Encoder(), normalize=False).retrieve(corpus, queries, top_k=25)
print(json.dumps(list(run["q"])))
"""

HYBRID_SCRIPT = """
import json
from rb.experiments.ladder.retrievers.hybrid import HybridRetriever

class Fixed:
    def __init__(self, name, ranking):
        self.name = name
        self.ranking = ranking
    def retrieve(self, corpus, queries, top_k):
        return {q: {d: float(len(docs) - i) for i, d in enumerate(docs[:top_k])}
                for q, docs in self.ranking.items()}

# Deliberately tied RRF scores for several documents (same rank in both
# components after the first two), so the fused ranking depends entirely on
# the tie-break rather than on any real score separation.
docs = [f"d{i:02d}" for i in range(20)]
lexical = Fixed("lex", {"q": docs})
dense = Fixed("dense", {"q": list(reversed(docs))})
run = HybridRetriever(lexical, dense, candidate_k=20).retrieve({}, {"q": ""}, top_k=20)["q"]
print(json.dumps(list(run)))
"""


def test_dense_retrieval_is_identical_across_processes():
    outputs = _run_three_times(DENSE_SCRIPT)
    assert outputs[0] == outputs[1] == outputs[2]


def test_hybrid_retrieval_is_identical_across_processes():
    outputs = _run_three_times(HYBRID_SCRIPT)
    assert outputs[0] == outputs[1] == outputs[2]
