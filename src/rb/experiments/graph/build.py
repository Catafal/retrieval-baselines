"""
The knowledge graph — protocols/003-graph-arm.md §2.

SHAPE, following HippoRAG (Gutiérrez et al., NeurIPS 2024). Nodes are entity strings. Two
entities are adjacent when they co-occur in a passage, which is the only relation a
named-entity recogniser can supply: spaCy does NER, not OpenIE, so there are no typed
relations to build edges from. That difference is stated in the protocol rather than papered
over, and it is why no published retrieval row can gate this arm.

WHY CO-OCCURRENCE IS ENOUGH FOR THE QUESTION BEING ASKED. The experiment tests whether a
graph helps where the bridge entity is absent from the query. Reaching an unnamed document
requires exactly one thing: a path from an entity the query DOES name to the document that
holds the answer. Co-occurrence edges provide that path. A richer relation set would give
better paths, which is precisely what experiment 004 tests by swapping the extractor.

NORMALISATION IS EXACT-STRING, matching §8.2's scoring. "U.S." and "United States" are
different nodes because the graph would treat them as different nodes; a linker that merged
them would be a component this experiment has not registered and cannot attribute.
"""

import numpy as np
from scipy import sparse

from rb.experiments.graph.extraction_score import normalise


def build(entities_by_doc: dict[str, list[str]]):
    """
    Returns (nodes, doc_ids, incidence) where `incidence` is a documents x entities sparse
    matrix with 1 where a document contains an entity.

    The entity-entity adjacency is never materialised. incidence.T @ incidence is an
    entities x entities matrix, and on a corpus with a hub entity that product is dense
    enough to exhaust memory. Every operation below is expressed as two sparse
    matrix-vector products against `incidence` instead, which is the same walk without ever
    holding the square matrix.
    """
    doc_ids = sorted(entities_by_doc)
    index: dict[str, int] = {}
    rows, cols = [], []
    for r, doc in enumerate(doc_ids):
        seen = set()
        for raw in entities_by_doc[doc]:
            e = normalise(raw)
            if not e or e in seen:
                continue
            seen.add(e)
            if e not in index:
                index[e] = len(index)
            rows.append(r)
            cols.append(index[e])
    nodes = sorted(index, key=index.get)
    incidence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(doc_ids), len(nodes)),
    )
    return nodes, doc_ids, incidence


def node_specificity(incidence) -> np.ndarray:
    """
    1 / number of documents containing the entity.

    HippoRAG's own ablation shows this matters: removing node specificity drops its HotpotQA
    R@2 from 60.5 to 56.3. The reason is that an entity in half the corpus carries almost no
    information about which document is wanted, while an entity in two documents nearly
    identifies the pair — which is exactly the bridge case this experiment is about.
    """
    df = np.asarray(incidence.sum(axis=0)).ravel()
    return np.divide(1.0, df, out=np.zeros_like(df), where=df > 0)


def personalized_pagerank(incidence, seed_weights: np.ndarray, damping: float = 0.5,
                          iterations: int = 20, tol: float = 1e-10) -> np.ndarray:
    """
    PPR over the entity graph induced by co-occurrence, restarting at the query's entities.

    `damping` is 0.5, the value HippoRAG reports tuning to. Kept rather than re-tuned: tuning
    a hyperparameter on the corpus being measured is how a result gets manufactured, and no
    tuning run is registered in this protocol.

    One step of the walk is entities -> documents -> entities, both hops sparse, so the
    entity-entity matrix is never formed. Iteration stops on convergence rather than always
    running the full count, which matters because this runs once per query.
    """
    total = seed_weights.sum()
    if total <= 0:
        return np.zeros(incidence.shape[1], dtype=np.float64)
    restart = seed_weights / total
    rank = restart.copy()
    for _ in range(iterations):
        spread = incidence.T @ (incidence @ rank)      # entities -> docs -> entities
        s = spread.sum()
        if s > 0:
            spread /= s
        nxt = damping * restart + (1.0 - damping) * spread
        if np.abs(nxt - rank).sum() < tol:
            return nxt
        rank = nxt
    return rank


def score_documents(incidence, rank: np.ndarray) -> np.ndarray:
    """A document's score is the PPR mass its entities carry. Documents holding several
    highly-ranked entities outrank documents holding one."""
    return incidence @ rank
