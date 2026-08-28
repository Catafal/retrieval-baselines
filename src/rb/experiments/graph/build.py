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

NORMALISATION IS EXACT-STRING BY DEFAULT, matching §8.2's scoring. "U.S." and "United States"
are different nodes because the graph would treat them as different nodes; for 003 and 004 a
linker that merged them would have been a component neither experiment registered and neither
could attribute. Experiment 005 registers exactly that component and injects it through `link`,
which is why the merge is a parameter here rather than an edit.
"""

import numpy as np
from scipy import sparse

from rb.experiments.graph.extraction_score import normalise


def build(entities_by_doc: dict[str, list[str]], link=None):
    """
    Returns (nodes, doc_ids, incidence) where `incidence` is a documents x entities sparse
    matrix with 1 where a document contains an entity.

    `link` maps a surface form to the node key it belongs to. It defaults to `normalise`, which
    is 003's and 004's exact-string identity, so an unqualified call is the published arm.
    Experiment 005 passes a linker that resolves aliases to a canonical instead — see
    protocols/005-identity.md section 5. The default resolves HERE rather than in the signature
    because binding a module attribute at definition time is how 004's injected extractor was
    first defeated.

    The entity-entity adjacency is never materialised. incidence.T @ incidence is an
    entities x entities matrix, and on a corpus with a hub entity that product is dense
    enough to exhaust memory. Every operation below is expressed as two sparse
    matrix-vector products against `incidence` instead, which is the same walk without ever
    holding the square matrix.
    """
    key = link or normalise
    doc_ids = sorted(entities_by_doc)
    index: dict[str, int] = {}
    rows, cols = [], []
    for r, doc in enumerate(doc_ids):
        seen = set()
        for raw in entities_by_doc[doc]:
            e = key(raw)
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


def degrees(incidence) -> np.ndarray:
    """
    Weighted degree of each entity in the co-occurrence graph, WITHOUT the self-loop.

    The adjacency implied by two hops is A = B^T B, whose diagonal is each entity's own
    document frequency — an entity co-occurring with ITSELF once per document it appears in.
    That is not a co-occurrence with another entity and it must not count toward degree, or a
    hub both receives more mass and leaks less of what it holds.

    Computed as row sums of B^T B minus that diagonal, without ever forming the square matrix.
    """
    df = np.asarray(incidence.sum(axis=0)).ravel()          # documents per entity = diagonal
    row_sums = incidence.T @ (incidence @ np.ones(incidence.shape[1]))
    return row_sums - df


def personalized_pagerank(incidence, seed_weights: np.ndarray, damping: float = 0.5,
                          iterations: int = 50, tol: float = 1e-10,
                          deg: np.ndarray | None = None) -> np.ndarray:
    """
    Personalized PageRank over the entity graph induced by co-occurrence.

    CORRECTED 2026-08-21. The first version was not a PageRank at all: it applied
    `B^T (B rank)` and rescaled the whole vector to sum 1. That is damped power iteration on
    an UNNORMALISED co-occurrence matrix, which converges toward a degree-dominated
    eigenvector rather than a random walk, and it made the arm score below a baseline that
    did no propagation whatsoever. See protocols/003-amendment-3-ppr-correction.md.

    What makes it a walk is that mass leaving a node is divided by that node's degree, so a
    hub's individual edges are discounted precisely because it has many. Two corrections:

      1. Divide by degree before propagating   -> the transition is row-stochastic.
      2. Subtract the self-loop term           -> A = B^T B has diagonal df(entity), an
                                                  entity co-occurring with itself, which is
                                                  stickiness proportional to popularity.

    A @ x is still never materialised: it is B^T (B x) minus the diagonal contribution.

    `damping` is the RESTART probability, matching HippoRAG's stated convention, and stays
    at the 0.5 they report tuning to. Iterations raised 20 -> 50 because a correctly
    normalised walk converges more slowly than the rescaled version appeared to; `tol` still
    exits early once it settles, so the extra ceiling costs nothing when it is not needed.
    """
    total = seed_weights.sum()
    if total <= 0:
        return np.zeros(incidence.shape[1], dtype=np.float64)
    restart = seed_weights / total
    if deg is None:
        deg = degrees(incidence)
    df = np.asarray(incidence.sum(axis=0)).ravel()
    # Entities with no co-occurrence partner are dangling: a walk arriving there has nowhere
    # to go. Their mass returns to the restart vector rather than being silently dropped,
    # which would leak probability and quietly renormalise the result.
    dangling = deg <= 0
    safe_deg = np.where(dangling, 1.0, deg)

    rank = restart.copy()
    for _ in range(iterations):
        weighted = rank / safe_deg
        weighted[dangling] = 0.0
        spread = incidence.T @ (incidence @ weighted) - df * weighted
        spread += rank[dangling].sum() * restart      # dangling mass restarts
        nxt = damping * restart + (1.0 - damping) * spread
        s = nxt.sum()
        if s > 0:
            nxt /= s                                   # guard against float drift only
        if np.abs(nxt - rank).sum() < tol:
            return nxt
        rank = nxt
    return rank


def score_documents(incidence, rank: np.ndarray) -> np.ndarray:
    """A document's score is the PPR mass its entities carry. Documents holding several
    highly-ranked entities outrank documents holding one."""
    return incidence @ rank
