"""
The graph arm — protocols/003-graph-arm.md §2, implementing the shared `Retriever` seam.

RETRIEVAL, in HippoRAG's shape. Entities are recognised in the query, linked to graph nodes
by exact normalised string, and used to seed a personalized PageRank walk over the entity
graph. A document scores by the PPR mass its entities carry. Reaching a document the query
never names is the entire point: the walk arrives there through a shared entity, which is the
mechanism 002 measured a loss against and could not isolate.

LINKING IS EXACT STRING, not embedding similarity. HippoRAG links query entities to nodes with
a retrieval encoder; doing that here would smuggle a dense component into an arm whose whole
claim is to be deterministic, and would make a loss unattributable between the graph and the
encoder. The protocol says spaCy NER plus string linking, and this is that.

NOTHING IS SCORED BY IMPORTING THIS. Building the graph is a `fit` step the caller invokes
explicitly, and the protocol forbids a scored run before `protocol-003` is tagged.
"""

import numpy as np

from rb.experiments.graph import build as kg
from rb.experiments.graph.extraction_score import normalise
from rb.experiments.graph.extractor import extract_many, manifest, node_strings

# Separation below the scorer's resolution. pytrec_eval compares at reduced precision, so two
# documents split only by a tie-break must still be a real rank apart to it; run_rung
# re-encodes ranks for exactly this reason. The epsilon here only satisfies the Retriever
# contract's strict-ordering requirement, matching what the lexical rung does.
_TIE = 1e-9


class GraphRetriever:
    """spaCy-extracted entity graph, retrieved by personalized PageRank."""

    name = "graph-spacy-ppr"

    def __init__(self, damping: float = 0.5, use_specificity: bool = True):
        # Both fixed by the protocol rather than tuned here. HippoRAG reports 0.5 and reports
        # that removing specificity costs it 4.2 R@2 on HotpotQA; tuning either on the corpus
        # being measured is how a result gets manufactured.
        self.damping = damping
        self.use_specificity = use_specificity
        self._fitted = None

    def fit(self, corpus: dict[str, str]) -> dict:
        """
        Extract entities over the corpus and build the graph.

        Separate from `retrieve` because it is the expensive half and its cost is reported on
        its own: the protocol requires build cost and query cost stated separately, since the
        other arms have near-zero build cost and an arm whose advantage costs an extraction
        pass must show that in the same table as its nDCG.
        """
        raw = extract_many(corpus)
        entities = {doc: node_strings(ents) for doc, ents in raw.items()}
        nodes, doc_ids, incidence = kg.build(entities)
        self._fitted = {
            "nodes": nodes,
            "node_index": {n: i for i, n in enumerate(nodes)},
            "doc_ids": doc_ids,
            "doc_id_set": frozenset(doc_ids),
            "incidence": incidence,
            "specificity": kg.node_specificity(incidence),
            # Computed once. `degrees` is a pure function of the incidence matrix, which does
            # not change between queries, so recomputing it per query was an O(nnz) pass whose
            # only effect was to make the arm's reported query cost wrong.
            "degrees": kg.degrees(incidence),
            "entities": entities,
        }
        return {
            "documents": len(doc_ids),
            "nodes": len(nodes),
            "edges_as_incidence_nnz": int(incidence.nnz),
            "documents_without_entities": int((np.asarray(incidence.sum(axis=1)).ravel() == 0).sum()),
            **manifest(),
        }

    @property
    def entities_by_doc(self) -> dict[str, list[str]]:
        """The extractor's output, for §8.2's diagnostic and §8.3's gate. Exposed rather than
        recomputed so both controls describe the graph that was actually built."""
        if self._fitted is None:
            raise RuntimeError("fit() first")
        return self._fitted["entities"]

    def _seed(self, query: str) -> np.ndarray:
        """Query entities, linked by exact normalised string, weighted by node specificity.

        An entity in half the corpus says almost nothing about which document is wanted; one
        in two documents nearly identifies the pair. Weighting the restart vector is where
        that asymmetry enters the walk."""
        f = self._fitted
        seed = np.zeros(len(f["nodes"]), dtype=np.float64)
        # THE SAME whitelist filter the document side uses, via the same function.
        # It previously did not, and the asymmetry was real: 8.0% of linked seeds entered
        # through the unfiltered path and 1.85% of queries were seeded ONLY by excluded types.
        # entity_types.py's docstring asserted the two sides were treated alike; nothing
        # enforced it. Routed through node_strings rather than given a second filter of its
        # own, because a second filter is how the two sides drift apart again.
        for text in node_strings(_query_entities(query)):
            i = f["node_index"].get(normalise(text))
            if i is not None:
                seed[i] += f["specificity"][i] if self.use_specificity else 1.0
        return seed

    def retrieve(self, corpus: dict[str, str], queries: dict[str, str],
                 top_k: int) -> dict[str, dict[str, float]]:
        if self._fitted is None:
            raise RuntimeError("fit(corpus) must be called before retrieve()")
        f = self._fitted
        # Refuse a corpus this arm was not fitted on. Every document identity below comes from
        # the fitted state, so a caller that fits on one snapshot and retrieves against another
        # would be silently scored against the wrong document set. Raising beats re-fitting:
        # scoring the wrong corpus is the failure being prevented, and a silent re-fit would
        # hide it.
        if corpus and set(corpus) != f["doc_id_set"]:
            raise RuntimeError(
                f"retrieve() was given {len(corpus)} documents but the arm was fitted on "
                f"{len(f['doc_id_set'])}. Refusing to score against a corpus it was not built from."
            )
        out: dict[str, dict[str, float]] = {}
        for qid in sorted(queries):
            rank = kg.personalized_pagerank(f["incidence"], self._seed(queries[qid]),
                                            self.damping, deg=f["degrees"])
            scores = kg.score_documents(f["incidence"], rank)
            # A query whose entities match no node retrieves nothing. That is the honest
            # outcome — the graph cannot reach anything from a seed it does not have — and it
            # must not be disguised as a ranking of arbitrary documents.
            nz = np.flatnonzero(scores > 0)
            if nz.size == 0:
                out[qid] = {}
                continue
            order = nz[np.argsort(-scores[nz], kind="stable")][:top_k]
            # Document-id tie-break, then a strictly decreasing epsilon, satisfying the
            # Retriever contract without pretending the gaps are meaningful.
            ranked = sorted(((f["doc_ids"][i], float(scores[i])) for i in order),
                            key=lambda kv: (-kv[1], kv[0]))
            out[qid] = {d: s - rank_i * _TIE for rank_i, (d, s) in enumerate(ranked)}
        return out


def _query_entities(query: str):
    """Entities in one query. Split out so tests can substitute a stub and exercise the walk
    without loading a model."""
    from rb.experiments.graph.extractor import extract
    return extract(query)
