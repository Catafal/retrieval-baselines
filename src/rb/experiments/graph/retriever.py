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

    def __init__(self, damping: float = 0.5, use_specificity: bool = True,
                 extract_docs=None, extract_query=None, link=None, name: str | None = None):
        # damping and use_specificity are both fixed by the protocol rather than tuned here.
        # HippoRAG reports 0.5 and reports that removing specificity costs it 4.2 R@2 on
        # HotpotQA; tuning either on the corpus being measured is how a result gets manufactured.
        self.damping = damping
        self.use_specificity = use_specificity
        self._fitted = None

        # INJECTED FOR EXPERIMENT 004, DEFAULTING TO 003's BEHAVIOUR EXACTLY. Both default to
        # None and resolve to the spaCy functions, so an unqualified GraphRetriever() is the
        # arm 003 published and its numbers cannot move.
        #
        # BOTH sides are injected together, never one. `_seed` links query entities to graph
        # nodes by exact normalised string, so a graph built by one extractor and seeded by
        # another fails to link on span-boundary differences alone — lowering the seed rate and
        # raising the empty rate for a reason that has nothing to do with graph traversal, while
        # looking exactly like a confirmation of 003's finding. See
        # protocols/004-amendment-3-query-extraction.md.
        if (extract_docs is None) != (extract_query is None):
            raise ValueError(
                "extract_docs and extract_query must be supplied together: a graph built by one "
                "extractor and seeded by another measures extractor mismatch, not extraction "
                "quality. See protocols/004-amendment-3-query-extraction.md."
            )
        # Stored as None rather than resolved to the spaCy functions here. Binding the default
        # at construction captures the module attribute as it was, so a test that monkeypatches
        # `_query_entities` to a stub would silently keep hitting the real spaCy path — which is
        # exactly what happened, and two existing tests caught it. Resolved at call time instead.
        self._extract_docs = extract_docs
        self._extract_query = extract_query

        # INJECTED FOR EXPERIMENT 005 — protocols/005-identity.md section 5. Maps a surface form
        # to the node key it belongs to; None is 003's and 004's exact-string identity, so an
        # unqualified GraphRetriever() is still the published arm.
        #
        # ONE linker, used by BOTH sides. The extractor needed a both-or-neither guard because it
        # is two functions that can disagree. Identity is one function reached from two places, so
        # storing it once and passing it to build() from fit() makes the mismatch unrepresentable
        # rather than merely forbidden. A graph keyed by one identity and seeded by another is not
        # an experiment, and here it cannot be constructed.
        self._link = link
        if name:
            self.name = name

    def fit(self, corpus: dict[str, str]) -> dict:
        """
        Extract entities over the corpus and build the graph.

        Separate from `retrieve` because it is the expensive half and its cost is reported on
        its own: the protocol requires build cost and query cost stated separately, since the
        other arms have near-zero build cost and an arm whose advantage costs an extraction
        pass must show that in the same table as its nDCG.
        """
        raw = (self._extract_docs or extract_many)(corpus)
        entities = {doc: node_strings(ents) for doc, ents in raw.items()}
        nodes, doc_ids, incidence = kg.build(entities, link=self._link)
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
        #
        # DEDUPLICATED ON THE GRAPH'S OWN KEY, the normalised string. `node_strings` dedupes by
        # SURFACE FORM, so "U.S." and "U.S" survive it as two entries and then normalise to one
        # node — which used to add that node's specificity twice, while a document containing
        # both counts it once (build.build dedupes after normalise). The same document/query
        # asymmetry as the whitelist bug, one layer down. Measured at 3 of 7,405 queries. Fixed
        # here rather than in `node_strings`, which is deliberately surface-form-level and is
        # shared with the §8.2 diagnostic, where collapsing surface forms would change what the
        # extraction score measures.
        extract_q = self._extract_query or _query_entities
        # THE SAME identity the graph was built under, from the same stored linker.
        key = self._link or normalise
        for node in {key(text) for text in node_strings(extract_q(query))}:
            i = f["node_index"].get(node)
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
