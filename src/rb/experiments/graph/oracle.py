"""
The ORACLE-EXTRACTOR ceiling for 003's graph arm.

    make reproduce-003-oracle

WHAT IT MEASURES. Replace spaCy entirely with a perfect extractor AND a perfect linker: a
document's entities are exactly the corpus document TITLES that occur in its text. Titles are what
the coverage class is defined on and what a bridge entity IS on this corpus, so this is the graph
this arm would build if extraction and linking were solved. No NER, no whitelist, no normalisation
mismatch, no `Kiss and Tell` span problem.

WHY IT MATTERS MORE THAN IT LOOKS. The entry's account of the arm's failure is that it cannot link
queries to nodes, and the obvious next move is a better extractor (experiment 004). This run puts a
ceiling on how much that could possibly buy: it is the best case for extraction, and the arm still
loses to BM25 by roughly 21 R@2 points and still returns nothing for about one query in five. So
the binding constraint is not only extraction quality.

WHY THIS FILE EXISTS NOW. `results/003/oracle-entity-graph.json` was already committed, and NO code
in the repository produced it. It appeared in no protocol, no amendment, no correction record and
no entry. It is the seventeenth defect and the fifth of its exact class, found by a pre-publication
review seat that cloned the repo and read the results directory. Reconstructed here and checked
against the committed values before being adopted; had it not reproduced them, the artifact would
have been removed rather than quietly redefined.

NOT A REGISTERED ARM. It is a diagnostic ceiling, not a competitor: it uses gold corpus titles,
which a real system does not have. It is reported as such and gates nothing.
"""

import json
import time
from pathlib import Path

from rb import datasets, metrics
from rb.experiments.graph import build as kg
from rb.experiments.graph import pool
from rb.experiments.graph.extraction_score import normalise
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.experiments.graph.retriever import GraphRetriever

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"
TOP_K = 100


def oracle_entities(corpus: dict[str, str], titles: dict[str, str]) -> dict[str, list[str]]:
    """doc_id -> the corpus titles that occur in that document's text.

    Matched on the normalised string, the same key the graph itself uses, so a title the graph
    could not have keyed on is not credited here either.

    BY N-GRAM LOOKUP, not by substring scan. The obvious form -- test every title against every
    passage -- is 66,304 x 66,581 checks and does not finish in any time worth waiting for; the
    first version of this module was written that way and had to be killed after twelve minutes.
    Instead each passage's word n-grams are looked up in a set of titles, which is linear in the
    corpus and bounded by the longest title.
    """
    by_norm: dict[str, str] = {}
    for t in titles.values():
        n = normalise(t)
        if n:
            by_norm.setdefault(n, t)
    longest = max(len(n.split()) for n in by_norm)

    out = {}
    for doc, text in corpus.items():
        words = normalise(text).split()
        found = {}
        for i in range(len(words)):
            for k in range(1, min(longest, len(words) - i) + 1):
                cand = " ".join(words[i:i + k])
                t = by_norm.get(cand)
                if t is not None:
                    found[cand] = t
        out[doc] = list(found.values())
    return out


def query_titles(query: str, by_norm: dict[str, str]) -> list[str]:
    """The same oracle on the query side: titles occurring in the question text."""
    words = normalise(query).split()
    longest = max(len(n.split()) for n in by_norm)
    found = {}
    for i in range(len(words)):
        for k in range(1, min(longest, len(words) - i) + 1):
            cand = " ".join(words[i:i + k])
            t = by_norm.get(cand)
            if t is not None:
                found[cand] = t
    return list(found.values())


def main() -> None:
    t0 = time.perf_counter()
    ctx = pool.load_distractor_context()
    corpus_all = datasets.load_corpus("hotpotqa")
    titles_all = datasets.load_titles("hotpotqa")
    corpus, resolved = pool.build(corpus_all, titles_all, ctx)
    titles = {d: t for t, d in resolved.items()}
    queries = datasets.load_queries("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")
    queries = {q: t for q, t in queries.items() if q in qrels}

    ents = oracle_entities(corpus, titles)
    nodes, doc_ids, incidence = kg.build(ents)

    arm = GraphRetriever()
    arm._fitted = {
        "nodes": nodes, "node_index": {n: i for i, n in enumerate(nodes)},
        "doc_ids": doc_ids, "doc_id_set": frozenset(doc_ids), "incidence": incidence,
        "specificity": kg.node_specificity(incidence), "degrees": kg.degrees(incidence),
        "entities": ents,
    }
    # The query side is linked by the same oracle: titles occurring in the question text.
    #
    # THE LABEL MUST BE WHITELISTED. `_seed` routes query entities through
    # `extractor.node_strings`, which drops any entity whose label is not in WHITELIST. Labelling
    # these "ORACLE" made every seed vector zero and every query retrieve nothing, silently -- the
    # arm returns an empty list for an unseeded query by design, so a fully broken oracle looks
    # exactly like a very bad one. Caught by a review seat reading the code rather than by the
    # numbers, because the numbers never got written.
    import rb.experiments.graph.retriever as rmod
    by_norm = {normalise(t): t for t in titles.values() if normalise(t)}
    rmod._query_entities = lambda q: [(t, "ORG") for t in query_titles(q, by_norm)]
    run = arm.retrieve(corpus, queries, TOP_K)

    scored = metrics.score_ranked(qrels, run, GRAPH_MEASURES)
    ranked = {m: round(metrics.mean([scored[q][m] for q in scored]), 4)
              for m in sorted(GRAPH_MEASURES)}
    payload = {
        "oracle_entities": ("corpus document titles present in the passage - a perfect extractor "
                            "AND linker"),
        "status": ("DIAGNOSTIC CEILING, not a registered arm and not a competitor: it uses gold "
                   "corpus titles, which a real system does not have. Gates nothing."),
        "ranked": ranked,
        "empty_results": sum(1 for q in run if not run[q]),
        "queries": len(run),
        "mean_entities_per_doc": round(sum(len(v) for v in ents.values()) / len(ents), 2),
        "nodes": len(nodes),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT / "oracle-entity-graph.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "status"}, indent=2))


if __name__ == "__main__":
    main()
