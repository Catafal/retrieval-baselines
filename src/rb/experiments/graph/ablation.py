"""
§8's scoring ablation: summed versus mean document scoring.

    make reproduce-003-ablation

The registered arm SUMS the PPR mass of a document's entities, which is what HippoRAG does.
Dividing by the document's entity count instead is the obvious alternative, and substituting it
after seeing results would be a method change rather than an ablation. So it is reported beside
the registered scoring, never in place of it, so a reader can see how much of the arm's failure
belongs to the scoring function rather than to the graph.

`results/003/scoring-ablation.json` was committed with no producing code. This is that producer.
"""

import json
import time
from pathlib import Path

import numpy as np

from rb import datasets, metrics
from rb.experiments.graph import build as kg
from rb.experiments.graph import pool, run_controls
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.experiments.graph.retriever import GraphRetriever

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"


def main() -> None:
    t0 = time.perf_counter()
    ctx = pool.load_distractor_context()
    corpus_all = datasets.load_corpus("hotpotqa")
    titles = datasets.load_titles("hotpotqa")
    corpus, _ = pool.build(corpus_all, titles, ctx)
    queries = datasets.load_queries("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")
    queries = {q: t for q, t in queries.items() if q in qrels}

    ents = run_controls.extract_pool()
    nodes, doc_ids, incidence = kg.build(ents)
    arm = GraphRetriever()
    arm._fitted = {
        "nodes": nodes, "node_index": {n: i for i, n in enumerate(nodes)},
        "doc_ids": doc_ids, "doc_id_set": frozenset(doc_ids), "incidence": incidence,
        "specificity": kg.node_specificity(incidence), "degrees": kg.degrees(incidence),
        "entities": ents,
    }
    f = arm._fitted
    # Entities per document: the divisor the mean variant differs by, and nothing else.
    per_doc = np.asarray(incidence.sum(axis=1)).ravel()
    safe = np.where(per_doc > 0, per_doc, 1.0)

    runs = {"sum": {}, "mean": {}}
    for qid in sorted(queries):
        rank = kg.personalized_pagerank(f["incidence"], arm._seed(queries[qid]),
                                        arm.damping, deg=f["degrees"])
        base = kg.score_documents(f["incidence"], rank)
        for name, scores in (("sum", base), ("mean", base / safe)):
            nz = np.flatnonzero(scores > 0)
            if nz.size == 0:
                runs[name][qid] = {}
                continue
            order = nz[np.argsort(-scores[nz], kind="stable")][:100]
            ranked = sorted(((f["doc_ids"][i], float(scores[i])) for i in order),
                            key=lambda kv: (-kv[1], kv[0]))
            # SCORED BY RANK POSITION, not by the raw mass. Every other path in this repository
            # scores the retriever's ORDER: run_rung re-encodes ranks before handing them to
            # pytrec_eval, and the analyses re-score the committed per_query.jsonl the same way.
            # Scoring the masses directly here gave 0.2151 against the arm's published 0.2148,
            # because a PPR mass is around 1e-5 and the strict-ordering epsilon is 1e-9, which is
            # large enough relative to the gaps to reorder documents. An ablation whose baseline
            # column does not reproduce the arm it ablates is measuring something else.
            runs[name][qid] = {d: float(len(ranked) - i) for i, (d, _) in enumerate(ranked)}

    out = {}
    for name, run in runs.items():
        sc = metrics.score_ranked(qrels, run, GRAPH_MEASURES)
        out[name] = {m: round(metrics.mean([sc[q][m] for q in sc]), 4)
                     for m in sorted(GRAPH_MEASURES)}
    out["note"] = (
        "ABLATION ONLY. The registered arm keeps SUMMED scoring, which is what HippoRAG does; "
        "substituting mean after seeing results would be a method change. Reported so a reader "
        "can see how much of the arm's failure is the scoring function."
    )
    out["seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "scoring-ablation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("sum", "mean")}, indent=2))


if __name__ == "__main__":
    main()
