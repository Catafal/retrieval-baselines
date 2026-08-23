"""
§8.1 harness closure — the GATE that checks our indexing, scoring and pooling against an external
number computed under matching conditions.

    make reproduce-003-closure

WHAT IT CHECKS. Run full BM25 over a nested 1,000-question subset of the distractor pool and
compare against HippoRAG Table 2's BM25 row (R@2 55.4, R@5 72.2) at 0.05 absolute tolerance. If
our BM25 cannot land near a published BM25 under the same construction, nothing downstream is
worth reading, so failure halts before anything is written.

WHY THIS FILE EXISTS NOW. `results/003/closure-8-1.json` was committed with NO producing code —
the most serious of five such artifacts, because it is the only one that GATES. A control nobody
can re-run is not a control. It was found by a pre-publication review seat, not by the three
audits that preceded it.

THE SUBSET RULE IS STATED HERE BECAUSE THE ORIGINAL IS NOT RECOVERABLE. The committed artifact
reports 9,811 pooled passages. Taking the first 1,000 questions in file order gives 9,769, and
sorted by question id gives 9,755; neither matches, and no code recording the original draw
exists. This module therefore fixes the rule explicitly — first 1,000 question ids in sorted
order — and republishes what that produces, rather than keeping a number no one can reconstruct.
"""

import json
import time
from pathlib import Path

from rb import datasets, metrics
from rb.experiments.graph import pool
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.experiments.ladder.retrievers.lexical import full_bm25

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"

SUBSET_QUESTIONS = 1000
TOLERANCE = 0.05
# HippoRAG Table 2, BM25 row, transcribed from protocols/003-graph-arm.md §2 as tagged. A literal
# because it is prior art: a reference that recomputed itself from our run would not be one.
PUBLISHED_BM25 = {"recall_2": 0.554, "recall_5": 0.722}


def subset_pool():
    """The nested subset, by a rule fixed in this module rather than remembered."""
    ctx = pool.load_distractor_context()
    keep = sorted(ctx)[:SUBSET_QUESTIONS]
    sub = {q: ctx[q] for q in keep}
    corpus = datasets.load_corpus("hotpotqa")
    titles = datasets.load_titles("hotpotqa")
    corpus_sub, _ = pool.build(corpus, titles, sub)
    queries = datasets.load_queries("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")
    q = {i: queries[i] for i in keep if i in queries and i in qrels}
    return corpus_sub, q, {i: qrels[i] for i in q}


def run() -> dict:
    t0 = time.perf_counter()
    corpus, queries, qrels = subset_pool()
    arm = full_bm25()
    run_dict = arm.retrieve(corpus, queries, 100)
    scored = metrics.score_ranked(qrels, run_dict, GRAPH_MEASURES)
    ours = {m: round(metrics.mean([scored[q][m] for q in scored]), 4)
            for m in sorted(GRAPH_MEASURES)}
    deltas = {m: round(abs(ours[m] - v), 4) for m, v in PUBLISHED_BM25.items()}
    return {
        "corpus_passages": len(corpus),
        "queries": len(queries),
        "subset_rule": f"first {SUBSET_QUESTIONS} question ids in sorted order",
        "ours": ours,
        "published_bm25": PUBLISHED_BM25,
        "tolerance": TOLERANCE,
        "deltas": deltas,
        "passed": all(d <= TOLERANCE for d in deltas.values()),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    r = run()
    if not r["passed"]:
        raise RuntimeError(f"closure 8.1 FAILED: {r}. Nothing downstream is worth reading.")
    (OUT / "closure-8-1.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
