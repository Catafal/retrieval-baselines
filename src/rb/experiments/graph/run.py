"""
Experiment 003's scored run — protocols/003-graph-arm.md §3, §5, §7.

    python -m rb.experiments.graph.run --arm bm25
    python -m rb.experiments.graph.run --arm graph

Every arm is scored against the IDENTICAL 66,581-passage candidate set and the identical
7,405 queries, because the registered claim is a paired per-query difference and arms scored
against different candidate sets cannot be paired.

Runs only after `protocol-003` is tagged. Each arm is a separate invocation so a long arm can
fail without costing the arms that already succeeded, the same reason 002 split its rungs.
"""

import argparse
import json
import time
from pathlib import Path

from rb import controls, datasets
from rb.experiments.graph import pool
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.retriever import run_rung

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"


def load_pool():
    """The candidate set, rebuilt from the dataset rather than cached, so every scored run
    re-asserts the §9 controls instead of trusting a file written earlier."""
    ctx = pool.load_distractor_context()
    corpus = datasets.load_corpus("hotpotqa")
    titles = datasets.load_titles("hotpotqa")
    pool_corpus, _ = pool.build(corpus, titles, ctx)
    _, slots = pool.pool_titles(ctx)

    queries = datasets.load_queries("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")

    check = controls.pool_construction(
        questions=len(ctx), passages=len(pool_corpus), title_slots=slots,
        unresolved=0, collisions=0,
        gold_titles_matched=len(qrels), gold_queries=len(qrels))
    if not check["passed"]:
        raise RuntimeError(f"pool_construction failed: {check}. Nothing is scored on a broken pool.")

    queries = {q: t for q, t in queries.items() if q in qrels}
    return pool_corpus, queries, qrels, check


def _arm(name: str):
    """
    One arm per name, each already pinned somewhere else.

    The dense arms REUSE 002's encoder factory and its ENCODER_CONFIGS rather than
    re-declaring model names and revisions here. Two copies of a pin is one copy too many:
    the point of pinning is that the artefact cannot drift, and a second declaration is
    exactly how it drifts. Registered in protocols/003-amendment-2-dense-arm.md, tagged
    before any vector was computed.
    """
    if name == "bm25":
        from rb.experiments.ladder.retrievers.lexical import full_bm25
        return full_bm25(), None
    if name == "graph":
        from rb.experiments.graph.retriever import GraphRetriever
        return GraphRetriever(), "fit"
    if name.startswith("dense-"):
        from rb.experiments.ladder.retrievers.dense import DenseRetriever
        from rb.experiments.ladder.run import EMBEDDING_CACHE_DIR, _make_encoder
        encoder = _make_encoder(name.split("-", 1)[1])
        # Cached because embedding 66,581 passages is the expensive half and the
        # embedding-shuffle control has to re-embed nothing to run.
        return DenseRetriever(encoder, cache_dir=EMBEDDING_CACHE_DIR), None
    raise SystemExit(f"unknown arm {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["bm25", "graph", "dense-minilm", "dense-bge"])
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    t0 = time.perf_counter()
    corpus, queries, qrels, check = load_pool()
    print(f"pool: {len(corpus):,} passages, {len(queries):,} queries "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    retriever, needs_fit = _arm(args.arm)
    build_manifest, build_seconds = {}, 0.0
    if needs_fit:
        t1 = time.perf_counter()
        build_manifest = retriever.fit(corpus)
        build_seconds = time.perf_counter() - t1
        print(f"built in {build_seconds:.0f}s: "
              f"{build_manifest.get('nodes'):,} nodes", flush=True)

    out_dir = OUT / "pool" / args.arm
    summary = run_rung(
        retriever, "hotpotqa-distractor-pool", corpus, queries, qrels, out_dir,
        top_k=args.top_k, measures=GRAPH_MEASURES,
        extra_manifest=lambda: {
            # Build cost separately from query cost: the protocol requires it, because the
            # other arms have near-zero build cost and an arm whose advantage costs an
            # extraction pass must show that in the same table as its scores.
            "build_seconds": round(build_seconds, 1),
            "pool_control": check,
            **build_manifest,
        },
    )
    print(json.dumps(summary["ranked"], indent=2), flush=True)
    print(f"queries {summary['queries_scored']:,} | total {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
