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
from rb.experiments.graph import pool, pool2wiki
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.retriever import run_rung

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"


def load_pool_2wiki():
    """The 2Wiki candidate set — protocols/003-amendment-6-second-corpus.md.

    Separate from `load_pool` rather than branching inside it. The two corpora resolve documents
    by different routes (BEIR ids inherited vs ids minted from titles) and share no step, so a
    shared function would be two functions wearing one name.
    """
    corpus, titles, queries, qrels = pool2wiki.build()
    check = pool2wiki.control()
    if not check["passed"]:
        raise RuntimeError(f"2wiki pool control failed: {check}. Nothing is scored on a broken pool.")
    return corpus, titles, queries, qrels, check


def load_pool():
    """The candidate set, rebuilt from the dataset rather than cached, so every scored run
    re-asserts the §9 controls instead of trusting a file written earlier."""
    ctx = pool.load_distractor_context()
    corpus = datasets.load_corpus("hotpotqa")
    titles = datasets.load_titles("hotpotqa")
    queries = datasets.load_queries("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")

    # MEASURED, all of it, and EVALUATED BEFORE build(). This used to pass `unresolved=0,
    # collisions=0` as literals and the same `len(qrels)` expression as both sides of the
    # gold-title check, so three of the control's published fields were assertions and one of its
    # four checks was `x == x`. The ordering matters as much as the measurement: build() raises on
    # an unresolved title, so a control evaluated after it could never report those fields failing.
    # See protocols/003-amendment-4-pool-control.md.
    check = controls.pool_construction(**pool.construction_counts(titles, ctx, qrels))
    if not check["passed"]:
        raise RuntimeError(f"pool_construction failed: {check}. Nothing is scored on a broken pool.")

    # Only now. build()'s and title_index()'s own raises are defence in depth behind a control
    # that has already had its chance to fail on its own terms.
    pool_corpus, _ = pool.build(corpus, titles, ctx)

    queries = {q: t for q, t in queries.items() if q in qrels}
    return pool_corpus, queries, qrels, check


def _arm(name: str, dataset: str = "hotpotqa", corpus: dict | None = None):
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
    if name == "graph-glm":
        # Experiment 004: the same graph, the same walk, the same everything, with the extractor
        # swapped on BOTH sides. Supplying one side without the other raises in GraphRetriever —
        # see protocols/004-amendment-3-query-extraction.md for why a half-swap would measure
        # extractor mismatch and look like a confirmation of 003.
        from rb.experiments.graph import llm_extractor as llm
        from rb.experiments.graph.retriever import GraphRetriever
        return GraphRetriever(extract_docs=llm.extract_docs_offline,
                              extract_query=llm.extract_query_offline,
                              name="graph-glm-ppr"), "fit"
    if name in ("graph-typed", "graph-glm-typed", "graph-typed-capped", "graph-glm-typed-capped"):
        # Experiment 005: the same walk, the same extractor as its string-identity twin, with
        # identity resolved through the committed redirect registry instead of by exact string.
        # protocols/005-typed-identity.md section 2 — the 2x2 exists so extractor quality and
        # identity resolution cannot be attributed to each other.
        #
        # ONE linker, passed once. fit() hands it to build() and _seed() reads the same stored
        # value, so a graph keyed by one identity and seeded by another is not constructible.
        from rb.experiments.graph import linker as lk
        from rb.experiments.graph import redirects
        from rb.experiments.graph.identity_coverage import _load
        from rb.experiments.graph.retriever import GraphRetriever

        corpus_name = "hotpotqa" if dataset == "hotpotqa" else "2wiki"
        _, _, pool_titles = _load(corpus_name)
        registry, drops = lk.build_registry(redirects.load(corpus_name), pool_titles)
        print(f"identity: {drops['kept']:,} aliases kept, "
              f"{drops['dropped_ambiguous']:,} ambiguous dropped", flush=True)

        from rb.experiments.graph import llm_extractor as llm
        glm = "glm" in name

        # R9 — protocols/005-amendment-2-hub-cap.md. Applied here, after build_registry, so the
        # capped arm differs from its uncapped twin by this one rule and nothing else. The
        # entities it measures document frequency over are the SAME ones the arm will build with.
        if name.endswith("-capped"):
            from rb.experiments.graph.extractor import extract_many as spacy_extract_many
            from rb.experiments.graph.extractor import node_strings
            raw = (llm.extract_docs_offline if glm else spacy_extract_many)(corpus)
            ents = {d: node_strings(e) for d, e in raw.items()}
            registry, cap_stats = lk.apply_df_cap(registry, ents, len(corpus))
            print(f"R9 cap: {cap_stats}", flush=True)

        if not glm:
            return GraphRetriever(link=lk.linker(registry),
                                  name=f"{name}-ppr"), "fit"
        return GraphRetriever(extract_docs=llm.extract_docs_offline,
                              extract_query=llm.extract_query_offline,
                              link=lk.linker(registry),
                              name=f"{name}-ppr"), "fit"
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
                    choices=["bm25", "graph", "graph-glm", "graph-typed", "graph-glm-typed",
                             "graph-typed-capped", "graph-glm-typed-capped",
                             "dense-minilm", "dense-bge"])
    ap.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wiki"])
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.dataset == "2wiki":
        corpus, _titles, queries, qrels, check = load_pool_2wiki()
    else:
        corpus, queries, qrels, check = load_pool()
    print(f"pool[{args.dataset}]: {len(corpus):,} passages, {len(queries):,} queries "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    if args.arm == "graph-glm":
        # Checked before the expensive build, not during the walk.
        from rb.experiments.graph import llm_extractor as llm
        llm.assert_queries_cached(queries)

    retriever, needs_fit = _arm(args.arm, args.dataset, corpus)
    build_manifest, build_seconds = {}, 0.0
    if needs_fit:
        t1 = time.perf_counter()
        build_manifest = retriever.fit(corpus)
        build_seconds = time.perf_counter() - t1
        print(f"built in {build_seconds:.0f}s: "
              f"{build_manifest.get('nodes'):,} nodes", flush=True)

    # 2Wiki writes under its own subtree so the published HotpotQA artifacts cannot be overwritten
    # by a run of the second corpus.
    out_dir = OUT / "pool" / args.arm if args.dataset == "hotpotqa" else OUT / "2wiki" / args.arm
    summary = run_rung(
        retriever, f"{args.dataset}-distractor-pool", corpus, queries, qrels, out_dir,
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
