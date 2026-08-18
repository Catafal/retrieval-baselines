"""
Experiment 002 runner.

    python -m rb.experiments.ladder.run --dataset scifact --rung coordination
    python -m rb.experiments.ladder.run --dataset scifact --rung lexical-factorial

Mirrors 001's rb.run in shape (one command, results/002/<dataset>/<rung>/ artifacts)
but is written once against the Retriever interface via rb.retriever.run_rung(), so
it does not grow a new branch per rung the way a from-scratch runner would.

Query subsample is reused from rb.run.select_queries with the SAME seed, not
reimplemented, so the subsample is identical to 001's and per-query pairing
across the two entries stays valid (spec requirement: same corpora, same query
subsample, same metric code).

The dense and hybrid rungs are deliberately not wired into this CLI's --rung
choices yet: they need a real pinned encoder (rb.experiments.ladder.retrievers.dense.
SentenceTransformerEncoder), and running them is a separate, explicitly gated step
per protocols/002-ladder.md, not something `python -m rb.experiments.ladder.run`
should do by accident.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from rb import controls, datasets, metrics
from rb.retriever import run_rung
from rb.run import select_queries
from rb.stats import holm_correction, paired_bootstrap
from rb.experiments.ladder.retrievers.coordination import CoordinationRetriever
from rb.experiments.ladder.retrievers.lexical import ALL_CONFIGS, LADDER, build_index, full_bm25, shapley_from_ndcg

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results" / "002"
# 001's BM25 numbers are the external anchor the closure control checks against.
ANCHOR_DIR = ROOT / "results" / "001"


def _load_subsampled(dataset: str):
    corpus, queries, qrels = datasets.load(dataset)
    qids, sampled = select_queries(queries)
    subsampled_queries = {q: queries[q] for q in qids}
    return corpus, subsampled_queries, qrels, sampled


def _per_query_ndcg(out_dir: Path, qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """
    Recompute per-query nDCG@10 from the per_query.jsonl artifact run_rung()
    already wrote, instead of re-running retrieval or having run_rung() write
    out a redundant second artifact. per_query.jsonl only records rank ORDER
    (the retrieved doc-id list), not the retriever's raw score magnitude — but
    pytrec_eval's ranked measures depend only on order, so assigning any
    strictly-decreasing synthetic score to that order (len(retrieved) - i)
    reproduces exactly the per-query nDCG run_rung computed the first time.
    """
    run: dict[str, dict[str, float]] = {}
    with open(out_dir / "per_query.jsonl", encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            retrieved = row["retrieved"]
            run[row["query_id"]] = {d: len(retrieved) - i for i, d in enumerate(retrieved)}
    per_query = metrics.score_ranked(qrels, run)
    return {q: per_query[q]["ndcg_cut_10"] for q in per_query}


def run_coordination(dataset: str, top_k: int = 100) -> dict:
    """Rung 0, carried forward unchanged (see retrievers/coordination.py)."""
    corpus, queries, qrels, sampled = _load_subsampled(dataset)
    out_dir = RESULTS / dataset / "coordination"
    return run_rung(
        CoordinationRetriever(), dataset, corpus, queries, qrels, out_dir,
        top_k=top_k, subsampled=sampled, seed=20260818 if sampled else None,
    )


def run_lexical_factorial(dataset: str, top_k: int = 100) -> dict:
    """
    All eight lexical configurations on one dataset, plus the Shapley attribution
    computed from that factorial. Cheap: minutes per dataset (spec estimate),
    which is why this is the target wired into the Makefile's new
    reproduce-002-lexical rule.
    """
    corpus, queries, qrels, sampled = _load_subsampled(dataset)
    seed = 20260818 if sampled else None

    # The significance block below recomputes nDCG@10 from each rung's
    # per_query.jsonl, which stores only the top_k the run wrote. Below ten,
    # that reconstruction scores against fewer than ten documents and returns
    # something that looks like nDCG@10 and is not. No caller does this today;
    # the guard is here so none can start without noticing.
    if top_k < 10:
        raise ValueError(f"top_k={top_k} cannot support ndcg_cut_10; need at least 10")

    # Built ONCE per dataset, shared read-only across all eight configs below.
    # The index (postings, doc lengths, document frequencies, average length) is
    # config-independent — only the scoring formula changes between configs —
    # so rebuilding it inside each of the eight run_rung() calls would repeat
    # the one genuinely expensive step (a full tokenisation pass over the
    # corpus) eight times for no reason. See LexicalIndex's docstring.
    t_index = time.perf_counter()
    shared_index = build_index(corpus)
    index_seconds = time.perf_counter() - t_index
    t_scoring = time.perf_counter()

    ndcg_by_config = {}
    summaries = {}
    out_dirs = {}
    for cfg in ALL_CONFIGS:
        cfg_with_index = dataclasses.replace(cfg, index=shared_index)
        out_dir = RESULTS / dataset / cfg.name
        summary = run_rung(
            cfg_with_index, dataset, corpus, queries, qrels, out_dir, top_k=top_k, subsampled=sampled, seed=seed
        )
        # `index` is excluded from LexicalRetriever equality/hash (compare=False),
        # so cfg and cfg_with_index are the same dict key — using the plain cfg
        # here keeps ndcg_by_config[full_bm25()] below working without callers
        # needing to know an index was ever attached.
        ndcg_by_config[cfg] = summary["ranked"]["ndcg_cut_10"]
        summaries[cfg.name] = summary
        out_dirs[cfg.name] = out_dir

    # The closure control, run BEFORE anything is written. The all-on corner is full
    # BM25 by construction, so it has to agree with the externally anchored BM25 that
    # 001 already checked against the published BEIR figures. Without this gate the
    # factorial happily produces eight plausible numbers and a Shapley attribution
    # even when the scorer is wrong, which is the exact failure that retracted the
    # previous entry: a number that looks entirely reasonable and is not right.
    anchor = json.loads((ANCHOR_DIR / dataset / "bm25_control.json").read_text())
    closure = controls.bm25_closure(
        ndcg_by_config[full_bm25()],
        anchor["published_bm25_ndcg_cut_10"],
        anchor_ndcg=anchor["ndcg_cut_10"],
    )
    if not closure["passed"]:
        raise RuntimeError(
            f"BM25 closure control failed on {dataset}: {closure}. The all-on corner of "
            "the factorial is meant to BE full BM25, so disagreeing with the anchor means "
            "the lexical scorer is wrong. Nothing is written."
        )

    scoring_seconds = time.perf_counter() - t_scoring

    shapley = shapley_from_ndcg(ndcg_by_config)

    # Adjacent-rung significance, run BEFORE anything is written — same shape as
    # the closure control above. The point estimates in `ndcg_by_config` cannot
    # say whether adding the next mechanism up LADDER bought a real gain or is
    # noise on this query subsample; only the paired per-query difference can,
    # because it compares the two rungs on the SAME queries rather than two
    # marginal means. `_per_query_ndcg` reads the per_query.jsonl artifact each
    # LADDER rung's run_rung() call already wrote above — no rescoring, no
    # second retrieval. Holm correction is then applied once across the three
    # comparisons made on this dataset, per protocols/002-ladder.md section 5.
    qids = sorted(queries)
    ladder_qrels = {q: qrels[q] for q in qids}
    ladder_ndcg = [_per_query_ndcg(out_dirs[cfg.name], ladder_qrels) for cfg in LADDER]

    comparisons = []
    for i in range(len(LADDER) - 1):
        lo_cfg, hi_cfg = LADDER[i], LADDER[i + 1]
        lo_scores = [ladder_ndcg[i][q] for q in qids]
        hi_scores = [ladder_ndcg[i + 1][q] for q in qids]
        boot = paired_bootstrap(hi_scores, lo_scores)
        comparisons.append({"from": lo_cfg.name, "to": hi_cfg.name, **boot})
    significant = holm_correction([c["p_value"] for c in comparisons])
    for comparison, is_significant in zip(comparisons, significant):
        comparison["holm_significant"] = is_significant

    factorial_summary = {
        "dataset": dataset,
        "configs": {cfg.name: ndcg for cfg, ndcg in ndcg_by_config.items()},
        "shapley_ndcg_cut_10": shapley,
        "adjacent_rung_comparisons": comparisons,
        "controls": {"bm25_closure": closure},
        # The index is built once and shared, so its cost lands in no config's own
        # `cost.total_seconds` and would otherwise appear nowhere at all. Every
        # per-config figure would then understate what reproducing that config from
        # cold actually costs, and the total wall clock for the factorial would exist
        # only as prose. Both numbers are recorded here so a reader can check the
        # cost claims against an artifact rather than against a sentence.
        "cost": {
            "index_build_seconds": round(index_seconds, 1),
            "scoring_seconds": round(scoring_seconds, 1),
            "total_seconds": round(index_seconds + scoring_seconds, 1),
        },
    }
    out_dir = RESULTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lexical_factorial.json").write_text(json.dumps(factorial_summary, indent=2) + "\n")
    return factorial_summary


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 002 — the retrieval ladder")
    p.add_argument("--dataset", required=True, choices=datasets.DATASETS)
    p.add_argument("--rung", required=True, choices=("coordination", "lexical-factorial"))
    args = p.parse_args()

    t0 = time.time()
    if args.rung == "coordination":
        summary = run_coordination(args.dataset)
    else:
        summary = run_lexical_factorial(args.dataset)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
