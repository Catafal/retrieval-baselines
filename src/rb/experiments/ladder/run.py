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
import json
import time
from pathlib import Path

from rb import controls, datasets
from rb.retriever import run_rung
from rb.run import select_queries
from rb.experiments.ladder.retrievers.coordination import CoordinationRetriever
from rb.experiments.ladder.retrievers.lexical import ALL_CONFIGS, full_bm25, shapley_from_ndcg

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results" / "002"
# 001's BM25 numbers are the external anchor the closure control checks against.
ANCHOR_DIR = ROOT / "results" / "001"


def _load_subsampled(dataset: str):
    corpus, queries, qrels = datasets.load(dataset)
    qids, sampled = select_queries(queries)
    subsampled_queries = {q: queries[q] for q in qids}
    return corpus, subsampled_queries, qrels, sampled


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

    ndcg_by_config = {}
    summaries = {}
    for cfg in ALL_CONFIGS:
        out_dir = RESULTS / dataset / cfg.name
        summary = run_rung(cfg, dataset, corpus, queries, qrels, out_dir, top_k=top_k, subsampled=sampled, seed=seed)
        ndcg_by_config[cfg] = summary["ranked"]["ndcg_cut_10"]
        summaries[cfg.name] = summary

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

    shapley = shapley_from_ndcg(ndcg_by_config)
    factorial_summary = {
        "dataset": dataset,
        "configs": {cfg.name: ndcg for cfg, ndcg in ndcg_by_config.items()},
        "shapley_ndcg_cut_10": shapley,
        "controls": {"bm25_closure": closure},
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
