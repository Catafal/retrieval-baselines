"""
Experiment 001 runner. One command, clone to numbers.

    python -m rb.run --dataset scifact

Writes results/001/<dataset>/per_query.jsonl and summary.json. The per-query file
is the artifact: it holds the retrieved document ids in rank order for every query,
so every aggregate in the published entry can be recomputed from it without rerunning
ripgrep — and disagreements between the prose and the data are findable by a reader.
"""

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from rb import controls, datasets, metrics
from rb.grep_baseline import materialise, run_query

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "001"

# Pre-registered query subsample. Subsampling QUERIES is unbiased; subsampling the
# corpus would shrink the haystack and inflate recall, so it is never done.
SAMPLE_SIZE = 500
SEED = 20260818


def environment() -> dict:
    """Everything a stranger needs to tell an environment difference from a finding."""
    rg = subprocess.run(["rg", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    manifest_path = ROOT / "manifests" / "datasets.json"
    return {
        "ripgrep": rg,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": commit,
        "dataset_checksums": json.loads(manifest_path.read_text()) if manifest_path.exists() else {},
    }


def select_queries(queries: dict[str, str]) -> tuple[list[str], bool]:
    """Deterministic subsample. Sorted first so dict order cannot affect the draw."""
    qids = sorted(queries)
    if len(qids) <= SAMPLE_SIZE:
        return qids, False
    return sorted(random.Random(SEED).sample(qids, SAMPLE_SIZE)), True


def run(dataset: str, word_bounded: bool = True, top_k: int = 100) -> dict:
    corpus, queries, qrels = datasets.load(dataset)
    corpus_path = datasets.DATA_DIR / dataset / "rg_corpus.txt"
    doc_ids = materialise(corpus, corpus_path)

    checks = {
        "gold_presence": controls.gold_presence(corpus, qrels),
        "empty_query": controls.empty_query(corpus_path, doc_ids),
    }
    for name, c in checks.items():
        if not c["passed"]:
            raise RuntimeError(f"control {name} failed: {c}. Nothing is scored on a broken harness.")

    qids, sampled = select_queries(queries)
    # The unbounded-substring sensitivity check writes to its own directory. Sharing one
    # would let the check silently overwrite the headline artifact.
    out_dir = RESULTS / dataset if word_bounded else RESULTS / dataset / "substring"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Document id -> ripgrep line number, so full-set recall is an O(|gold|) lookup
    # rather than materialising a set of millions of ids per query.
    line_of = {did: i + 1 for i, did in enumerate(doc_ids)}

    run_dict: dict[str, dict[str, float]] = {}
    rows, set_recalls, set_sizes, times = [], [], [], []

    with open(out_dir / "per_query.jsonl", "w", encoding="utf8") as f:
        for i, qid in enumerate(qids, 1):
            results, hits, elapsed, terms = run_query(
                queries[qid], corpus_path, doc_ids, top_k=top_k, word_bounded=word_bounded
            )
            set_size = len(hits)
            run_dict[qid] = dict(results)
            gold = qrels[qid]
            found = sum(1 for g in gold if line_of.get(g) in hits)
            sr = found / len(gold) if gold else 0.0
            set_recalls.append(sr)
            set_sizes.append(set_size)
            times.append(elapsed)
            rows.append(qid)
            f.write(json.dumps({
                "query_id": qid,
                "query": queries[qid],
                "terms": terms,
                "gold": sorted(qrels[qid]),
                "retrieved": [d for d, _ in results],
                "set_size": set_size,
                "seconds": round(elapsed, 4),
            }) + "\n")
            if i % 50 == 0:
                print(f"  {dataset}: {i}/{len(qids)}", flush=True)

    per_query = metrics.score_ranked({q: qrels[q] for q in rows}, run_dict)

    # Invariant control: recall over grep's whole output cannot be lower than recall
    # over its first 100. Equality across every query means set recall was computed
    # from the truncated list — the bug this check was added after finding.
    full_set_recall = metrics.mean(set_recalls)
    recall_at_100 = metrics.mean([per_query[q]["recall_100"] for q in rows])
    checks["set_recall_dominates_recall_100"] = {
        "recall_full_set": round(full_set_recall, 4),
        "recall_100": round(recall_at_100, 4),
        "passed": full_set_recall >= recall_at_100 - 1e-9,
    }
    if not checks["set_recall_dominates_recall_100"]["passed"]:
        raise RuntimeError(f"invariant control failed: {checks['set_recall_dominates_recall_100']}")
    summary = {
        "dataset": dataset,
        "queries_scored": len(rows),
        "queries_available": len(queries),
        "subsampled": sampled,
        "seed": SEED if sampled else None,
        "corpus_documents": len(corpus),
        "word_bounded": word_bounded,
        "controls": checks,
        "ranked": {
            m: round(metrics.mean([per_query[q][m] for q in rows]), 4) for m in sorted(metrics.MEASURES)
        },
        "set": {
            "recall_full_set": round(full_set_recall, 4),
            "median_set_size": sorted(set_sizes)[len(set_sizes) // 2],
            "mean_set_size": round(metrics.mean([float(s) for s in set_sizes]), 1),
            "queries_returning_nothing": sum(1 for s in set_sizes if s == 0),
        },
        "cost": {
            "mean_seconds_per_query": round(metrics.mean(times), 3),
            "total_seconds": round(sum(times), 1),
            "usd": 0.0,
        },
        "environment": environment(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 001 — grep baseline")
    p.add_argument("--dataset", required=True, choices=datasets.DATASETS)
    p.add_argument("--substring", action="store_true",
                   help="sensitivity check: unbounded substring matching instead of word-bounded")
    args = p.parse_args()
    t0 = time.time()
    summary = run(args.dataset, word_bounded=not args.substring)
    print(json.dumps({k: summary[k] for k in ("dataset", "queries_scored", "ranked", "set", "cost")}, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
