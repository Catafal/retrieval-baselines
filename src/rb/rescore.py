"""
Recompute ranked metrics from the committed per-query artifacts.

    python -m rb.rescore

Needed because the scores originally handed to trec_eval tied heavily: the encoding
`distinct + total/(total+1)` produced 16 distinct values across one query's top 100, so
trec_eval was breaking 84 ties by its own internal rule instead of by document id as
protocol.md section 3 specifies. `rank()` now emits strictly decreasing scores, which
makes the scored order the pre-registered order.

Rescoring does not need ripgrep to run again, because per_query.jsonl already stores the
retrieved ids in rank order. That is the point of committing it: an error in the scoring
layer is repairable without repeating a two-hour measurement, and anyone else can repeat
this repair from the same files.

Set metrics are not recomputed — full-set recall is an aggregate over grep's entire hit
map, which the artifact deliberately does not store — so those fields are carried through
untouched.
"""

import json
import random
from pathlib import Path

import pytrec_eval

from rb import datasets, metrics

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_ROUNDS = 2000


def bootstrap_ci(values: list[float], rounds: int = BOOTSTRAP_ROUNDS) -> tuple[float, float]:
    """Percentile bootstrap 95% interval over per-query scores."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def rescore(dataset: str) -> dict:
    path = ROOT / "results" / "001" / dataset
    rows = [json.loads(l) for l in open(path / "per_query.jsonl", encoding="utf8")]

    qrels = {r["query_id"]: {g: 1 for g in r["gold"]} for r in rows}
    # Strictly decreasing by stored rank position: the pre-registered order, exactly.
    run = {
        r["query_id"]: {doc: float(len(r["retrieved"]) - i) for i, doc in enumerate(r["retrieved"])}
        for r in rows
    }
    per_query = pytrec_eval.RelevanceEvaluator(qrels, metrics.MEASURES).evaluate(run)

    summary = json.loads((path / "summary.json").read_text())
    ranked, intervals = {}, {}
    for m in sorted(metrics.MEASURES):
        vals = [per_query[r["query_id"]][m] for r in rows]
        ranked[m] = round(metrics.mean(vals), 4)
        lo, hi = bootstrap_ci(vals)
        intervals[m] = [round(lo, 4), round(hi, 4)]

    summary["ranked"] = ranked
    summary["ranked_ci95"] = intervals
    summary["rescored"] = {
        "reason": "strictly decreasing scores so trec_eval applies the pre-registered "
                  "document-id tie-break rather than its own",
        "source": "per_query.jsonl, no retrieval rerun",
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "seed": BOOTSTRAP_SEED},
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    for d in datasets.DATASETS:
        for sub in ("", "substring"):
            p = ROOT / "results" / "001" / d / sub
            if (p / "per_query.jsonl").exists():
                name = f"{d}/{sub}" if sub else d
                s = rescore(str(Path(d) / sub) if sub else d)
                ci = s["ranked_ci95"]["ndcg_cut_10"]
                print(f"{name:>20}: nDCG@10 {s['ranked']['ndcg_cut_10']:.4f}  95% CI {ci}")
