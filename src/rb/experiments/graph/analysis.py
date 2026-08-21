"""
Experiment 003's registered analysis — protocols/003-graph-arm.md §7.

Every choice below was fixed in the tag before any arm was scored: the contrast, the
statistic, the resample count, the seed, the decision margin, and the correction family.
Nothing here is selected after seeing a number.
"""

import json
import random
from pathlib import Path

from rb import datasets, metrics
from rb.experiments.graph import coverage as cov
from rb.experiments.graph.measures import GRAPH_MEASURES, PRIMARY_MEASURE, SECONDARY_MEASURES

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"

B = 10_000
SEED = 20260820
MARGIN = 0.02  # §7's decision rule: a direction without this margin is "not resolved"


def _per_query(arm: str, qrels: dict) -> dict[str, dict[str, float]]:
    """Re-score from the committed per_query.jsonl rather than re-running retrieval, the same
    way 002's analysis recomputes from its artifact: the aggregate must be derivable from what
    was published, or a reader cannot check it."""
    run = {}
    for line in (OUT / "pool" / arm / "per_query.jsonl").read_text().splitlines():
        d = json.loads(line)
        docs = d["retrieved"]
        run[d["query_id"]] = {doc: float(len(docs) - i) for i, doc in enumerate(docs)}
    return metrics.score_ranked({q: qrels[q] for q in run}, run, GRAPH_MEASURES)


def _classes(definition: str) -> dict[str, int]:
    queries = datasets.load_queries("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")
    titles = datasets.load_titles("hotpotqa")
    gold = {q: [titles.get(d, "") for d in sorted(docs)] for q, docs in qrels.items()}
    return cov.coverage_all(queries, gold, definition)


def _contrast(diffs: dict[str, float], classes: dict[str, int]) -> dict:
    """
    Difference of class means, with a STRATIFIED percentile bootstrap.

    Stratified because class membership is a fixed property of each query, not something the
    sampling could have produced differently: resampling the pooled set would let the class
    sizes wander and inflate the interval with variation the design does not have.
    """
    absent = [diffs[q] for q in diffs if classes.get(q, 2) <= 1]
    named = [diffs[q] for q in diffs if classes.get(q) == 2]
    if not absent or not named:
        return {"error": "a class is empty"}
    mean = lambda xs: sum(xs) / len(xs)
    point = mean(absent) - mean(named)
    rng = random.Random(SEED)
    draws = []
    for _ in range(B):
        a = [absent[rng.randrange(len(absent))] for _ in absent]
        n = [named[rng.randrange(len(named))] for _ in named]
        draws.append(mean(a) - mean(n))
    draws.sort()
    lo, hi = draws[int(0.025 * B)], draws[int(0.975 * B)]
    p = 2 * min(sum(1 for x in draws if x <= 0), sum(1 for x in draws if x >= 0)) / B
    return {
        "n_bridge_absent": len(absent), "n_comparison": len(named),
        "mean_bridge_absent": round(mean(absent), 4), "mean_comparison": round(mean(named), 4),
        "difference": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "p_value": round(min(p, 1.0), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "meets_margin": bool(point >= MARGIN),
        "supported": bool(lo > 0 and point >= MARGIN),
    }


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni over the family §7 declares. Applied to whatever is in the family, so
    a member cannot be dropped after the fact to make another one significant."""
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(ordered):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = round(adj, 4)
    return out


def run() -> dict:
    qrels = datasets.load_qrels("hotpotqa")
    graph, bm25 = _per_query("graph", qrels), _per_query("bm25", qrels)
    shared = sorted(set(graph) & set(bm25))
    classes = {d: _classes(d) for d in (cov.PRIMARY, cov.SENSITIVITY)}

    results, pvals = {}, {}
    for definition in (cov.PRIMARY, cov.SENSITIVITY):
        for measure in (PRIMARY_MEASURE, *SECONDARY_MEASURES):
            diffs = {q: graph[q][measure] - bm25[q][measure] for q in shared}
            key = f"A|{measure}|{definition}"
            r = _contrast(diffs, classes[definition])
            results[key] = r
            pvals[key] = r["p_value"]
            # Prediction B: the negative control on coverage 2 alone.
            named = [diffs[q] for q in shared if classes[definition].get(q) == 2]
            rng = random.Random(SEED)
            dr = sorted(sum(named[rng.randrange(len(named))] for _ in named) / len(named)
                        for _ in range(B))
            lo, hi = dr[int(0.025 * B)], dr[int(0.975 * B)]
            pb = 2 * min(sum(1 for x in dr if x <= 0), sum(1 for x in dr if x >= 0)) / B
            bkey = f"B|{measure}|{definition}"
            results[bkey] = {"n": len(named), "mean_advantage": round(sum(named) / len(named), 4),
                             "ci95": [round(lo, 4), round(hi, 4)], "p_value": round(min(pb, 1.0), 4),
                             "no_advantage": bool(hi <= 0 or (lo <= 0 <= hi))}
            pvals[bkey] = results[bkey]["p_value"]

    holm = _holm(pvals)
    for k, v in results.items():
        v["p_holm"] = holm[k]
    overall = {m: {"graph": round(metrics.mean([graph[q][m] for q in shared]), 4),
                   "bm25": round(metrics.mean([bm25[q][m] for q in shared]), 4)}
               for m in sorted(GRAPH_MEASURES)}
    return {"queries": len(shared), "overall": overall, "contrasts": results,
            "family_size": len(pvals), "B": B, "seed": SEED, "margin": MARGIN}


if __name__ == "__main__":
    r = run()
    (OUT / "analysis.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["overall"], indent=2))
    for k, v in r["contrasts"].items():
        if k.startswith("A|"):
            print(f"{k}: diff {v['difference']:+.4f} CI {v['ci95']} p_holm {v['p_holm']} "
                  f"supported={v['supported']}")
