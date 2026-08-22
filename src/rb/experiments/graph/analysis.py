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
from rb.stats import bootstrap_p_value, holm_adjusted
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
    # An unclassified query belongs to NEITHER arm. Previously spelled three
    # different ways in this file (.get(q, 2), .get(q), .get(q, -1)); one of them
    # would have silently swept unclassified queries into a class had a default
    # ever been hit.
    absent = [diffs[q] for q in diffs if classes.get(q) in (0, 1)]
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
    p = bootstrap_p_value(draws, B)
    return {
        "n_bridge_absent": len(absent), "n_comparison": len(named),
        "mean_bridge_absent": round(mean(absent), 4), "mean_comparison": round(mean(named), 4),
        "difference": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "p_value": round(p, 6),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "meets_margin": bool(point >= MARGIN),
        "supported": bool(lo > 0 and point >= MARGIN),
    }


def headroom(graph, bm25, shared, classes) -> dict:
    """
    §7's headroom control: is the differential a mechanism, or just more room to improve?

    WHY THIS LIVES HERE NOW. The committed headroom-control.json had NO producer anywhere in
    the source tree — a third artifact in this experiment written by code that no longer
    exists, and its own status field calls it "required by protocol section 7". It was also
    stale, carrying the pre-PPR-fix differential. Rebuilt here rather than in a new module
    because `run()` already assembles every input it needs: this is another view of the same
    per-query data, not a separate computation.

    NORMALISED BY HEADROOM, because a class where BM25 already scores well has less room left,
    so a smaller raw deficit there could mean either a better graph or a lower ceiling. The
    normalised figure is what separates those two readings.
    """
    m = PRIMARY_MEASURE
    out = {}
    for label, keep in (("bridge_absent", (0, 1)), ("comparison", (2,))):
        qs = [q for q in shared if classes.get(q) in keep]
        if not qs:
            continue
        b = metrics.mean([bm25[q][m] for q in qs])
        g = metrics.mean([graph[q][m] for q in qs])
        out[label] = {
            "n": len(qs),
            "bm25_recall_2": round(b, 4),
            "graph_recall_2": round(g, 4),
            "raw_deficit": round(g - b, 4),
            "headroom": round(1.0 - b, 4),
            "headroom_normalised_deficit": round((g - b) / (1.0 - b), 4) if b < 1 else None,
        }
    result = {
        "status": "HEADROOM CONTROL, required by protocol section 7 and reported alongside every delta.",
        "measure": m,
        "per_class": out,
    }
    if {"bridge_absent", "comparison"} <= set(out):
        result["raw_differential"] = round(
            out["bridge_absent"]["raw_deficit"] - out["comparison"]["raw_deficit"], 4)
        # Only when BOTH normalised deficits are defined. A class whose BM25 baseline is at
        # ceiling has zero headroom, so its normalised deficit is undefined rather than zero,
        # and differencing None against a float would crash where it should simply be absent.
        na = out["bridge_absent"]["headroom_normalised_deficit"]
        nc = out["comparison"]["headroom_normalised_deficit"]
        if na is not None and nc is not None:
            result["normalised_differential"] = round(na - nc, 4)
    return result


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
            pb = bootstrap_p_value(dr, B)
            bkey = f"B|{measure}|{definition}"
            results[bkey] = {"n": len(named), "mean_advantage": round(sum(named) / len(named), 4),
                             "ci95": [round(lo, 4), round(hi, 4)], "p_value": round(pb, 6),
                             "no_advantage": bool(hi <= 0 or (lo <= 0 <= hi))}
            pvals[bkey] = results[bkey]["p_value"]

    # Holm over the family §7 declares, via the SHARED implementation. There used to be a
    # second copy here; two Holm functions in one repository is how 002's decisions and 003's
    # adjusted p-values drift apart with no test noticing. `holm_adjusted` returns INPUT order,
    # so zipping the keys back is safe — but the keys are sorted first so the mapping does not
    # depend on dict insertion order.
    keys = sorted(pvals)
    for k, adj in zip(keys, holm_adjusted([pvals[k] for k in keys])):
        results[k]["p_holm"] = round(adj, 6)
    overall = {m: {"graph": round(metrics.mean([graph[q][m] for q in shared]), 4),
                   "bm25": round(metrics.mean([bm25[q][m] for q in shared]), 4)}
               for m in sorted(GRAPH_MEASURES)}
    # Returns the PAIR explicitly. This used to smuggle the headroom control through the
    # results dict under a leading-underscore key that __main__ had to remember to pop, and a
    # caller who forgot would have leaked it into the published analysis.json. A tuple makes
    # the second artifact part of the signature instead of a naming convention.
    return {"queries": len(shared), "overall": overall, "contrasts": results,
            "family_size": len(pvals), "B": B, "seed": SEED, "margin": MARGIN}, \
        headroom(graph, bm25, shared, classes[cov.PRIMARY])


if __name__ == "__main__":
    r, head = run()
    # Written as its own artifact because §7 requires it reported alongside every delta, and
    # because it previously existed as a file with no code behind it.
    (OUT / "headroom-control.json").write_text(json.dumps(head, indent=2) + "\n")
    (OUT / "analysis.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["overall"], indent=2))
    for k, v in r["contrasts"].items():
        if k.startswith("A|"):
            print(f"{k}: diff {v['difference']:+.4f} CI {v['ci95']} p_holm {v['p_holm']} "
                  f"supported={v['supported']}")
