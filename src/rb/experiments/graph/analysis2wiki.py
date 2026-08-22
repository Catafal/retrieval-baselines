"""
The 2Wiki analysis — protocols/003-amendment-6-second-corpus.md, predictions C, D and E.

Adjudicates the predictions that were tagged before this corpus was scored. Same statistic, same
B, same seed, same margin and the same shared percentile rule as the HotpotQA analysis.

WHY A SEPARATE MODULE. `analysis.py` is the registered §7 analysis for HotpotQA and its outputs
are published. Parameterising it by corpus would put the two runs on one code path where a change
made for the second could silently move the first. The shared pieces — `rb.stats`, `coverage`,
`measures` — are imported rather than copied; what is separate is the adjudication, because the
predictions are different ones.

HOTPOTQA'S NUMBERS ARE NOT RECOMPUTED HERE. They are read as published constants, so this module
cannot alter them, and the crossover is a comparison between two runs rather than a re-analysis
of one.
"""

import json
import random
from pathlib import Path

from rb import metrics
from rb.experiments.graph import coverage as cov
from rb.experiments.graph import pool2wiki
from rb.experiments.graph.measures import GRAPH_MEASURES, PRIMARY_MEASURE, SECONDARY_MEASURES
from rb.stats import bootstrap_p_value, holm_adjusted, percentile_ci

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003" / "2wiki"

B = 10_000
SEED = 20260820
MARGIN = 0.02

# Published in results/003/pool/*/summary.json, quoted rather than recomputed. Prediction D is a
# comparison against these, so they are constants here by design.
HOTPOT_GRAPH_R2 = 0.2148
HOTPOT_BM25_R2 = 0.5490
HOTPOT_GRAPH_CLASS_DIFF = 0.0065          # graph's own between-class R@2 difference
HOTPOT_GRAPH_CLASS_DIFF_CI = [-0.0082, 0.0212]


def _per_query(arm: str, qrels: dict) -> dict:
    run = {}
    for line in (OUT / arm / "per_query.jsonl").read_text().splitlines():
        d = json.loads(line)
        docs = d["retrieved"]
        run[d["query_id"]] = {doc: float(len(docs) - i) for i, doc in enumerate(docs)}
    return metrics.score_ranked({q: qrels[q] for q in run}, run, GRAPH_MEASURES)


def own_class_difference(scores: dict, absent: list[str], named: list[str], measure: str) -> dict:
    """
    PREDICTION C. One arm's OWN difference between query classes — not a differential against a
    baseline.

    This is the shape amendment 5 forced. §7's prediction A was a difference of advantages, which
    moves if either arm moves, and the decomposition showed HotpotQA's passing result was the
    baseline's doing. Asking about one arm at a time cannot have that ambiguity.
    """
    mean = lambda xs: sum(xs) / len(xs)
    point = mean([scores[q][measure] for q in absent]) - mean([scores[q][measure] for q in named])
    rng = random.Random(SEED)
    draws = []
    for _ in range(B):
        a = [absent[rng.randrange(len(absent))] for _ in absent]
        n = [named[rng.randrange(len(named))] for _ in named]
        draws.append(mean([scores[q][measure] for q in a]) - mean([scores[q][measure] for q in n]))
    draws.sort()
    lo, hi = percentile_ci(draws)
    return {
        "n_bridge_absent": len(absent), "n_comparison": len(named),
        "difference": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "p_value": round(bootstrap_p_value(draws, B), 6),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "meets_margin": bool(point >= MARGIN),
        # Registered as: interval excludes zero AND point >= +0.02, in the POSITIVE direction.
        "supported": bool(lo > 0 and point >= MARGIN),
    }


def run() -> dict:
    _corpus, titles, queries, qrels = pool2wiki.build()
    graph, bm25 = _per_query("graph", qrels), _per_query("bm25", qrels)
    shared = sorted(set(graph) & set(bm25))
    gold = {q: [titles.get(d, "") for d in sorted(docs)] for q, docs in qrels.items()}

    results, pvals = {}, {}
    per_class = {}
    for definition in (cov.PRIMARY, cov.SENSITIVITY):
        classes = cov.coverage_all(queries, gold, definition)
        absent = [q for q in shared if classes.get(q) in (0, 1)]
        named = [q for q in shared if classes.get(q) == 2]
        for measure in (PRIMARY_MEASURE, *SECONDARY_MEASURES):
            key = f"C|{measure}|{definition}"
            results[key] = own_class_difference(graph, absent, named, measure)
            pvals[key] = results[key]["p_value"]
            # PREDICTION E, the negative control, carried over from §7: no advantage on coverage-2.
            adv = [graph[q][measure] - bm25[q][measure] for q in named]
            rng = random.Random(SEED)
            dr = sorted(sum(adv[rng.randrange(len(adv))] for _ in adv) / len(adv) for _ in range(B))
            lo, hi = percentile_ci(dr)
            ekey = f"E|{measure}|{definition}"
            results[ekey] = {"n": len(adv), "mean_advantage": round(sum(adv) / len(adv), 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "p_value": round(bootstrap_p_value(dr, B), 6),
                             "no_advantage": bool(hi <= 0 or (lo <= 0 <= hi))}
            pvals[ekey] = results[ekey]["p_value"]
        if definition == cov.PRIMARY:
            for lbl, keep in (("coverage_0", (0,)), ("coverage_1", (1,)),
                              ("bridge_absent", (0, 1)), ("coverage_2", (2,))):
                qs = [q for q in shared if classes.get(q) in keep]
                if qs:
                    per_class[lbl] = {
                        "n": len(qs),
                        "graph_recall_2": round(metrics.mean([graph[q]["recall_2"] for q in qs]), 4),
                        "bm25_recall_2": round(metrics.mean([bm25[q]["recall_2"] for q in qs]), 4)}

    keys = sorted(pvals)
    for k, adj in zip(keys, holm_adjusted([pvals[k] for k in keys])):
        results[k]["p_holm"] = round(adj, 6)

    # PREDICTION D — the crossover, against HotpotQA's published constants.
    g = metrics.mean([graph[q]["recall_2"] for q in shared])
    b = metrics.mean([bm25[q]["recall_2"] for q in shared])
    deficit = g - b
    hotpot_deficit = HOTPOT_GRAPH_R2 - HOTPOT_BM25_R2
    crossover = {
        "twowiki_deficit_r2": round(deficit, 4),
        "hotpotqa_deficit_r2": round(hotpot_deficit, 4),
        "shrink_points": round((deficit - hotpot_deficit) * 100, 2),
        "registered_threshold_points": 10.0,
        "supported": bool((deficit - hotpot_deficit) >= 0.10),
    }
    empty = sum(1 for line in (OUT / "graph" / "per_query.jsonl").read_text().splitlines()
                if not json.loads(line)["retrieved"])
    overall = {m: {"graph": round(metrics.mean([graph[q][m] for q in shared]), 4),
                   "bm25": round(metrics.mean([bm25[q][m] for q in shared]), 4)}
               for m in sorted(GRAPH_MEASURES)}
    return {
        "queries": len(shared), "overall": overall, "per_class_recall_2": per_class,
        "prediction_c_and_e": results, "prediction_d_crossover": crossover,
        "graph_no_seed": {"queries": empty, "rate": round(empty / len(shared), 4)},
        "hotpotqa_reference": {
            "graph_own_class_difference": HOTPOT_GRAPH_CLASS_DIFF,
            "ci95": HOTPOT_GRAPH_CLASS_DIFF_CI},
        "family_size": len(pvals), "B": B, "seed": SEED, "margin": MARGIN,
    }


if __name__ == "__main__":
    r = run()
    (OUT / "analysis.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["overall"], indent=2))
    print(json.dumps(r["per_class_recall_2"], indent=2))
    for k, v in r["prediction_c_and_e"].items():
        if k.startswith("C|"):
            print(f"{k}: diff {v['difference']:+.4f} CI {v['ci95']} p_holm {v['p_holm']} supported={v['supported']}")
    print("crossover:", json.dumps(r["prediction_d_crossover"]))
