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
from rb.stats import bootstrap_p_value, holm_adjusted, percentile_ci
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
    # Shared rule, not a hand-written index. Both bounds here used to be written out, and the
    # upper one omitted the `-1` that rb.stats derives, so it cut one fewer draw than the lower
    # bound did and every published upper bound sat one draw too high. See NB-26 D3.
    lo, hi = percentile_ci(draws)
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


def advantage(named: list[float]) -> dict:
    """
    §7's prediction B: the negative control, on coverage-2 queries alone.

    Extracted from `run()` rather than left inline. It was eight lines in the middle of a
    double loop, which is why the percentile call site inside it had no test of its own — NB-26's
    review found the reverting mutation surviving the whole suite. A named function with one
    argument can be pinned directly.
    """
    rng = random.Random(SEED)
    draws = sorted(sum(named[rng.randrange(len(named))] for _ in named) / len(named)
                   for _ in range(B))
    lo, hi = percentile_ci(draws)
    return {"n": len(named), "mean_advantage": round(sum(named) / len(named), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "p_value": round(bootstrap_p_value(draws, B), 6),
            # SPLIT, because these are different findings that the registered rule treats
            # alike. `no_advantage` is what section 7 registered and it stays as registered.
            # `verdict` distinguishes an interval entirely below zero (the graph is WORSE, a
            # confirmed negative control) from one straddling zero (inconclusive). All twelve
            # published B and E cells are entirely negative, so nothing published changes -- but
            # experiment 004 reuses this path and would otherwise report an inconclusive result
            # as a confirmed control.
            "no_advantage": bool(hi <= 0 or (lo <= 0 <= hi)),
            "verdict": ("confirmed_no_advantage" if hi <= 0
                        else "inconclusive" if lo <= 0 <= hi else "advantage")}


def decompose(graph, bm25, shared, classes, measure=PRIMARY_MEASURE) -> dict:
    r"""
    POST-HOC. Registered in protocols/003-amendment-5-differential-decomposition.md, AFTER the
    results were seen. Not a falsifier, not part of §7's Holm family, and it cannot be reported
    as though it had been predicted.

    WHAT IT ANSWERS. §7's prediction A is a DIFFERENCE OF ADVANTAGES: (graph - bm25) on
    bridge-absent queries, minus the same on coverage-2. Such a differential moves if EITHER arm
    moves, and nothing in the protocol required saying which did. The mechanism under test — a
    graph reaching a document the query does not name — predicts the GRAPH improves where the
    bridge entity is absent. A rising BM25 on coverage-2 produces the identical differential while
    meaning the opposite.

    So the same quantity is split into the two class swings it is the difference of:

        differential = (graph_absent - graph_cov2) - (bm25_absent - bm25_cov2)
                       \_______ graph term _______/   \_______ bm25 term _______/

    Same stratified bootstrap, same B, same seed as `_contrast`, so the decomposition and the
    registered contrast are the same resampling procedure read two ways.
    """
    absent = [q for q in shared if classes.get(q) in (0, 1)]
    named = [q for q in shared if classes.get(q) == 2]
    if not absent or not named:
        return {"error": "a class is empty"}
    mean = lambda xs: sum(xs) / len(xs)

    def terms(a_ids, c_ids):
        gt = mean([graph[q][measure] for q in a_ids]) - mean([graph[q][measure] for q in c_ids])
        bt = mean([bm25[q][measure] for q in a_ids]) - mean([bm25[q][measure] for q in c_ids])
        return gt, -bt, gt - bt

    g_point, b_point, total = terms(absent, named)
    rng = random.Random(SEED)
    g_draws, b_draws, share = [], [], []
    for _ in range(B):
        a = [absent[rng.randrange(len(absent))] for _ in absent]
        c = [named[rng.randrange(len(named))] for _ in named]
        gt, bt, tot = terms(a, c)
        g_draws.append(gt)
        b_draws.append(bt)
        if tot != 0:
            share.append(bt / tot)
    g_draws.sort(); b_draws.sort(); share.sort()
    g_lo, g_hi = percentile_ci(g_draws)
    b_lo, b_hi = percentile_ci(b_draws)
    s_lo, s_hi = percentile_ci(share) if share else (None, None)
    return {
        "measure": measure,
        "differential": round(total, 4),
        "graph_term": {"point": round(g_point, 4), "ci95": [round(g_lo, 4), round(g_hi, 4)],
                       "excludes_zero": bool(g_lo > 0 or g_hi < 0)},
        "bm25_term": {"point": round(b_point, 4), "ci95": [round(b_lo, 4), round(b_hi, 4)],
                      "excludes_zero": bool(b_lo > 0 or b_hi < 0)},
        # NOT A PROPORTION, despite the name. It is a ratio of two noisy quantities and is
        # unbounded: when the graph term is negative the baseline term must exceed the whole
        # differential to compensate, so this exceeds 1.0 in three of the six cells. Read it as
        # "how much of the differential the baseline accounts for", where values above 1 mean the
        # graph term worked against it. The bootstrap is stable here only because the denominator
        # never approaches zero on this data (minimum draw 0.0307); on data where it does, this
        # field is not interpretable and should be dropped rather than reported.
        "bm25_share_of_differential": {
            "point": round(b_point / total, 4) if total else None,
            "ci95": [round(s_lo, 4), round(s_hi, 4)] if share else None},
    }


def per_class_profile(graph, bm25, shared, classes, empty_qids) -> dict:
    """
    Per-coverage-class recall AND empty-retrieval rate, for both arms.

    WHY THIS EXISTS. The entry quotes the graph arm's R@2 by coverage class (0.1952 / 0.2272 /
    0.2100) and those numbers were in no committed artifact: `headroom()` only ever pools
    coverage {0,1} against {2} and never splits 0 from 1. A reader could not spot-check them from
    any file. That is the defect class this experiment has now closed four times, so it is not
    being left open a fifth.

    THE EMPTY RATE IS THE POINT. A query whose entities link to no node retrieves nothing and
    scores zero by arithmetic rather than by ranking. Reporting recall per class without it
    leaves the reader unable to tell a bad ranking from an absent one — and on 2Wiki the two
    classes differ by roughly a factor of two, which is the mechanism the entry names.
    """
    out = {}
    for label, keep in (("coverage_0", (0,)), ("coverage_1", (1,)),
                        ("coverage_2", (2,)), ("bridge_absent", (0, 1))):
        qs = [q for q in shared if classes.get(q) in keep]
        if not qs:
            continue
        empty = sum(1 for q in qs if q in empty_qids)
        out[label] = {
            "n": len(qs),
            "graph_recall_2": round(metrics.mean([graph[q]["recall_2"] for q in qs]), 4),
            "bm25_recall_2": round(metrics.mean([bm25[q]["recall_2"] for q in qs]), 4),
            "graph_retrieved_nothing": empty,
            "graph_empty_rate": round(empty / len(qs), 4),
        }
    return out


def empty_query_ids(arm: str) -> set[str]:
    """Queries for which `arm` returned an empty ranking."""
    return {json.loads(line)["query_id"]
            for line in (OUT / "pool" / arm / "per_query.jsonl").read_text().splitlines()
            if not json.loads(line)["retrieved"]}


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
            bkey = f"B|{measure}|{definition}"
            results[bkey] = advantage([diffs[q] for q in shared
                                       if classes[definition].get(q) == 2])
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
    # THREE artifacts, returned explicitly. The third is post-hoc and is kept OUT of `contrasts`
    # and out of `family_size` so it cannot be mistaken for part of the registered family.
    profile = per_class_profile(graph, bm25, shared, classes[cov.PRIMARY],
                                empty_query_ids("graph"))
    decomposition = {
        f"{m}|{d}": decompose(graph, bm25, shared, classes[d], m)
        for d in (cov.PRIMARY, cov.SENSITIVITY)
        for m in (PRIMARY_MEASURE, *SECONDARY_MEASURES)
    }
    return ({"queries": len(shared), "overall": overall, "contrasts": results,
             "per_class_profile": profile,
             "family_size": len(pvals), "B": B, "seed": SEED, "margin": MARGIN},
            headroom(graph, bm25, shared, classes[cov.PRIMARY]),
            decomposition)


if __name__ == "__main__":
    r, head, decomposition = run()
    # Written as its own artifact because §7 requires it reported alongside every delta, and
    # because it previously existed as a file with no code behind it.
    (OUT / "headroom-control.json").write_text(json.dumps(head, indent=2) + "\n")
    (OUT / "analysis.json").write_text(json.dumps(r, indent=2) + "\n")
    # Its own file, never merged into analysis.json: a post-hoc statistic sitting inside the
    # registered artifact is how it gets read as registered.
    (OUT / "decomposition.json").write_text(json.dumps({
        "status": (
            "POST-HOC, registered in protocols/003-amendment-5-differential-decomposition.md "
            "AFTER the results were seen. Not a falsifier and not in section 7's Holm family."),
        "cells": decomposition}, indent=2) + "\n")
    print(json.dumps(r["overall"], indent=2))
    for k, v in r["contrasts"].items():
        if k.startswith("A|"):
            print(f"{k}: diff {v['difference']:+.4f} CI {v['ci95']} p_holm {v['p_holm']} "
                  f"supported={v['supported']}")
