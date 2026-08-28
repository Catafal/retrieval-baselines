"""
Experiment 005's registered analysis — protocols/005-typed-identity.md §4, §5, §6.

Every choice below was fixed in `protocol-005` before the first typed arm was scored: the
contrast, the statistic, the resample count, the seed, the correction family, and the decision
rules including the `underpowered` label. Nothing here is selected after seeing a number.

The alias-affected subsets are READ FROM DISK rather than recomputed, from files written by
Stage 0's own function before any per-query difference had been looked at. That is the whole
basis for prediction B meaning anything: a subset recomputed at analysis time is a subset that
could have been redefined once it was known which queries improved.
"""

import json
import random
from pathlib import Path

from rb import datasets, metrics
from rb.experiments.graph import pool2wiki
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.stats import bootstrap_p_value, holm_adjusted, paired_bootstrap, percentile_ci

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results"
OUT = RESULTS / "005"

B = 10_000
SEED = 20260820
ALPHA = 0.05
PRIMARY = "recall_2"

# The 2x2. The string column is 003's and 004's published arms, not re-run.
CELLS = [
    {"corpus": "hotpotqa", "extractor": "spacy", "string": "graph", "typed": "graph-typed"},
    {"corpus": "hotpotqa", "extractor": "glm", "string": "graph-glm", "typed": "graph-glm-typed"},
    {"corpus": "2wiki", "extractor": "spacy", "string": "graph", "typed": "graph-typed"},
    {"corpus": "2wiki", "extractor": "glm", "string": "graph-glm", "typed": "graph-glm-typed"},
]

# §5, quoted from Stage 0 rather than recomputed here. A cell whose interval includes zero AND
# whose width exceeds its MDE reports `underpowered`, because silence from a sample that could
# not have spoken is not evidence of absence.
MDE = {
    ("hotpotqa", "spacy"): 0.0568,
    ("hotpotqa", "glm"): 0.0546,
    ("2wiki", "spacy"): 0.0974,
    ("2wiki", "glm"): 0.0601,
}


def _arm_dir(corpus: str, arm: str) -> Path:
    sub = "pool" if corpus == "hotpotqa" else "2wiki"
    return RESULTS / "003" / sub / arm


def _qrels(corpus: str) -> dict:
    if corpus == "hotpotqa":
        return datasets.load_qrels("hotpotqa")
    _, _, _, qrels = pool2wiki.build()
    return qrels


def _per_query(corpus: str, arm: str, qrels: dict) -> dict[str, dict[str, float]]:
    """Re-scored from the committed per_query.jsonl, the way 003's analysis does it: the
    aggregate must be derivable from what was published or a reader cannot check it."""
    run = {}
    for line in (_arm_dir(corpus, arm) / "per_query.jsonl").read_text().splitlines():
        d = json.loads(line)
        docs = d["retrieved"]
        run[d["query_id"]] = {doc: float(len(docs) - i) for i, doc in enumerate(docs)}
    return metrics.score_ranked({q: qrels[q] for q in run}, run, GRAPH_MEASURES)


def _affected(corpus: str, extractor: str) -> set[str]:
    path = OUT / f"affected-{corpus}-{extractor}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Prediction B decomposes over the subset Stage 0 fixed; "
            "recomputing it here would defeat the point of having fixed it. Run "
            "`python -m rb.experiments.graph.identity_coverage --ids` first."
        )
    return set(json.loads(path.read_text())["affected"])


def _subset_contrast(diffs: dict[str, float], affected: set[str]) -> dict:
    """
    §5's statistic: (typed − string) on the affected subset, minus the same on its complement.

    Stratified — each query's membership is a fixed property decided a stage earlier, not
    something the sampling could have produced differently, so the two strata are resampled
    separately and the class sizes are held.
    """
    inside = [diffs[q] for q in diffs if q in affected]
    outside = [diffs[q] for q in diffs if q not in affected]
    if not inside or not outside:
        return {"resolved": False, "reason": "a stratum is empty"}

    observed = sum(inside) / len(inside) - sum(outside) / len(outside)
    rng = random.Random(SEED)
    draws = []
    for _ in range(B):
        a = [inside[rng.randrange(len(inside))] for _ in range(len(inside))]
        b = [outside[rng.randrange(len(outside))] for _ in range(len(outside))]
        draws.append(sum(a) / len(a) - sum(b) / len(b))
    draws.sort()
    lo, hi = percentile_ci(draws)
    return {
        "observed": round(observed, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p": bootstrap_p_value(draws, B),
        "n_affected": len(inside),
        "n_unaffected": len(outside),
        "resolved": True,
    }


def run() -> dict:
    cells = []
    for spec in CELLS:
        corpus, extractor = spec["corpus"], spec["extractor"]
        qrels = _qrels(corpus)
        string = _per_query(corpus, spec["string"], qrels)
        typed = _per_query(corpus, spec["typed"], qrels)
        shared = sorted(set(string) & set(typed))

        a = [typed[q][PRIMARY] for q in shared]
        b = [string[q][PRIMARY] for q in shared]
        overall = paired_bootstrap(a, b, rounds=B, seed=SEED)
        diffs = {q: typed[q][PRIMARY] - string[q][PRIMARY] for q in shared}

        cells.append({
            **spec,
            "queries": len(shared),
            "r2_string": round(sum(b) / len(b), 4),
            "r2_typed": round(sum(a) / len(a), 4),
            # §6 F3: both directions reported per cell whatever happens, so a recall gain
            # bought with precision cannot be shown as a clean win.
            "secondary": {
                m: {"string": round(sum(string[q][m] for q in shared) / len(shared), 4),
                    "typed": round(sum(typed[q][m] for q in shared) / len(shared), 4)}
                for m in GRAPH_MEASURES if m != PRIMARY
            },
            "prediction_a": overall,
            "prediction_b": _subset_contrast(diffs, _affected(corpus, extractor)),
            "mde": MDE[(corpus, extractor)],
        })

    # §4 and §5 each get their OWN Holm family across the same four cells.
    #
    # The two producers name their p-value differently — `paired_bootstrap` returns "p_value",
    # `_subset_contrast` returns "p" — and reading only "p" with a 1.0 default silently ran Holm
    # on [1,1,1,1] for prediction A and labelled four significant cells `no_advantage`. The
    # default is gone: a cell with no p-value is a bug, and it now says so instead of becoming
    # a null result.
    for key in ("prediction_a", "prediction_b"):
        ps = []
        for c in cells:
            r = c[key]
            if not r.get("resolved", True):
                ps.append(1.0)
                continue
            if "p_value" not in r and "p" not in r:
                raise KeyError(f"{key} for {c['corpus']}/{c['extractor']} carries no p-value; "
                               "refusing to Holm-correct a family with a missing member")
            ps.append(r["p_value"] if "p_value" in r else r["p"])
        for cell, adj in zip(cells, holm_adjusted(ps)):
            r = cell[key]
            lo, hi = r.get("ci95", [0.0, 0.0])
            r["p_holm"] = round(adj, 5)
            if not r.get("resolved", True):
                r["decision"] = "not_resolved"
            elif adj <= ALPHA and lo > 0:
                r["decision"] = "supported"
            elif adj <= ALPHA and hi < 0:
                r["decision"] = "against"
            elif (hi - lo) > cell["mde"]:
                # Registered in §5 before running: an interval wider than the effect the
                # sample could resolve is not a null, it is an absence of evidence.
                r["decision"] = "underpowered"
            else:
                r["decision"] = "no_advantage"

    return {
        "protocol": "protocols/005-typed-identity.md, tagged protocol-005",
        "primary": PRIMARY,
        "bootstrap": {"rounds": B, "seed": SEED, "alpha": ALPHA,
                      "correction": "holm-bonferroni, one family per prediction"},
        "cells": cells,
    }


if __name__ == "__main__":
    r = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
