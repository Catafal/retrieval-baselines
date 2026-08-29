"""
Experiment 006's registered analysis — protocol 006 §5 and §6.

THE PAIRING LIVES IN THE VALUES, NOT IN THE ESTIMATOR. Every contrast here is reduced to one
scalar per QUESTION before it reaches `stats.paired_bootstrap`, which then resamples question
indices. That is what keeps a fully crossed repeated-measures design paired at the query
level: a bootstrap round that draws question i draws every one of its arm-by-tier cells
together. Handing the estimator two unpaired arrays instead would silently treat tiers as
independent samples and overstate precision.

The interaction (P3) is a DOUBLE difference computed per question:

    d_i = [EM(graph-facts, haiku, i) - EM(grep, haiku, i)]
        - [EM(graph-facts, opus,  i) - EM(grep, opus,  i)]

and is then the same one-sample mean as every other contrast. No new estimator is needed; the
correctness burden is entirely on building d_i, which is exactly the class of error that
005-amendment-1 records and that the p/p_value key bug in analysis005.py nearly shipped.

The realised discordance rate is reported alongside every contrast so a reader can check the
MDE column of §6 that actually applies, rather than the one the protocol guessed.
"""

import json
from collections import defaultdict
from pathlib import Path

from rb.stats import holm_adjusted, paired_bootstrap

SEED = 20260828
B = 10_000
TIERS = ("haiku", "sonnet", "opus")
ARMS = ("closed-book", "grep", "bm25", "dense", "oracle", "graph-facts")

# Infrastructure failures are excluded from the denominator; a model that ran out of turns is
# not, because that is a fact about the model. Protocol 006 §8.
EXCLUDED_OUTCOMES = {"timeout", "api_error", "no_json"}


def index(rows: list[dict]) -> dict:
    """(arm, model, query_id) -> scored row, keeping only the last attempt of each."""
    out = {}
    for r in rows:
        out[(r["arm"], r["model"], r["query_id"])] = r
    return out


def usable(r: dict | None) -> bool:
    return r is not None and r.get("outcome") not in EXCLUDED_OUTCOMES


def paired(idx: dict, arm_a: str, arm_b: str, model: str, qids: list[str],
           field: str = "em") -> tuple[list[float], list[float], list[str]]:
    """Values for two arms over the questions where BOTH produced a usable call."""
    a, b, kept = [], [], []
    for q in qids:
        ra, rb = idx.get((arm_a, model, q)), idx.get((arm_b, model, q))
        if usable(ra) and usable(rb):
            a.append(float(ra[field]))
            b.append(float(rb[field]))
            kept.append(q)
    return a, b, kept


def discordance(a: list[float], b: list[float]) -> float:
    """Share of pairs where the two arms disagree. Drives the MDE that actually applies."""
    if not a:
        return 0.0
    return round(sum(1 for x, y in zip(a, b) if x != y) / len(a), 4)


def contrast(idx: dict, arm_a: str, arm_b: str, model: str, qids: list[str],
             field: str = "em") -> dict:
    a, b, kept = paired(idx, arm_a, arm_b, model, qids, field)
    if len(kept) < 2:
        return {"n": len(kept), "resolved": False, "reason": "too few paired observations"}
    r = paired_bootstrap(a, b, rounds=B, seed=SEED)
    return {"arm_a": arm_a, "arm_b": arm_b, "model": model, "field": field, "n": len(kept),
            "mean_a": round(sum(a) / len(a), 4), "mean_b": round(sum(b) / len(b), 4),
            "mean_diff": round(sum(x - y for x, y in zip(a, b)) / len(a), 4),
            "ci95": [round(v, 4) for v in r["ci95"]], "p_value": r["p_value"],
            "discordance": discordance(a, b), "resolved": True}


def interaction(idx: dict, arm_a: str, arm_b: str, weak: str, strong: str,
                qids: list[str], field: str = "em") -> dict:
    """P3. One double difference per question, then the same one-sample bootstrap."""
    d, zeros, kept = [], [], []
    for q in qids:
        cells = [idx.get((arm, m, q)) for m in (weak, strong) for arm in (arm_a, arm_b)]
        if not all(usable(c) for c in cells):
            continue
        wa, wb, sa, sb = (float(c[field]) for c in cells)
        d.append((wa - wb) - (sa - sb))
        zeros.append(0.0)
        kept.append(q)
    if len(kept) < 2:
        return {"n": len(kept), "resolved": False, "reason": "too few complete question blocks"}
    r = paired_bootstrap(d, zeros, rounds=B, seed=SEED)
    n = len(kept)
    return {"weak": weak, "strong": strong, "arm_a": arm_a, "arm_b": arm_b, "n": n,
            "mean_diff": round(sum(d) / n, 4),
            "ci95": [round(v, 4) for v in r["ci95"]], "p_value": r["p_value"],
            "nonzero": round(sum(1 for x in d if x != 0) / n, 4), "resolved": True}


def mde(n: int, disc: float, interaction_test: bool = False) -> float:
    """Protocol §6: 2.8016*sqrt(d/n), times sqrt(2) for the double difference.

    Discordant-pair based, NOT a two-proportion formula -- using the latter on a differenced
    statistic is the estimand error recorded in 005-amendment-1.
    """
    if n <= 0:
        return float("inf")
    base = 2.8016 * (disc / n) ** 0.5
    return round(base * (2 ** 0.5 if interaction_test else base / base), 4) if not interaction_test \
        else round(base * (2 ** 0.5), 4)


def decide(c: dict, mde_value: float, underpowered_allowed: bool) -> str:
    """`supported` / `no_advantage` / `underpowered`.

    A cell may only report `underpowered` if the protocol registered it as such in advance.
    Silence from a sample that could not have spoken is not evidence of absence, but neither
    is it a licence to relabel any inconvenient null.
    """
    if not c.get("resolved"):
        return "unresolved"
    lo, hi = c["ci95"]
    if lo > 0:
        return "supported"
    if underpowered_allowed and lo <= 0 <= hi and (hi - lo) > mde_value:
        return "underpowered"
    return "no_advantage"


def resistant(idx: dict, model: str, qids: list[str]) -> list[str]:
    """Protocol §3: the questions THIS tier got wrong closed-book. Per tier, never recomputed."""
    out = []
    for q in qids:
        r = idx.get(("closed-book", model, q))
        if usable(r) and not r["em"]:
            out.append(q)
    return out


def arm_table(idx: dict, qids: list[str]) -> list[dict]:
    """Every arm x tier, reported whichever way the result goes. Protocol §9."""
    rows = []
    for model in TIERS:
        for arm in ARMS:
            rs = [idx[(arm, model, q)] for q in qids if (arm, model, q) in idx]
            ok = [r for r in rs if usable(r)]
            if not ok:
                continue
            n = len(ok)
            rows.append({
                "arm": arm, "model": model, "n": n,
                "em": round(sum(r["em"] for r in ok) / n, 4),
                "em_strict": round(sum(r["em_strict"] for r in ok) / n, 4),
                "f1": round(sum(r["f1"] for r in ok) / n, 4),
                "abstained": round(sum(r["abstained"] for r in ok) / n, 4),
                "max_turns_rate": round(sum(1 for r in rs if r["outcome"] == "max_turns")
                                        / max(len(rs), 1), 4),
                "excluded": len(rs) - n,
                "turns": round(sum(r["num_turns"] for r in ok) / n, 2),
                "context_tokens": round(sum(r["input_tokens"] + r["cache_read_tokens"]
                                            + r["cache_creation_tokens"] for r in ok) / n, 1),
                "cost_usd": round(sum(r["cost_usd"] for r in ok), 4),
                "latency_ms": round(sum(r["duration_ms"] for r in ok) / n, 1),
            })
    return rows


def run(scored: list[dict], qids: list[str]) -> dict:
    idx = index(scored)

    # Harness overhead: the context a zero-context arm still pays. Subtracted from the P4
    # measure so CLI overhead is not reported as corpus the model read.
    cb = [r for r in scored if r["arm"] == "closed-book" and usable(r)]
    baseline = round(sum(r["input_tokens"] + r["cache_read_tokens"] + r["cache_creation_tokens"]
                         for r in cb) / len(cb), 1) if cb else 0.0

    strata = {"all": qids}
    for m in TIERS:
        strata[f"resistant-{m}"] = resistant(idx, m, qids)

    # P1 primary, on all questions, haiku only.
    p1 = contrast(idx, "graph-facts", "grep", "haiku", qids)
    p1_mde = mde(p1.get("n", 0), p1.get("discordance", 0.3))
    p1["mde"] = p1_mde
    p1["decision"] = decide(p1, p1_mde, underpowered_allowed=False)

    # P2 family of 3.
    p2 = [contrast(idx, "graph-facts", "grep", m, qids) for m in TIERS]
    ps = holm_adjusted([c.get("p_value", 1.0) for c in p2])
    for c, ph in zip(p2, ps):
        c["p_holm"] = ph
        c["mde"] = mde(c.get("n", 0), c.get("discordance", 0.3))
        c["decision"] = decide(c, c["mde"], underpowered_allowed=False)

    # P3 interaction, registered underpowered in advance.
    p3 = interaction(idx, "graph-facts", "grep", "haiku", "opus", qids)
    p3_disc = p3.get("nonzero", 0.3)
    p3["mde"] = mde(p3.get("n", 0), p3_disc, interaction_test=True)
    p3["decision"] = decide(p3, p3["mde"], underpowered_allowed=True)

    # P4 context tokens, family of 3. Lower is better, so the sign is inverted relative to EM.
    p4 = []
    for m in TIERS:
        a, b, kept = paired(idx, "grep", "graph-facts", m, qids, "context_tokens_net")
        p4.append(contrast(idx, "grep", "graph-facts", m, qids, "context_tokens_net")
                  if kept else {"model": m, "resolved": False, "reason": "field missing"})
    ps = holm_adjusted([c.get("p_value", 1.0) for c in p4])
    for c, ph in zip(p4, ps):
        c["p_holm"] = ph

    # P5 adversarial: graph-facts vs dense.
    p5 = [contrast(idx, "graph-facts", "dense", m, qids) for m in TIERS]
    ps = holm_adjusted([c.get("p_value", 1.0) for c in p5])
    for c, ph in zip(p5, ps):
        c["p_holm"] = ph
        c["decision"] = decide(c, mde(c.get("n", 0), c.get("discordance", 0.3)), False)

    # Exploratory: the mechanism guard (F2) and the strawman guard (F4).
    explor = {
        "oracle_vs_graph": [contrast(idx, "oracle", "graph-facts", m, qids) for m in TIERS],
        "grep_vs_oracle": [contrast(idx, "grep", "oracle", m, qids) for m in TIERS],
        "graph_vs_closed_book": [contrast(idx, "graph-facts", "closed-book", m, qids)
                                 for m in TIERS],
        "by_stratum": {name: arm_table(idx, ids) for name, ids in strata.items()
                       if name != "all"},
    }

    return {"harness_baseline_context_tokens": baseline,
            "strata_sizes": {k: len(v) for k, v in strata.items()},
            "arms": arm_table(idx, qids),
            "p1_primary": p1, "p2_family": p2, "p3_interaction": p3,
            "p4_context_tokens": p4, "p5_adversarial": p5,
            "exploratory": explor}


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[4]
    scored = json.loads((root / "results/006/scored.json").read_text())
    qids = [q["id"] for q in json.loads((root / "results/006/questions.json").read_text())]
    out = run(scored, qids)
    (root / "results/006/analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "exploratory"}, indent=2)[:3000])
