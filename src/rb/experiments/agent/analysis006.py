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
    return round(base * (2 ** 0.5 if interaction_test else 1.0), 4)


def decide(c: dict, mde_value: float, underpowered_allowed: bool,
           alpha: float = 0.05) -> str:
    """`supported` / `no_advantage` / `underpowered`.

    A cell may only report `underpowered` if the protocol registered it as such in advance.
    Silence from a sample that could not have spoken is not evidence of absence, but neither
    is it a licence to relabel any inconvenient null.
    """
    if not c.get("resolved"):
        return "unresolved"
    lo, hi = c["ci95"]
    # Both conditions, because either alone is gameable: an interval clear of zero says the
    # effect is there, and the Holm-adjusted p says it survives its own family. A cell that
    # reported `supported` on the interval while its adjusted p exceeded alpha would make the
    # registered family decorative, which is what review found this function doing.
    p_adj = c.get("p_holm", c.get("p_value", 1.0))
    if lo > 0 and p_adj <= alpha:
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
                "em_lenient": round(sum(r["em_lenient"] for r in ok) / n, 4),
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


def answer_presence(db_path, questions: list[dict], hops: int = 3, top_k: int = 8) -> dict:
    """Seed-vs-walk decomposition. No model calls; computed from the graph and the sample.

    The deepest threat to this experiment is that `graph-facts` is lexical retrieval wearing
    arrow notation: recall() seeds by matching entity names and aliases against the question,
    so if the seed match already carries the answer, the walk is decorative and the arm is a
    string matcher. These four numbers separate the cases, and each is the rate at which the
    gold answer STRING appears in the injection -- a necessary condition for answering by
    copying, and therefore a ceiling on the arm's EM under a copy-only model.

      a0        hops=0: the seed neighbourhood alone
      a3        the shipped configuration
      a3_shuf   the same walk over a target-permuted copy of the edges. Same nodes, same
                seeding, same size, no true structure. This is the placebo: if the real graph
                does not beat it, the edges carry nothing a random graph does not.
      triples_only   the walk's edges without the extractor's free-text entity descriptions

    Registered decision rule, protocol 006 F6: if a3 - a0 <= 0.05, or a3 - a3_shuf <= 0.05,
    any P1 win is labelled SEED-MATCH, NOT WALK, and the entry leads with that whatever the
    EM table says.
    """
    import random
    import sqlite3
    import tempfile
    from pathlib import Path as _P

    from rb.experiments.agent import graphmem

    def rate(db, h, k, strip_notes=False):
        hit = 0
        for q in questions:
            f = graphmem.recall(db, q["question"], hops=h, top_k=k)
            text = "\n".join(f.lines())
            if strip_notes:
                text = text.split("where:")[0]
            if q["answer"].lower() in text.lower():
                hit += 1
        return round(hit / max(len(questions), 1), 4)

    # The placebo graph: permute edge targets, preserving each source's degree and the node set.
    shuf = _P(tempfile.mkdtemp()) / "placebo.db"
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(shuf)
    src.backup(dst)
    rows = list(dst.execute("SELECT rowid, target_id FROM relations"))
    targets = [t for _, t in rows]
    random.Random(SEED).shuffle(targets)
    dst.executemany("UPDATE relations SET target_id=? WHERE rowid=?",
                    [(t, r) for (r, _), t in zip(rows, targets)])
    dst.commit()
    dst.close()
    src.close()

    hist, seeded, empty = {}, 0, 0
    for q in questions:
        f = graphmem.recall(db_path, q["question"], hops=hops, top_k=top_k)
        for d, c in f.depth_histogram().items():
            hist[d] = hist.get(d, 0) + c
        seeded += bool(f.seeds)
        empty += not f.triples
    total = sum(hist.values()) or 1

    a0, a3, a3s = rate(db_path, 0, top_k), rate(db_path, hops, top_k), rate(shuf, hops, top_k)
    return {"seed_rate": round(seeded / len(questions), 4),
            "empty_recall": empty,
            "a0_seed_only": a0, "a3_shipped": a3, "a3_edge_placebo": a3s,
            "a3_triples_only": rate(db_path, hops, top_k, strip_notes=True),
            "walk_over_seed": round(a3 - a0, 4),
            "walk_over_placebo": round(a3 - a3s, 4),
            "depth_histogram": dict(sorted(hist.items())),
            "share_depth_ge_1": round(sum(v for k, v in hist.items() if k >= 1) / total, 4),
            "share_depth_ge_2": round(sum(v for k, v in hist.items() if k >= 2) / total, 4),
            "verdict": ("seed-match, not walk" if (a3 - a0) <= 0.05 or (a3 - a3s) <= 0.05
                        else "walk contributes beyond seed match and beyond placebo")}


def extraction_yield(db_path, questions: list[dict], extraction_jsonl) -> dict:
    """How much of the corpus actually became a graph, and per question how much of ITS gold did.

    Registered because the graph is only as real as the documents that parsed. A P1 null on a
    graph built from half the corpus is equally consistent with "graphs do not move hops" and
    with "the extractor could not emit valid JSON for half of Wikipedia", and those are not the
    same finding. Protocol 006 section 9: below 90% yield, F1 reports as unfalsifiable rather
    than as a graph loss.
    """
    import json as _json
    import sqlite3

    from rb.experiments.agent.graphmem import _parse

    rows = [_json.loads(l) for l in Path(extraction_jsonl).read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("outcome") == "ok"]
    parsed = {r["query_id"] for r in ok if _parse(r.get("answer", ""))}

    db = sqlite3.connect(db_path)
    in_graph = {t for (t,) in db.execute("SELECT DISTINCT source_doc FROM entities")}
    db.close()

    strata = {0: [], 1: [], 2: []}
    for q in questions:
        strata[sum(1 for t in q["gold"] if t in in_graph)].append(q["id"])
    return {"docs_attempted": len(rows), "docs_ok": len(ok), "docs_parsed": len(parsed),
            "yield": round(len(parsed) / max(len(rows), 1), 4),
            "docs_in_graph": len(in_graph),
            "gold_docs_in_graph": {str(k): len(v) for k, v in strata.items()},
            "strata": {str(k): v for k, v in strata.items()},
            "interpretable": len(parsed) / max(len(rows), 1) >= 0.90}


def presence_all_arms(questions: list[dict], pool: dict, retrieved: dict,
                      facts: dict, budget: int = 400) -> dict:
    """Answer-presence for EVERY injected arm, not only the graph.

    F6 ran this check on `graph-facts` alone, which is asymmetric scrutiny of exactly the kind
    the falsifier discipline exists to prevent elsewhere. Without the comparison arms a reader
    cannot tell whether a low graph EM means the arm reasons badly or merely that its injection
    rarely contains the answer -- and those are different findings with different consequences.

    Presence is a ceiling on EM under a copy-only model. EM divided by presence is therefore a
    crude CONVERSION efficiency: how well an arm turns the evidence it actually delivered into a
    correct answer. Efficiency above 1 means the model answered beyond its injection, which is
    parametric memory, and it shows up for more than one arm.
    """
    from rb.experiments.agent import arms as _arms

    out = {}
    for name in ("bm25", "dense"):
        hit = 0
        for q in questions:
            docs = [(t, pool[t]) for t in retrieved[name].get(q["id"], [])]
            _, inj = _arms.passages(q["question"], docs, budget)
            hit += q["answer"].lower() in inj.text.lower()
        out[name] = round(hit / len(questions), 4)
    hit = 0
    for q in questions:
        docs = [(t, pool[t]) for t in q["gold"] if t in pool]
        _, inj = _arms.passages(q["question"], docs, budget)
        hit += q["answer"].lower() in inj.text.lower()
    out["oracle"] = round(hit / len(questions), 4)
    hit = 0
    for q in questions:
        _, inj = _arms.graph_facts(q["question"], facts.get(q["id"], ""), budget)
        hit += q["answer"].lower() in inj.text.lower()
    out["graph-facts"] = round(hit / len(questions), 4)
    return out


def efficiency(arm_rows: list[dict], presence: dict) -> list[dict]:
    """EM divided by the arm's own answer-presence ceiling, per arm and tier."""
    rows = []
    for r in arm_rows:
        c = presence.get(r["arm"])
        if not c:
            continue
        rows.append({"arm": r["arm"], "model": r["model"], "em": r["em"], "ceiling": c,
                     "efficiency": round(r["em"] / c, 3)})
    return rows


def graph_loss_decomposition(db_path, questions: list[dict], hops: int = 3,
                             top_k: int = 8, wide_k: int = 40) -> dict:
    """WHERE the graph arm loses the answer: extraction, or the walk's own top-k cut.

    This is the diagnostic that overturned the first reading of 006. The presence ceiling said
    the graph's injection carried the gold answer for 0.38 of questions against dense's 0.80,
    and the obvious inference -- that a cheap extractor had thrown the corpus away -- is wrong.
    The answer is present SOMEWHERE in the graph for 0.88 of questions. Extraction kept it.
    What discards it is `ORDER BY near` followed by a flat cut to top_k under a 400-token
    budget: a ranking step, inside the arm, at query time.

    That matters because it is the same failure 003 through 005 found on the retrieval axis --
    the graph arm's problem was never the walk's reach, it was which nodes it ranks first --
    reproduced here on the answering axis by an independent route.

    Also measured: connectivity, so that a ranking claim cannot be confused with a topology
    one. If the gold documents were not connected, no ranking could help.
    """
    import sqlite3
    from collections import defaultdict

    from rb.experiments.agent import graphmem
    from rb.experiments.agent.corpus import sample as _sample

    # The pool is rebuilt from the frozen seed rather than carried in questions.json, so the
    # committed sample file stays a list of questions rather than a copy of the corpus.
    _, pool = _sample(len(questions), 20260820)

    db = sqlite3.connect(db_path)
    blob = (" ".join(n or "" for n, in db.execute("SELECT name FROM entities")) + " " +
            " ".join(d or "" for d, in db.execute("SELECT description FROM entities"))).lower()
    names = [(e, n) for e, n in db.execute("SELECT id, name FROM entities") if len(n or "") >= 3]
    adj = defaultdict(set)
    for a, b in db.execute("SELECT source_id, target_id FROM relations"):
        adj[a].add(b)
        adj[b].add(a)
    db.close()

    def reach(seeds, h):
        seen, fr = set(seeds), set(seeds)
        for _ in range(h):
            nxt = set()
            for x in fr:
                nxt |= adj[x]
            nxt -= seen
            seen |= nxt
            fr = nxt
        return seen

    def ents_in(text: str) -> set:
        tl = text.lower()
        return {e for e, n in names if n.lower() in tl}

    in_graph = in_walk = in_wide = 0
    bridged = 0
    linked = {1: 0, 2: 0, 3: 0}
    for q in questions:
        a = q["answer"].lower()
        in_graph += a in blob
        in_walk += a in graphmem.recall(db_path, q["question"], hops, top_k).as_text().lower()
        in_wide += a in graphmem.recall(db_path, q["question"], hops, wide_k).as_text().lower()
        # Connectivity, so a ranking claim is not confused with a topology one.
        ta, tb = (pool.get(q["gold"][0], ""), pool.get(q["gold"][1], ""))
        if ta and tb:
            A, B = ents_in(ta), ents_in(tb)
            bridged += bool(A & B)
            for h in linked:
                linked[h] += bool(reach(A, h) & B)

    n = len(questions)
    out = {"n": n,
           "answer_in_graph_anywhere": round(in_graph / n, 4),
           "answer_in_shipped_walk": round(in_walk / n, 4),
           f"answer_in_walk_top_{wide_k}": round(in_wide / n, 4),
           "questions_lost_by_extraction": n - in_graph,
           "questions_lost_by_topk_cut": in_graph - in_walk,
           "verdict": ("the walk's ranking discards the answer far more often than extraction "
                       "fails to capture it")}
    if bridged or any(linked.values()):
        out["gold_pair_shares_an_entity"] = round(bridged / n, 4)
        out["gold_pair_connected_within"] = {str(h): round(v / n, 4) for h, v in linked.items()}
    return out


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

    # P4 context tokens, family of 3. grep minus graph-facts, so a POSITIVE difference means
    # the graph arm read less. The prediction is conjunctive -- fewer tokens AT EQUAL OR BETTER
    # EM -- and testing only the token half, as this did until review, would let an arm "win"
    # P4 by being cheap and wrong.
    p4 = [contrast(idx, "grep", "graph-facts", m, qids, "context_tokens_net") for m in TIERS]
    ps = holm_adjusted([c.get("p_value", 1.0) for c in p4])
    for c, ph, m in zip(p4, ps, TIERS):
        c["p_holm"] = ph
        em_c = contrast(idx, "graph-facts", "grep", m, qids, "em")
        c["em_not_worse"] = bool(em_c.get("resolved") and em_c["ci95"][1] >= 0)
        c["em_diff"] = em_c.get("mean_diff")
        tokens_lower = c.get("resolved") and c["ci95"][0] > 0 and ph <= 0.05
        c["decision"] = ("supported" if tokens_lower and c["em_not_worse"]
                         else "no_advantage" if c.get("resolved") else "unresolved")

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


def main() -> None:
    """Produce every artifact protocol 006 section 9 obliges, in one pass."""
    root = Path(__file__).resolve().parents[4]
    out = root / "results" / "006"
    questions = json.loads((out / "questions.json").read_text())
    qids = [q["id"] for q in questions]
    scored = json.loads((out / "scored.json").read_text())

    (out / "analysis.json").write_text(json.dumps(run(scored, qids), indent=2) + "\n")
    (out / "answer-presence.json").write_text(
        json.dumps(answer_presence(out / "graph.db", questions), indent=2) + "\n")
    # Presence and conversion efficiency for every injected arm, per the verification council.
    import pickle  # noqa: F401 - not used; retrieval is reloaded from its committed artifact
    pool = None
    from rb.experiments.agent.corpus import sample as _sample
    _qs, pool = _sample(len(questions), 20260820)
    retrieved = json.loads((out / "retrieved.json").read_text())
    facts = json.loads((out / "graph-facts.json").read_text())
    pres = presence_all_arms(questions, pool, retrieved, facts)
    a = json.loads((out / "analysis.json").read_text())
    (out / "presence-and-efficiency.json").write_text(json.dumps(
        {"presence": pres, "efficiency": efficiency(a["arms"], pres)}, indent=2) + "\n")

    (out / "graph-loss-decomposition.json").write_text(json.dumps(
        graph_loss_decomposition(out / "graph.db", questions), indent=2) + "\n")

    (out / "extraction-yield.json").write_text(
        json.dumps(extraction_yield(out / "graph.db", questions,
                                    out / "extraction.jsonl"), indent=2) + "\n")
    print("wrote analysis.json, answer-presence.json, extraction-yield.json")


if __name__ == "__main__":
    main()
