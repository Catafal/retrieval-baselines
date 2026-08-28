"""
Dose-response — EXPLORATORY, not registered in protocol 005.

WHY THIS EXISTS. Prediction B splits queries by whether identity changed which documents their
entities reach, and compares the typed-minus-string difference across that split. A validator
measured the unaffected stratum and found it is 98-99.5% EXACTLY ZERO. That is structurally
expected: the walk is a personalized PageRank seeded from the query's own entities, so a query
whose entities' reachability did not change is walking a nearly unchanged local subgraph and its
result barely can move.

So prediction B is close to circular. It largely establishes "the result changed where the
mechanism changed", which is most of the way to restating the definition of the stratification
variable. The residual is real information — 0.5-1.9% of unaffected queries do move, which is the
global specificity/degree reweighting leaking in — but it is not the mechanism evidence the
protocol wanted.

WHAT THIS DOES INSTEAD. Among AFFECTED queries only, ask whether the size of the effect scales
with the size of the reachability change. "Changed a little" versus "changed a lot" is not
recoverable from the retrieval mechanism by definition the way "changed versus did not" is, so a
positive dose-response is evidence the binary split cannot give.

The dose is measured per query as the total number of documents entering or leaving its entities'
reachable sets — the symmetric difference, summed over the query's entity keys.

REPORTED AS EXPLORATORY. It was designed after the registered result was known, so it cannot
carry a registered decision and does not get one. It is reported with that label attached.
"""

import json
from pathlib import Path

from rb import datasets, metrics
from rb.experiments.graph import identity_coverage as ic
from rb.experiments.graph import pool2wiki, redirects
from rb.experiments.graph.extraction_score import normalise
from rb.experiments.graph.linker import build_registry
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.stats import spearman_correlation

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results"
OUT = RESULTS / "005"
PRIMARY = "recall_2"
SEED = 20260820


def _per_query(corpus: str, arm: str, qrels: dict) -> dict:
    sub = "pool" if corpus == "hotpotqa" else "2wiki"
    run = {}
    for line in (RESULTS / "003" / sub / arm / "per_query.jsonl").read_text().splitlines():
        d = json.loads(line)
        docs = d["retrieved"]
        run[d["query_id"]] = {doc: float(len(docs) - i) for i, doc in enumerate(docs)}
    return metrics.score_ranked({q: qrels[q] for q in run}, run, GRAPH_MEASURES)


def doses(corpus: str, extractor: str) -> dict[str, int]:
    """Per query, how many documents entered or left its entities' reachable sets.

    Built from the same entities and the same registry the scored arms used, by the same
    reachability maps prediction B's membership test is derived from — so the dose is the
    magnitude of exactly the thing the binary indicator only records the presence of.
    """
    passages, queries, titles = ic._load(corpus)
    registry, _ = build_registry(redirects.load(corpus), titles)
    docs, qents = ic._entities(corpus, passages, queries, extractor)

    docs_string: dict[str, set] = {}
    docs_typed: dict[str, set] = {}
    for doc, ents in docs.items():
        for surface in ents:
            key = normalise(surface)
            if not key:
                continue
            docs_string.setdefault(key, set()).add(doc)
            docs_typed.setdefault(registry.get(key, key), set()).add(doc)

    out = {}
    for qid, ents in qents.items():
        total = 0
        for k in {normalise(s) for s in ents if normalise(s)}:
            before = docs_string.get(k, frozenset())
            after = docs_typed.get(registry.get(k, k), frozenset())
            total += len(before ^ after)
        out[qid] = total
    return out



def hubs(corpus: str, extractor: str, top: int = 12) -> dict:
    """
    The blast radius of each alias: how many documents its resolution moves.

    The dose-response came back NEGATIVE on all four cells — among affected queries, a larger
    reachability change predicts a WORSE outcome. This function finds out why, and the answer is
    that the registry is not one kind of edit. It is overwhelmingly small surgical merges plus a
    handful of catastrophic ones.

    `america -> united states` is a correct Wikipedia redirect. It is also retrieval poison: it
    produces a node present in a tenth of the corpus, whose node specificity (1/document
    frequency) is therefore near zero, and every query naming the US now seeds into it.
    """
    passages, queries, titles = ic._load(corpus)
    registry, _ = build_registry(redirects.load(corpus), titles)
    docs, _ = ic._entities(corpus, passages, queries, extractor)

    ds, dt = {}, {}
    for doc, ents in docs.items():
        for surface in ents:
            k = normalise(surface)
            if not k:
                continue
            ds.setdefault(k, set()).add(doc)
            dt.setdefault(registry.get(k, k), set()).add(doc)

    blast = sorted(
        ((len(ds.get(k, set()) ^ dt.get(canon, set())), k, canon)
         for k, canon in registry.items() if k in ds),
        reverse=True,
    )
    return {
        "corpus": corpus,
        "extractor": extractor,
        "aliases_present_in_corpus": len(blast),
        "moving_over_1000_docs": sum(1 for n, _, _ in blast if n > 1000),
        "moving_10_docs_or_fewer": sum(1 for n, _, _ in blast if n <= 10),
        "largest": [{"docs_moved": n, "alias": k, "canonical": c} for n, k, c in blast[:top]],
    }


def run() -> dict:
    cells = []
    for corpus, string_arm, typed_arm, extractor in [
        ("hotpotqa", "graph", "graph-typed", "spacy"),
        ("hotpotqa", "graph-glm", "graph-glm-typed", "glm"),
        ("2wiki", "graph", "graph-typed", "spacy"),
        ("2wiki", "graph-glm", "graph-glm-typed", "glm"),
    ]:
        qrels = datasets.load_qrels("hotpotqa") if corpus == "hotpotqa" else pool2wiki.build()[3]
        string = _per_query(corpus, string_arm, qrels)
        typed = _per_query(corpus, typed_arm, qrels)
        affected = set(json.loads(
            (OUT / f"affected-{corpus}-{extractor}.json").read_text())["affected"])

        dose = doses(corpus, extractor)
        # AFFECTED ONLY. Including the unaffected queries would put ~8,000 zeros on both axes and
        # manufacture a correlation out of the same near-degeneracy that makes prediction B
        # circular in the first place.
        qs = sorted(q for q in (set(string) & set(typed) & affected) if q in dose)
        x = [float(dose[q]) for q in qs]
        y = [typed[q][PRIMARY] - string[q][PRIMARY] for q in qs]

        rho = spearman_correlation(x, y, seed=SEED)
        cells.append({
            "corpus": corpus,
            "extractor": extractor,
            "n_affected": len(qs),
            "dose": {
                "min": int(min(x)) if x else 0,
                "median": int(sorted(x)[len(x) // 2]) if x else 0,
                "max": int(max(x)) if x else 0,
            },
            "spearman": rho,
            # A flat gain across dose sizes would mean the binary indicator was the whole story
            # and merging more changed nothing — which is what prediction B cannot distinguish.
            "nonzero_outcome_rate": round(sum(1 for v in y if v != 0) / len(y), 4) if y else 0.0,
        })

    return {
        "hubs": [hubs(c, e) for c, e in
                 (("hotpotqa", "glm"), ("2wiki", "glm"))],
        "status": "EXPLORATORY. Designed after the registered result was known, in response to a "
                  "validator finding that prediction B is close to circular. Carries no registered "
                  "decision and must not be reported as one.",
        "question": "Among affected queries, does the effect scale with the SIZE of the "
                    "reachability change, rather than merely its presence?",
        "cells": cells,
    }


if __name__ == "__main__":
    r = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dose-response.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
