"""
Experiment 005's follow-up measurements — EXPLORATORY, not registered.

Everything here was computed in response to review, after the registered result was known. It
carries no registered decision. It exists as a producer rather than as numbers in prose because
the entry's contract is that no figure in it is typed by hand, and these figures are the ones the
entry leans on hardest:

  strata       why prediction B is close to circular — the unaffected stratum is almost all zeros
  headroom     why 2Wiki gains more per affected query than HotpotQA
  correlations whether the negative dose-response is really about merge size, or about difficulty
  cap          the pre-registered hub test (protocols/005-amendment-2), which falsified its own
               hypothesis

The dose-response coefficients themselves live in dose-response.json and are not recomputed here.
"""

import json
from pathlib import Path

from rb import datasets, metrics
from rb.experiments.graph import identity_coverage as ic
from rb.experiments.graph import pool2wiki
from rb.experiments.graph.dose_response import doses
from rb.experiments.graph.measures import GRAPH_MEASURES
from rb.stats import spearman_correlation

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results"
OUT = RESULTS / "005"
PRIMARY = "recall_2"
SEED = 20260820

CELLS = [
    ("hotpotqa", "pool", "spacy", "graph", "graph-typed"),
    ("hotpotqa", "pool", "glm", "graph-glm", "graph-glm-typed"),
    ("2wiki", "2wiki", "spacy", "graph", "graph-typed"),
    ("2wiki", "2wiki", "glm", "graph-glm", "graph-glm-typed"),
]


def _qrels(corpus: str):
    return datasets.load_qrels("hotpotqa") if corpus == "hotpotqa" else pool2wiki.build()[3]


def _per_query(sub: str, arm: str, qrels: dict) -> dict:
    run = {}
    for line in (RESULTS / "003" / sub / arm / "per_query.jsonl").read_text().splitlines():
        d = json.loads(line)
        docs = d["retrieved"]
        run[d["query_id"]] = {doc: float(len(docs) - i) for i, doc in enumerate(docs)}
    return metrics.score_ranked({q: qrels[q] for q in run}, run, GRAPH_MEASURES)


def _r2(sub: str, arm: str) -> float | None:
    """The headline R@2 an arm recorded, or None if that arm was never scored."""
    path = RESULTS / "003" / sub / arm / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["ranked"][PRIMARY]


def run(with_correlations: tuple[str, ...] = ("glm",)) -> dict:
    strata, headroom, corr = [], [], []

    for corpus, sub, extractor, string_arm, typed_arm in CELLS:
        qrels = _qrels(corpus)
        s = _per_query(sub, string_arm, qrels)
        t = _per_query(sub, typed_arm, qrels)
        affected = set(json.loads(
            (OUT / f"affected-{corpus}-{extractor}.json").read_text())["affected"])
        qs = sorted(set(s) & set(t))
        diff = {q: t[q][PRIMARY] - s[q][PRIMARY] for q in qs}

        inside = [diff[q] for q in qs if q in affected]
        outside = [diff[q] for q in qs if q not in affected]

        def summarise(v):
            return {"n": len(v),
                    "nonzero": round(sum(1 for x in v if x != 0) / len(v), 4) if v else 0.0,
                    "mean": round(sum(v) / len(v), 4) if v else 0.0}

        strata.append({"corpus": corpus, "extractor": extractor,
                       "affected": summarise(inside), "unaffected": summarise(outside)})

        # Headroom: affected queries start lower, so they have more room to recover. This is the
        # surviving partial account of why 2Wiki gains more per affected query.
        base_aff = [s[q][PRIMARY] for q in qs if q in affected]
        headroom.append({
            "corpus": corpus, "extractor": extractor,
            "baseline_r2_affected": round(sum(base_aff) / len(base_aff), 4) if base_aff else 0.0,
            "baseline_r2_all": round(sum(s[q][PRIMARY] for q in qs) / len(qs), 4),
        })

        # Correlations need the per-query dose, which for spaCy means re-extracting the corpus.
        # Restricted by default to the cached extractor so this stays a minutes-long job.
        if extractor in with_correlations:
            dose = doses(corpus, extractor)
            aq = sorted(q for q in qs if q in affected and q in dose)
            x = [float(dose[q]) for q in aq]
            base = [s[q][PRIMARY] for q in aq]
            dif = [diff[q] for q in aq]
            corr.append({
                "corpus": corpus, "extractor": extractor, "n": len(aq),
                # If dose correlated with baseline, the dose-response would be a difficulty
                # artefact. It does not, so the two effects are independent.
                "dose_vs_baseline": spearman_correlation(x, base, seed=SEED),
                # The ceiling: a query already scoring well can only fall.
                "baseline_vs_delta": spearman_correlation(base, dif, seed=SEED),
                "dose_vs_delta": spearman_correlation(x, dif, seed=SEED),
            })

    # The pre-registered hub test. Read from the scored arms rather than re-run.
    cap = []
    for corpus, sub, extractor, _, typed_arm in CELLS:
        capped = _r2(sub, f"{typed_arm}-capped")
        if capped is not None:
            cap.append({"corpus": corpus, "extractor": extractor,
                        "uncapped": _r2(sub, typed_arm), "capped": capped,
                        "delta": round(capped - _r2(sub, typed_arm), 4)})

    return {
        "status": "EXPLORATORY. Computed after the registered result was known, in response to "
                  "review. Carries no registered decision. The one exception is `cap`, whose test "
                  "was frozen in protocols/005-amendment-2-hub-cap.md before the arms existed — "
                  "and which falsified the explanation it was written for.",
        "strata": strata,
        "headroom": headroom,
        "correlations": corr,
        "cap": cap,
    }


if __name__ == "__main__":
    r = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "followups.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
