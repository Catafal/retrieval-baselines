"""
Two §8.2 side-analyses that had no producing code — annotation agreement and extraction error
characterisation.

    make reproduce-003-diagnostics

Both were committed as artifacts with nothing in the repository able to regenerate them, alongside
the §8.1 gate and the oracle ceiling. Neither gates anything; both are cited when the entry
discusses what the extractor gets wrong, which is reason enough for a reader to be able to check
them.

NEITHER PROMOTES A HEADLINE. The registered primary for §8.2 stays exact-string set match. The
error analysis characterises failures; it does not introduce a relaxed-match score.
"""

import json
import time
from pathlib import Path

from rb.experiments.graph import extractor
from rb.experiments.graph.entity_types import WHITELIST, filter_entities
from rb.experiments.graph.extraction_score import normalise

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"


def load_sample() -> list[dict]:
    return [json.loads(l) for l in (OUT / "extraction-sample.jsonl").read_text().splitlines()
            if l.strip()]


def agreement(rows: list[dict]) -> dict:
    """Inter-rater agreement over the three-model panel, from the sample's own per-passage record.

    This is the file that disproves the note `extraction_score.score` used to publish -- that the
    reference was single-rater with no agreement available. It is recomputed here rather than
    trusted, since the contradiction went unnoticed through three audits.
    """
    jac = [r["rater_jaccard"] for r in rows if isinstance(r.get("rater_jaccard"), (int, float))]
    kept = sum(len(r["entities"]) for r in rows)
    return {
        "annotator": sorted({r["annotator"] for r in rows if r.get("annotator")})[0],
        "rule_card": sorted({r["rule_card"] for r in rows if r.get("rule_card")})[0],
        "raters": 3,
        "adjudication": ("entity kept when >=2 of 3 raters listed it, applied in deterministic "
                         "code"),
        "mean_pairwise_jaccard": round(sum(jac) / len(jac), 4) if jac else None,
        "passages": len(rows),
        "entities_kept": kept,
        "per_doc": [{"doc": r["doc_id"], "raters": 3, "jaccard": r.get("rater_jaccard"),
                     "kept": len(r["entities"])} for r in rows],
    }


def error_analysis(rows: list[dict]) -> dict:
    """Characterise the extractor's false positives and negatives against the reference set.

    DESCRIPTIVE ONLY. The reading below is that most false positives are boundary disagreements
    rather than hallucinations, which matters because the arm links by EXACT string: a boundary
    disagreement is a link failure even when the extractor located the right thing.
    """
    predicted = extractor.extract_many({r["doc_id"]: r["text"] for r in rows})
    fp_total = fn_total = 0
    fp_boundary = fp_title = fp_article = 0
    for r in rows:
        gold = {normalise(x) for x in r["entities"] if normalise(x)}
        pred = {normalise(t) for t, _ in filter_entities(predicted.get(r["doc_id"], []), WHITELIST)
                if normalise(t)}
        fps, fns = pred - gold, gold - pred
        fp_total += len(fps)
        fn_total += len(fns)
        title = normalise(r.get("title", ""))
        for f in fps:
            if any(f in g or g in f for g in gold):
                fp_boundary += 1
            if title and title in f:
                fp_title += 1
            if any(f == a + " " + g for g in gold for a in ("the", "a", "an")):
                fp_article += 1
    return {
        "status": ("ERROR ANALYSIS, computed after the primary diagnostic. Descriptive only: it "
                   "characterises the errors, it is NOT a competing metric and no relaxed-match "
                   "score is promoted to a headline. The registered primary remains exact-string "
                   "set match."),
        "false_positives": fp_total,
        "fp_substring_or_superstring_of_a_gold_entity": fp_boundary,
        "fp_boundary_share": round(fp_boundary / fp_total, 4) if fp_total else 0.0,
        "fp_containing_the_prepended_title": fp_title,
        "fp_that_are_a_gold_entity_with_a_leading_article": fp_article,
        "false_negatives": fn_total,
        "reading": (
            "A false positive that is a substring or superstring of a gold entity means the "
            "extractor LOCATED the right thing and disagreed about the span boundary. That still "
            "breaks this arm, because linking is exact-string: a boundary disagreement is a link "
            "failure. See protocol section 2b."
        ),
    }


def main() -> None:
    t0 = time.perf_counter()
    rows = load_sample()
    (OUT / "annotation-agreement.json").write_text(json.dumps(agreement(rows), indent=2) + "\n")
    err = error_analysis(rows)
    (OUT / "extraction-error-analysis.json").write_text(json.dumps(err, indent=2) + "\n")
    print(json.dumps({k: v for k, v in err.items() if k not in ("status", "reading")}, indent=2))
    print(f"seconds {time.perf_counter()-t0:.1f}")


if __name__ == "__main__":
    main()
