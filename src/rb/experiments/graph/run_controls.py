"""
Runs Experiment 003's two controls — protocols/003-graph-arm.md §8.2 and §8.3.

    make reproduce-003-controls

WHY THIS MODULE NOW HAS A main(). It did not, and nothing imported it — 87 lines with no
caller — yet it produced two artifacts this experiment publishes. That is the defect entry 001
was retracted for: a number with no command behind it. The rule written in response is that a
measured artifact must be reproducible by a named command, and until now two of 003's were not.

Both are computed AFTER `protocol-003` was tagged and before any retrieval is scored, which
is the order §8 requires: the extraction number exists before any ranking does, so it cannot
be read in the light of a result.
"""

import json
import time
from pathlib import Path

from rb import datasets
from rb.experiments.graph import extraction_score as es
from rb.experiments.graph import extractor, pool
from rb.experiments.graph.entity_types import SPACY_ONTONOTES_ENTS_F

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"


def diagnostic() -> dict:
    """§8.2 — spaCy against the model-annotated reference set. Reported, gates nothing."""
    rows = [json.loads(l) for l in (OUT / "extraction-sample.jsonl").read_text().splitlines() if l.strip()]
    gold = {r["doc_id"]: r["entities"] for r in rows}
    t0 = time.perf_counter()
    predicted = extractor.extract_many({r["doc_id"]: r["text"] for r in rows})
    elapsed = time.perf_counter() - t0
    result = es.score(gold, predicted)
    result["cost"] = {"seconds": round(elapsed, 3), "passages": len(rows)}
    result["manifest"] = extractor.manifest()
    result["published_ontonotes_ents_f"] = SPACY_ONTONOTES_ENTS_F
    result["published_reference_note"] = (
        "spaCy's own OntoNotes figure, measured by Explosion on newswire. Reported as CONTEXT "
        "only: this corpus is Wikipedia intros and the annotation scheme differs, so it is a "
        "figure measured under different conditions and gates nothing."
    )
    return result


def gate(entities_by_doc: dict[str, list[str]]) -> dict:
    """§8.3 — do a query's two gold documents share an extracted entity more often than a
    random document pair does?"""
    qrels = datasets.load_qrels("hotpotqa")
    pairs = [tuple(sorted(docs)) for docs in qrels.values() if len(docs) == 2]
    return es.bridge_reachability(entities_by_doc, pairs)


def graph_summary(entities_by_doc: dict[str, list[str]]) -> dict:
    """
    Shape of the extracted graph — the `graph` block of gate-and-seed.json.

    WHY THIS EXISTS NOW. The committed artifact already carried this block, but NO function in
    the repository produced it: `distinct_surface_entities` and `mean_entities_per_document`
    had zero occurrences anywhere in the source tree. It was written by code that no longer
    exists — the same defect main() was added to close, sitting inside the very file main() was
    added to fix. Reconstructed here and verified to reproduce the published values exactly
    (66,581 / 65,987 / 291,837 / 9.25) before being adopted; had it not reproduced them, the
    block would have been REMOVED from the artifact rather than quietly redefined.

    SURFACE FORMS, not normalised nodes — that is what the published figure counted, and it is
    why this number (291,837) exceeds the graph's node count (285,013).
    """
    docs = len(entities_by_doc)
    if not docs:
        return {"documents": 0, "documents_with_an_entity": 0,
                "distinct_surface_entities": 0, "mean_entities_per_document": 0.0}
    return {
        "documents": docs,
        "documents_with_an_entity": sum(1 for e in entities_by_doc.values() if e),
        "distinct_surface_entities": len({e for v in entities_by_doc.values() for e in v}),
        "mean_entities_per_document": round(
            sum(len(v) for v in entities_by_doc.values()) / docs, 2),
    }


def seed_match_rate(entities_by_doc: dict[str, list[str]]) -> dict:
    """How often a query's entities link to any node at all.

    Not a control, but the number §2b predicts will be low, and the entry cannot interpret a
    loss without it: an arm that never gets a seed has not been tested as a graph. This is the
    source of amendment-1's 0.823. It was private and unreferenced, which is how a published
    number ends up with no live producer; it is public and called by main() for that reason.
    """
    queries = datasets.load_queries("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")
    nodes = {es.normalise(e) for ents in entities_by_doc.values() for e in ents}
    matched = 0
    scored = 0
    for qid in sorted(qrels):
        q = queries.get(qid)
        if q is None:
            continue
        scored += 1
        ents = extractor.node_strings(extractor.extract(q))
        if any(es.normalise(e) in nodes for e in ents):
            matched += 1
    return {"queries": scored, "with_a_linked_entity": matched,
            "seed_match_rate": round(matched / scored, 4) if scored else 0.0}


def extract_pool(cache: Path = None) -> dict[str, list[str]]:
    """Whitelisted entities over the full pooled corpus, cached because it is the expensive
    step and both §8.3 and the retrieval run consume it."""
    cache = cache or (ROOT / "data" / "003-pool-entities.json")
    if cache.exists():
        return json.loads(cache.read_text())
    ctx = pool.load_distractor_context()
    corpus = datasets.load_corpus("hotpotqa")
    titles = datasets.load_titles("hotpotqa")
    pool_corpus, _ = pool.build(corpus, titles, ctx)
    t0 = time.perf_counter()
    raw = extractor.extract_many(pool_corpus)
    entities = {d: extractor.node_strings(e) for d, e in raw.items()}
    print(f"extracted {len(entities)} passages in {time.perf_counter()-t0:.0f}s")
    cache.write_text(json.dumps(entities))
    return entities


def main() -> None:
    """
    Both controls plus the graph summary, written where the entry cites them.

    ORDER IS §8's ORDER: the extraction diagnostic is computed before the gate, and both before
    any retrieval is scored, so neither can be read in the light of a result.

    VERIFICATION NOTE. `extract_pool` caches to data/003-pool-entities.json because extraction
    is the expensive step. To check that this command reproduces the committed artifacts,
    DELETE THAT CACHE FIRST — verifying against a warm cache replays the cache's contents
    rather than re-deriving them, and would pass even against a broken extractor.
    """
    OUT.mkdir(parents=True, exist_ok=True)

    diag = diagnostic()
    (OUT / "extraction-diagnostic.json").write_text(json.dumps(diag, indent=2) + "\n")
    print(json.dumps(diag["micro"], indent=2), flush=True)

    t0 = time.perf_counter()
    entities = extract_pool()
    payload = {
        "gate_8_3": gate(entities),
        "seed_match": seed_match_rate(entities),
        "graph": graph_summary(entities),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT / "gate-and-seed.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gate_passed": payload["gate_8_3"].get("passed"),
                      "seed_match_rate": payload["seed_match"]["seed_match_rate"]}, indent=2))


if __name__ == "__main__":
    main()
