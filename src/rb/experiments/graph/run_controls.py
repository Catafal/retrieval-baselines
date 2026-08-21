"""
Runs Experiment 003's two controls — protocols/003-graph-arm.md §8.2 and §8.3.

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


def _seed_match_rate(entities_by_doc: dict[str, list[str]]) -> dict:
    """How often a query's entities link to any node at all.

    Not a control, but the number §2b predicts will be low, and the entry cannot interpret a
    loss without it: an arm that never gets a seed has not been tested as a graph.
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
