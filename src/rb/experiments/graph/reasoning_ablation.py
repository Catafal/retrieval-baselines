"""
The measurement experiment 004's entry leads on — protocol 004 amendment 4 §1.

WHY THIS IS A PRODUCER AND NOT A PARAGRAPH. The finding is that disabling a reasoning model's
reasoning silently destroyed its ability to extract from questions while leaving passages
untouched. That claim carried the entry's headline, and it existed only as numbers in an
amendment written from a session transcript. A published figure with no command behind it is what
entry 001 was retracted for.

Costs real API calls. The artifact is the record; `make` does not re-buy it.
"""

import json
import time
from pathlib import Path

from rb.experiments.graph import extraction_score as es
from rb.experiments.graph import llm_extractor as llm
from rb.experiments.graph import run as graph_run
from rb.experiments.graph.extractor import extract as spacy_extract
from rb.experiments.graph.extractor import node_strings

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "004"

QUERIES = 60      # questions, where the effect is
PASSAGES = 40     # annotated passages, where it is not
SEED_NOTE = "first N by sorted id, so the sample cannot be chosen after seeing the outcome"


def _extract(texts: dict[str, str], reasoning) -> dict[str, list[tuple[str, str]]]:
    """One pass over `texts` at a given reasoning setting, bypassing the cache.

    Bypassed deliberately: the cache is keyed by reasoning setting, so a cached read would
    answer with whichever setting was bought rather than the one under test.
    """
    out, ids = {}, sorted(texts)
    for start in range(0, len(ids), llm.BATCH):
        chunk = ids[start:start + llm.BATCH]
        ents, _ = llm._with_retry(llm._call, llm.http_client,
                                  [texts[d] for d in chunk], reasoning)
        out.update(dict(zip(chunk, ents)))
    return out


def run() -> dict:
    _, queries, _, _ = graph_run.load_pool()
    qids = sorted(queries)[:QUERIES]
    qtexts = {q: queries[q] for q in qids}

    rows = [json.loads(l) for l in
            (ROOT / "results/003/extraction-sample.jsonl").read_text().splitlines()][:PASSAGES]
    ptexts = {r["doc_id"]: r["text"] for r in rows}
    gold = {r["doc_id"]: r["entities"] for r in rows}

    result = {
        "status": ("Measured for protocol 004 amendment 4. Costs API calls; the artifact is the "
                   "record. Sample: " + SEED_NOTE),
        "model": llm.MODEL, "provider": llm.PROVIDER, "quantization": llm.QUANTIZATION,
        "queries": {"n": len(qids)}, "passages": {"n": len(ptexts)},
    }

    for label, reasoning in (("reasoning_off", llm.DOC_REASONING), ("reasoning_on", None)):
        t0 = time.perf_counter()
        q = _extract(qtexts, reasoning)
        empty = sum(1 for d in qids if not node_strings(q[d]))
        result["queries"][label] = {
            "empty": empty,
            "empty_rate": round(empty / len(qids), 4),
            "mean_entities": round(sum(len(node_strings(q[d])) for d in qids) / len(qids), 3),
        }
        p = _extract(ptexts, reasoning)
        result["passages"][label] = es.score(gold, p, rows=rows)["micro"]
        result[f"{label}_seconds"] = round(time.perf_counter() - t0, 1)

    # spaCy on the same inputs, so the comparison is against the arm 003 published rather than
    # against nothing.
    sp_empty = sum(1 for q in qids if not node_strings(spacy_extract(queries[q])))
    result["queries"]["spacy"] = {
        "empty": sp_empty, "empty_rate": round(sp_empty / len(qids), 4),
        "mean_entities": round(
            sum(len(node_strings(spacy_extract(queries[q]))) for q in qids) / len(qids), 3),
    }
    result["passages"]["spacy"] = es.score(
        gold, {d: spacy_extract(t) for d, t in ptexts.items()}, rows=rows)["micro"]
    return result


if __name__ == "__main__":
    r = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reasoning-ablation.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
