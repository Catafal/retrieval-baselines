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
import math
import time
from pathlib import Path

from rb.experiments.graph import extraction_score as es
from rb.experiments.graph import llm_extractor as llm
from rb.experiments.graph import run as graph_run
from rb.experiments.graph.extractor import extract as spacy_extract
from rb.experiments.graph.extractor import node_strings
from rb.stats import percentile_ci

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
        scored = es.score(gold, p, rows=rows)
        result["passages"][label] = scored["micro"]
        # PER-PASSAGE COUNTS, PERSISTED. The first version of this producer wrote only ["micro"],
        # which made a paired interval on the passage difference impossible to reproduce from the
        # artifact — a reviewer had to simulate one. A claim that a setting "barely touched"
        # passages needs an interval, and an interval needs the pairs.
        result["passages"].setdefault("per_passage", {})[label] = {
            c["doc_id"]: {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"]}
            for c in scored["counts"]
        }
        result[f"{label}_seconds"] = round(time.perf_counter() - t0, 1)

    # spaCy on the same inputs, so the comparison is against the arm 003 published rather than
    # against nothing.
    sp_empty = sum(1 for q in qids if not node_strings(spacy_extract(queries[q])))
    result["queries"]["spacy"] = {
        "empty": sp_empty, "empty_rate": round(sp_empty / len(qids), 4),
        "mean_entities": round(
            sum(len(node_strings(spacy_extract(queries[q]))) for q in qids) / len(qids), 3),
    }
    sp = es.score(gold, {d: spacy_extract(t) for d, t in ptexts.items()}, rows=rows)
    result["passages"]["spacy"] = sp["micro"]

    # The paired difference the entry's "barely touched passages" claim rests on, with an
    # interval, computed from the persisted pairs rather than asserted from two point estimates.
    pp = result["passages"].get("per_passage") or {}
    if pp.get("reasoning_off") and pp.get("reasoning_on"):
        result["passages"]["paired_f1_difference"] = _paired_f1_interval(
            pp["reasoning_on"], pp["reasoning_off"])

    # Entities per passage, both extractors, over the SCORED corpus rather than this sample —
    # the entry uses this figure to argue the oracle bounded a definition of entity rather than
    # extraction, so it is load-bearing and was previously uncited.
    result["entities_per_passage"] = _entities_per_passage()

    # Intervals on the query rates, so the entry can state them instead of a bare proportion.
    # Wilson rather than normal-approximation: at 0 or 1 successes out of 60 the normal interval
    # is degenerate or runs below zero, and both of those cases occur here.
    for label in ("reasoning_off", "reasoning_on", "spacy"):
        row = result["queries"][label]
        row["empty_rate_ci95"] = _wilson(row["empty"], result["queries"]["n"])

    # What a sample this size can actually resolve, so "60 is enough" is a statement with a
    # number behind it rather than an assurance.
    result["queries"]["mde_at_80_power"] = _mde(result["queries"]["n"])
    return result


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return [0.0, 1.0]
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _mde(n: int, alpha_z: float = 1.959963984540054, power_z: float = 0.8416212335729143) -> float:
    """
    Smallest two-proportion gap this n could reliably detect at 80% power, worst-case variance.

    Reported so a reader can see whether the sample was adequate for the effect that was found,
    which is a different and weaker claim than the sample being adequate in general.
    """
    return round((alpha_z + power_z) * math.sqrt(2 * 0.25 / n), 4)


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _paired_f1_interval(a: dict, b: dict, rounds: int = 10000, seed: int = 20260820) -> dict:
    """
    Percentile bootstrap on micro-F1(a) - micro-F1(b), resampling PASSAGES.

    Paired and passage-level on purpose. Entities within a passage are not independent draws, so
    resampling entities would understate the variance; resampling passages keeps each document's
    entities together, which is the unit the extractor actually operates on.
    """
    import random

    docs = sorted(set(a) & set(b))
    rng = random.Random(seed)
    observed = _f1(*(sum(a[d][k] for d in docs) for k in ("tp", "fp", "fn"))) - \
        _f1(*(sum(b[d][k] for d in docs) for k in ("tp", "fp", "fn")))
    draws = []
    for _ in range(rounds):
        pick = [docs[rng.randrange(len(docs))] for _ in docs]
        draws.append(
            _f1(*(sum(a[d][k] for d in pick) for k in ("tp", "fp", "fn")))
            - _f1(*(sum(b[d][k] for d in pick) for k in ("tp", "fp", "fn"))))
    draws.sort()
    lo, hi = percentile_ci(draws)
    return {"observed": round(observed, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "resamples": rounds, "seed": seed, "unit": "passage", "n_passages": len(docs),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _entities_per_passage() -> dict:
    """Mean whitelisted entities per passage for both extractors, over the HotpotQA pool."""
    from rb.experiments.graph import run as graph_run
    from rb.experiments.graph.extractor import extract_many as spacy_many

    corpus, _, _, _ = graph_run.load_pool()
    glm = llm.extract_docs_offline(corpus)
    glm_mean = sum(len(node_strings(e)) for e in glm.values()) / len(glm)
    sp = spacy_many(corpus)
    sp_mean = sum(len(node_strings(e)) for e in sp.values()) / len(sp)
    return {"passages": len(corpus),
            "glm_mean_whitelisted": round(glm_mean, 2),
            "spacy_mean_whitelisted": round(sp_mean, 2)}


if __name__ == "__main__":
    r = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reasoning-ablation.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
