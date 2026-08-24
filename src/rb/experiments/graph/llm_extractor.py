"""
The LLM entity extractor — protocols/004-llm-extractor.md §4, §5 and §6.

WHAT THIS IS, AND WHAT IT IS DELIBERATELY NOT. This produces ENTITIES on the same
`(surface_form, label)` contract as `extractor.extract`, not OpenIE triples. NB-20's roadmap row
says "LLM OpenIE"; triples would change the graph's node and edge semantics, so the arm would
differ from 003 in the extractor AND in the structure built on top of it, and no result could be
attributed to either. 002's amendment 2 claimed "the only variable is the encoder" while three
things differed. §5 of the protocol records the deviation and the reason.

THE CACHE IS THE ARTIFACT OF RECORD, NOT THE API. An LLM is not deterministic even at temperature
0 with a seed, so `make reproduce` cannot mean re-calling a paid endpoint. Every extraction is
written to a committed cache; a scored run reads it and treats a miss as a hard failure. The key
includes the prompt hash and the full model pin, so changing either cannot silently reuse
extractions produced under the old one.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

# --- the pin, frozen in protocol 004 §4 -------------------------------------------------------
# Provider and quantisation are part of the pin, not a routing preference. OpenRouter serves one
# model id from many backends at different numerical precisions: deepseek-v4-flash spans 17
# backends from fp4 to fp8, kimi-k2.5 spans 10 including int4, and each lab's own endpoint is
# often the quantised one. bf16 is the only tier where the weights served are the weights named.
MODEL = "z-ai/glm-4.7-flash"
PROVIDER = "DeepInfra"
QUANTIZATION = "bf16"
TEMPERATURE = 0
SEED = 20260820
BATCH = 10

# Concurrency is a throughput decision, not an experimental one: every call is independent, the
# cache is keyed by content, and results are written per batch, so the entities produced for a
# passage do not depend on what else is in flight. Serial extraction of 110,068 passages at the
# pilot's measured 12.6s per batch would take about 38 hours.
WORKERS = 12
MAX_RETRIES = 5

# GLM-4.7-Flash is a reasoning model. With reasoning left on, DeepInfra returns the structured
# payload in the `reasoning` field and sets `content` to None, so the extraction parses to nothing
# — and 30 of 46 completion tokens on a one-entity passage were reasoning, i.e. most of the output
# bill for a task that needs none. Disabled: `content` is populated and reasoning_tokens is 0.
#
# NOT `{"exclude": True}`, which is a trap: it returns content None AND reasoning None, losing the
# payload silently rather than loudly. Registered in protocols/004-amendment-1-reasoning-off.md.
REASONING = {"enabled": False}

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Frozen verbatim in protocol 004 §5. No few-shot examples and no chain of thought: both are free
# parameters that would be tuned against the outcome, and neither is registered.
PROMPT = (
    "Extract every named entity from each passage. For each entity return its exact surface form "
    "as it appears in the passage, and one label from this set: PERSON, NORP, FAC, ORG, GPE, LOC, "
    "PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, "
    "CARDINAL. Use the OntoNotes 5 definitions of these labels. Copy surface forms character for "
    "character from the passage; do not normalise, expand, or correct them. Return every "
    "occurrence of a distinct entity once. If a passage contains no named entities, return an "
    "empty list."
)

# Mirrors spaCy's OntoNotes alphabet so both extractors are scored on the same label set. The
# whitelist is NOT applied here — entity_types.filter_entities applies it downstream to the
# reference set and the predictions together, in one place, exactly as in 003.
LABELS = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
    "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "passages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string", "enum": LABELS},
                            },
                            "required": ["text", "label"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["index", "entities"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["passages"],
    "additionalProperties": False,
}

CACHE = Path(__file__).resolve().parents[4] / "results" / "004" / "extraction-cache.jsonl"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(passage: str) -> str:
    """
    Identifies an extraction by everything that could change it.

    Passage, model, provider, quantisation and prompt. Editing the prompt or repointing the pin
    therefore MISSES rather than silently returning extractions produced under the old one, which
    is the failure that would let a tuned prompt inherit a registered one's results.
    """
    return _sha("\x00".join(
        [passage, MODEL, PROVIDER, QUANTIZATION, _sha(PROMPT), json.dumps(REASONING, sort_keys=True)]
    ))


def load_cache(path: Path | None = None) -> dict[str, list[tuple[str, str]]]:
    """
    key -> entities. Absent file is an empty cache, not an error: the first run builds it.

    RESOLVED AT CALL TIME, NOT AS A DEFAULT ARGUMENT. `path: Path = CACHE` binds the module
    constant when the function is defined, so redirecting CACHE afterwards — which is exactly
    what a test must do — silently had no effect and two tests appended fake entities into the
    real cache. The cache is the artifact of record for a paid, pre-registered run, and it is
    append-only by design, so nothing would have flagged well-formed junk sitting in it.
    """
    path = CACHE if path is None else path
    if not path.exists():
        return {}
    out = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[row["key"]] = [(e["text"], e["label"]) for e in row["entities"]]
    log.info("extraction cache: %d entries from %s", len(out), path)
    return out


_CACHE_LOCK = threading.Lock()


def _with_retry(fn, *args):
    """
    Retry transient failures with exponential backoff and jitter.

    A run of 11,000 calls will meet rate limits and provider hiccups; without this a single 429
    three hours in would lose the run. Retries only transport-level failures — a schema violation
    or a refused pin is a real failure and must surface immediately rather than be retried into
    looking like success.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except (TimeoutError, OSError) as exc:  # urllib raises URLError(OSError) on 429/5xx
            if attempt == MAX_RETRIES - 1:
                raise
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            log.warning("transient failure (%s), retrying in %.1fs", type(exc).__name__, wait)
            time.sleep(wait)


def _append(rows: list[dict], path: Path | None = None) -> None:
    """Append-only. The cache is evidence; rewriting it in place would lose what was returned.

    Same call-time resolution as `load_cache`, and for the same reason."""
    path = CACHE if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialised: several workers append concurrently and interleaved writes would corrupt lines.
    with _CACHE_LOCK, path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _call(client, batch: list[str]) -> tuple[list[list[tuple[str, str]]], dict]:
    """One API call over `batch` passages. Returns per-passage entities and usage."""
    numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(batch))
    body = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "reasoning": REASONING,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": numbered},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "entities", "strict": True, "schema": SCHEMA},
        },
        # allow_fallbacks=False so a silent reroute FAILS the run rather than quietly changing
        # the numbers. A pin that degrades gracefully is not a pin.
        "provider": {
            "only": [PROVIDER],
            "quantizations": [QUANTIZATION],
            "allow_fallbacks": False,
        },
    }
    data = client(ENDPOINT, body)
    parsed = json.loads(data["choices"][0]["message"]["content"])

    by_index = {p["index"]: p["entities"] for p in parsed["passages"]}
    # A model that drops a passage must not silently shift every later passage's entities onto
    # its neighbour — the misalignment that would still look plausible in the artifact.
    out = [[(e["text"], e["label"]) for e in by_index.get(i, [])] for i in range(len(batch))]
    return out, data.get("usage", {})


def extract_many(texts: dict[str, str], client=None, offline: bool = False) -> dict[str, list[tuple[str, str]]]:
    """
    doc_id -> entities, reading the cache first and calling only for misses.

    `offline=True` is the scored-run mode required by protocol §6: a cache miss raises instead of
    reaching for the network, so a scored number can never come from an uncached call.
    """
    cache = load_cache()
    todo = {d: t for d, t in texts.items() if cache_key(t) not in cache}

    if todo and offline:
        raise RuntimeError(
            f"{len(todo)} passages are not in the extraction cache and offline=True. "
            "A scored run reads the cache; it does not call the API. See protocol 004 §6."
        )

    if todo:
        if client is None:
            raise ValueError("client is required when the cache does not cover every passage")
        ids = sorted(todo)
        chunks = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
        log.info("extracting %d uncached passages in %d batches, %d workers",
                 len(ids), len(chunks), WORKERS)
        done = [0]

        def run(chunk):
            ents, _ = _with_retry(_call, client, [todo[d] for d in chunk])
            rows = []
            for doc_id, e in zip(chunk, ents):
                key = cache_key(todo[doc_id])
                rows.append({"key": key, "doc_id": doc_id,
                             "entities": [{"text": t, "label": l} for t, l in e]})
            # Written before the in-memory cache is updated, so a crash leaves the file as the
            # authority and a resumed run skips exactly what was persisted, never more.
            _append(rows)
            with _CACHE_LOCK:
                for doc_id, e in zip(chunk, ents):
                    cache[cache_key(todo[doc_id])] = e
                done[0] += len(chunk)
                if done[0] % (BATCH * 50) < BATCH:
                    log.info("  %d/%d passages", done[0], len(ids))

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            # list() forces every future to be consumed, so an exception in any worker propagates
            # rather than being swallowed by the executor shutting down quietly.
            list(pool.map(run, chunks))

    return {d: cache[cache_key(t)] for d, t in texts.items()}


def http_client(endpoint: str, body: dict) -> dict:
    """
    The real transport, kept separate so every test above runs without a key or a network.

    Reads OPENROUTER_API_KEY at call time rather than import time: importing this module must not
    require a key, or the offline path and the tests would need one to do nothing.
    """
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())
