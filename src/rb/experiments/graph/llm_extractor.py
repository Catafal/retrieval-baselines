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

import functools
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)


class MalformedResponse(RuntimeError):
    """
    A call returned something that is not the agreed JSON.

    Its own type because it is RETRYABLE and a schema violation is not. The first full run died
    eight minutes in, at 1,300 of 110,068 passages, when one batch came back truncated
    ("Unterminated string at char 49119") and `json.loads` raised straight through
    `ThreadPoolExecutor.map`. A run of 11,000 calls will meet a bad response; dying on the first
    one and discarding the work in flight is a defect in the harness, not a fact about the
    provider.
    """

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

# Bounds one call's output. The pilot measured 197 completion tokens per passage, so a batch of 10
# needs about 2,000; 8,000 leaves room for a dense passage without letting a degenerate generation
# run unbounded. A response that hits this ceiling comes back as truncated JSON, which is now a
# retryable failure rather than a crash — see MalformedResponse.
MAX_TOKENS = 8000

# GLM-4.7-Flash is a reasoning model. With reasoning left on, DeepInfra returns the structured
# payload in the `reasoning` field and sets `content` to None, so the extraction parses to nothing
# — and 30 of 46 completion tokens on a one-entity passage were reasoning, i.e. most of the output
# bill for a task that needs none. Disabled: `content` is populated and reasoning_tokens is 0.
#
# NOT `{"exclude": True}`, which is a trap: it returns content None AND reasoning None, losing the
# payload silently rather than loudly. Registered in protocols/004-amendment-1-reasoning-off.md.
# SUPERSEDED BY AMENDMENT 4 FOR QUERIES. Amendment 1 disabled reasoning to fix a TRANSPORT
# problem — DeepInfra returns the payload in `reasoning` with `content` None, and the parser only
# read `content`. Disabling reasoning made `content` populate, so the parse worked. The correct
# fix was one line in the parser; changing the model's behaviour to suit the parser cost it an
# entire input type. With reasoning off, 9 of 20 questions extract NOTHING; with it on, 0 of 20.
#
# Documents keep the setting they were bought under. Measured on the same gold as the pilot,
# 40 passages: reasoning off F1 0.8255, reasoning on F1 0.8224 — a 0.003 difference, against
# spaCy's 0.6703. The setting does not matter on passages and is decisive on questions.
DOC_REASONING = {"enabled": False}

# None means the field is omitted from the request, which is the model's default and leaves
# reasoning ON. `None` is therefore a MEANINGFUL value here, not "unspecified" — hence the
# sentinel below, so a caller can ask for reasoning-on without it being mistaken for a default.
QUERY_REASONING = None

_UNSET = object()

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


def cache_key(passage: str, reasoning=_UNSET) -> str:
    """
    Identifies an extraction by everything that could change it.

    Passage, model, provider, quantisation and prompt. Editing the prompt or repointing the pin
    therefore MISSES rather than silently returning extractions produced under the old one, which
    is the failure that would let a tuned prompt inherit a registered one's results.
    """
    if reasoning is _UNSET:
        reasoning = DOC_REASONING
    return _sha("\x00".join(
        [passage, MODEL, PROVIDER, QUANTIZATION, _sha(PROMPT), json.dumps(reasoning, sort_keys=True)]
    ))


# Matches one {"text": ..., "label": ...} object, tolerating escaped quotes inside the surface
# form. Deliberately a regex over a partial document rather than a JSON parse: the input this runs
# on is by definition truncated, so no parser will accept it.
_PAIR_RE = re.compile(r'\{"text":\s*"((?:[^"\\]|\\.)*)",\s*"label":\s*"([A-Z_]+)"\}')


def salvage_pairs(content: str) -> list[tuple[str, str]]:
    """Every complete (text, label) object in a truncated response, in order, WITH duplicates."""
    return _PAIR_RE.findall(content)


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
        except (TimeoutError, OSError, MalformedResponse) as exc:  # URLError(OSError) on 429/5xx
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


def _call(client, batch: list[str], reasoning=_UNSET) -> tuple[list[list[tuple[str, str]]], dict]:
    """One API call over `batch` passages. Returns per-passage entities and usage."""
    reasoning = DOC_REASONING if reasoning is _UNSET else reasoning
    numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(batch))
    body = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
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
    # Omitted entirely when None: that is the model's default and the only way to leave
    # reasoning ON, which questions require.
    if reasoning is not None:
        body["reasoning"] = reasoning

    data = client(ENDPOINT, body)
    choice = data["choices"][0]
    # THE FIX AMENDMENT 1 SHOULD HAVE BEEN. Some providers return the structured payload in
    # `reasoning` and leave `content` None; reading only `content` is what made a working
    # extraction look like an empty one, and the "fix" for that — disabling reasoning — cost the
    # model its ability to handle questions at all. Read whichever field carries it.
    msg = choice["message"]
    content = msg.get("content") or msg.get("reasoning")

    # `finish_reason == "length"` says the generation was cut off, so the JSON is truncated by
    # construction. Named explicitly rather than left to surface as a parse error, because the two
    # need different responses: truncation means the batch is too large for the ceiling, a parse
    # error on a complete response means the model emitted something else.
    if choice.get("finish_reason") == "length":
        exc = MalformedResponse(f"response truncated at max_tokens={MAX_TOKENS}")
        # Carried on the exception so the failure path can salvage from THIS response. Re-asking
        # would be a different call, and picking the probe that recovers something is selecting
        # the input that gives the wanted output. See protocol 004 amendment 2 section 4.
        exc.truncated_content = content or ""
        raise exc
    if not content:
        raise MalformedResponse(f"empty content (finish_reason={choice.get('finish_reason')!r})")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MalformedResponse(f"content is not JSON: {exc}") from exc

    by_index = {p["index"]: p["entities"] for p in parsed["passages"]}
    # A model that drops a passage must not silently shift every later passage's entities onto
    # its neighbour — the misalignment that would still look plausible in the artifact.
    out = [[(e["text"], e["label"]) for e in by_index.get(i, [])] for i in range(len(batch))]
    return out, data.get("usage", {})


def extract_many(texts: dict[str, str], client=None, offline: bool = False,
                 reasoning=_UNSET) -> dict[str, list[tuple[str, str]]]:
    """
    doc_id -> entities, reading the cache first and calling only for misses.

    `offline=True` is the scored-run mode required by protocol §6: a cache miss raises instead of
    reaching for the network, so a scored number can never come from an uncached call.
    """
    cache = load_cache()
    todo = {d: t for d, t in texts.items() if cache_key(t, reasoning) not in cache}

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

        failures: dict[str, str] = {}
        salvaged: dict[str, int] = {}

        def persist(chunk, ents):
            rows = [{"key": cache_key(todo[d], reasoning), "doc_id": d,
                     "entities": [{"text": t, "label": l} for t, l in e]}
                    for d, e in zip(chunk, ents)]
            # Written before the in-memory cache is updated, so a crash leaves the file as the
            # authority and a resumed run skips exactly what was persisted, never more.
            _append(rows)
            with _CACHE_LOCK:
                for d, e in zip(chunk, ents):
                    cache[cache_key(todo[d], reasoning)] = e
                done[0] += len(chunk)
                if done[0] % (BATCH * 50) < BATCH:
                    log.info("  %d/%d passages", done[0], len(ids))

        def run(chunk):
            """
            One batch, with the batch itself as the unit of isolation.

            Retries first. If a multi-passage batch still fails, it is split to single passages so
            one pathological input cannot cost its nine neighbours their work. A single passage
            that still fails is RECORDED, not cached: writing empty entities would make a failed
            extraction indistinguishable from a passage that genuinely contains no entities, and
            that passage would then contribute nothing to the graph while looking normal.
            """
            try:
                ents, _ = _with_retry(_call, client, [todo[d] for d in chunk], reasoning)
                persist(chunk, ents)
                return
            except Exception as exc:
                if len(chunk) == 1:
                    doc_id = chunk[0]
                    # protocol 004 amendment 2: recover the complete entity objects from the
                    # truncated response. The model emits its full distinct list and then loops
                    # on an ambiguous member, so the prefix carries everything the graph would
                    # have received — the graph dedupes nodes by string regardless.
                    pairs = salvage_pairs(getattr(exc, "truncated_content", "") or "")
                    seen: list[tuple[str, str]] = []
                    for pair in pairs:
                        if pair not in seen:
                            seen.append(pair)
                    if seen:
                        _append([{"key": cache_key(todo[doc_id], reasoning), "doc_id": doc_id,
                                  "salvaged": True,
                                  "entities": [{"text": t, "label": l} for t, l in seen]}])
                        with _CACHE_LOCK:
                            cache[cache_key(todo[doc_id], reasoning)] = seen
                            salvaged[doc_id] = len(seen)
                            done[0] += 1
                        log.warning("passage %s salvaged %d distinct entities from a truncated "
                                    "response (%d objects emitted)", doc_id, len(seen), len(pairs))
                        return
                    # Nothing complete in the response: excluded, never cached as empty.
                    with _CACHE_LOCK:
                        failures[doc_id] = f"{type(exc).__name__}: {exc}"
                    log.error("passage %s failed after %d retries and salvaged nothing: %s",
                              doc_id, MAX_RETRIES, exc)
                    return
                log.warning("batch of %d failed (%s); splitting to isolate", len(chunk), exc)

            for doc_id in chunk:
                run([doc_id])

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            # list() forces every future to be consumed, so an exception in any worker propagates
            # rather than being swallowed by the executor shutting down quietly.
            list(pool.map(run, chunks))

        if salvaged:
            log.warning("%d passages salvaged from truncated responses: %s",
                        len(salvaged), dict(list(salvaged.items())[:8]))
        if failures:
            # Raised at the END, so the run banks every passage it could extract and reports the
            # exact set that needs attention, rather than dying on the first bad one and
            # discarding thousands of successful calls that were already paid for.
            # Recorded, not raised. Amendment 2 registers exclusion as the handling for a
            # passage that salvages nothing, so stopping the pipeline here would refuse a case
            # the protocol now covers. The ids are committed so the exclusion is checkable.
            out = Path(str(CACHE.parent / "extraction-failures.json"))
            existing = json.loads(out.read_text()) if out.exists() else {}
            existing.update(failures)
            out.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
            log.error("%d of %d passages excluded, ids written to %s",
                      len(failures), len(ids), out)

    # Excluded passages are ABSENT from the result, not present-and-empty. A caller building a
    # graph must be able to tell "this passage yielded no entities" from "this passage was never
    # extracted", and the count of missing keys is the exclusion count the entry reports.
    return {d: cache[cache_key(t, reasoning)] for d, t in texts.items()
            if cache_key(t, reasoning) in cache}


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


# --- adapters for the scored run -------------------------------------------------------------
# Both read the committed cache and never call the API: protocol 004 §6 makes the cache the
# artifact of record, so a scored number cannot come from a live generation. A cache miss raises.


def extract_docs_offline(corpus: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """doc_id -> entities, for GraphRetriever.fit. Cache only."""
    return extract_many(corpus, offline=True, reasoning=DOC_REASONING)


def extract_query_offline(query: str) -> list[tuple[str, str]]:
    """
    Entities for one query, for GraphRetriever._seed. Cache only.

    Returns [] for a query absent from the cache rather than raising, because `_seed` already
    treats an unseeded query as retrieving nothing — the honest outcome the protocol requires —
    and a missing query is a pre-run bookkeeping error that the scored-run gate below catches
    all at once rather than one query into the walk.
    """
    key = cache_key(query, QUERY_REASONING)
    cache = _query_cache()
    return cache.get(key, [])


@functools.lru_cache(maxsize=1)
def _query_cache() -> dict[str, list[tuple[str, str]]]:
    """The cache, read once per process. `_seed` is called per query and re-reading a 61 MB
    file 17,230 times would dominate the run."""
    return load_cache()


def assert_queries_cached(queries: dict[str, str]) -> None:
    """
    Every query must be in the cache before scoring starts.

    Called once, before the walk, so a bookkeeping gap surfaces as a refusal to score rather
    than as an arm that quietly retrieves nothing for the queries nobody extracted — which would
    look exactly like the empty-result finding this experiment is measuring.
    """
    cache = _query_cache()
    missing = [q for q, t in queries.items() if cache_key(t, QUERY_REASONING) not in cache]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(queries)} queries are not in the extraction cache "
            f"(e.g. {missing[:3]}). Run query extraction before scoring. See protocol 004 "
            f"amendment 3."
        )
