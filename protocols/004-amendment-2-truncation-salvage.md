# Amendment 2 to protocol-004 — salvaging truncated extractions

**Status: frozen on tagging as `protocol-004-amendment-2`.** Written after HotpotQA extraction
completed and **before any entity enters a scored graph**. No graph has been built and nothing has
been scored at the time of tagging.

## 1. The failure the protocol did not anticipate

Protocol 004 §9 says "everything else unchanged" and is silent on what happens to a passage the
extractor cannot process. Eight of HotpotQA's 66,581 passages could not be extracted — 0.012%.

The cause is one mechanism, and it is not passage length. GLM-4.7-Flash enters a **degenerate
repetition loop** on label-ambiguous entities, at temperature 0, under strict structured outputs.
On doc `10073694` (653 characters, an obituary of a college coach):

    {"text": "Cleveland State", "label": "ORG"}, {"text": "Cleveland State", "label": "GPE"},
    {"text": "Cleveland State", "label": "ORG"}, {"text": "Cleveland State", "label": "GPE"}, ...

"Cleveland State" is genuinely both an organisation and a place. The model does not commit, and
alternates until it hits the token ceiling. Five retries at `max_tokens=8000` all truncated; a
diagnostic at 16,000 also truncated, at 1,101 emitted entity objects.

## 2. Why the truncated output is nevertheless complete

Measured on all eight, at `max_tokens=16000`:

| doc | chars | entity objects emitted | distinct | first repeat at |
|---|---|---|---|---|
| 10073694 | 653 | 1,101 | 8 | 8 |
| 21305 | 1,014 | 1,065 | 2 | 2 |
| 27756022 | 981 | 1,140 | 5 | 5 |
| 38172774 | 1,413 | 999 | 4 | 4 |
| 43414063 | 674 | 1,229 | 4 | 4 |
| 4790970 | 783 | 1,065 | 2 | 2 |
| 49860 | 1,031 | 1,229 | 1 | 1 |
| 16823614 | 745 | see §4 | — | — |

**In every salvageable case the first repeat occurs at exactly the distinct count.** The model
emits its complete list and then loops on the last ambiguous member. Nothing after the first
repeat is new information.

This matters because the graph deduplicates nodes by string — `extraction_score.score_passage`
already treats entities as sets, not multisets, for the same reason. A repeated mention is one
node. So the distinct set recovered from the truncated prefix is exactly what the graph would
have received from a response that terminated normally.

## 3. The change

When a single passage fails every retry with a truncated response, the entity objects that ARE
complete in that response are recovered by `llm_extractor.salvage_pairs`, deduplicated with order
preserved, and cached with `salvaged: true`.

Recovery is a regex over the partial document rather than a JSON parse, because the input is
truncated by construction and no parser will accept it. Only fully-formed `{"text": ..., "label":
...}` objects match; a half-written final object is discarded rather than guessed at.

Salvage applies **only** after the registered retry path is exhausted. It is not a fallback for a
call that could have succeeded.

## 4. What is NOT salvaged, and the non-determinism behind it

A response containing no complete entity object yields nothing, and that passage is **excluded**
from the LLM arm's graph rather than cached as empty. Caching it empty would make a failed
extraction indistinguishable from a passage that genuinely contains no entities.

Doc `16823614` returned zero complete objects during extraction. A later diagnostic call on the
same passage, same pin, same seed, same temperature, returned a loop on `{"text": "Australia",
"label": "GPE"}` — which would have salvaged to one entity. **The same passage under the same
configuration produced two different responses.** That is expected of an LLM and is exactly why
§6 makes the cache the artifact of record rather than the API.

The consequence is registered here rather than discovered later: salvage operates on the response
actually received in the scored run, not on the best of several probes. Re-probing until a
passage salvages would be selecting the input that gives the wanted output.

## 5. What this costs the comparison

Any excluded passage contributes no nodes to the LLM arm's graph while contributing nodes to
spaCy's, so the two arms are not built from an identical passage set. The count is reported in
the entry alongside the results. At 0.012% it cannot move a retrieval metric, and that claim is
checkable rather than asserted: the excluded ids are committed in
`results/004/extraction-failures.json`.

## 6. What is not changed

The model, provider, quantisation, fallback policy, temperature, seed, batch size, prompt, output
contract, the pilot gate, and every registered prediction in §3. Salvage changes which entities
reach the graph for eight passages; it changes nothing about how the graph is built, walked, or
scored.
