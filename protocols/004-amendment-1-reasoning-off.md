# Amendment 1 to protocol-004 — reasoning disabled on the extractor

**Status: frozen on tagging as `protocol-004-amendment-1`.** Written after a pre-pilot smoke test
and **before any passage is scored**. The pilot in §8 has not been run at the time of tagging.

## 1. What happened

Protocol 004 §4 pinned the model, provider, quantisation, temperature and seed, and said nothing
about reasoning. A four-call smoke test against the pinned endpoint — not the pilot, not scored,
total cost about $0.0001 — showed the pin is honoured (`provider: DeepInfra`,
`model: z-ai/glm-4.7-flash`) and that the extraction is correct, but that it does not arrive where
the code reads it.

GLM-4.7-Flash is a reasoning model. With reasoning left at its default, DeepInfra returns the
structured payload in the `reasoning` field and sets `content` to `None`. Parsing `content` yields
nothing. The entities were right; the transport was wrong.

## 2. The change

`reasoning: {"enabled": false}` is added to every request. Measured on the same passage:

| setting | `content` | `reasoning` | completion tokens | reasoning tokens |
|---|---|---|---|---|
| default | None | set | 46 | 30 |
| `{"enabled": false}` | **set** | None | 46 | **0** |
| `{"exclude": true}` | None | None | 46 | 30 |

`{"exclude": true}` is explicitly **not** used. It returns `content` None *and* `reasoning` None,
losing the payload silently rather than loudly, which is the worse of the two failure modes.

## 3. Why this is a pin change and not a bug fix

It would be convenient to call this an implementation detail. It is not: reasoning changes what
the model does, so a run with it on and a run with it off are not the same experiment. §4
registered the configuration and this adds to it, so it is an amendment.

It also changes the cost basis registered in §10. Thirty of forty-six completion tokens on a
one-entity passage were reasoning — most of the output bill for a task that needs none. The
full-run estimate in the entry will be re-derived from the pilot's measured usage under this
setting, per §8.

## 4. Cache invalidation, deliberate

The reasoning setting is now part of the extraction cache key alongside the model, provider,
quantisation and prompt hash. Any extraction produced under a different setting therefore misses
rather than being silently reused. Nothing is invalidated in practice — the cache is empty, since
no passage has been extracted for score.

## 5. What is not changed

The model, provider, quantisation, fallback policy, temperature, seed, batch size, prompt, output
contract, the pilot gate and its 0.7203 bar, and every registered prediction. §5's entity contract
and the deviation from NB-20's OpenIE wording stand as tagged.
