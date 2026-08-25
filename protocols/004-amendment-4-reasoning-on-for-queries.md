# Amendment 4 to protocol-004 — reasoning back on for queries, and the transport fixed properly

**Status: frozen on tagging as `protocol-004-amendment-4`.** Supersedes amendment 1 for query
extraction. Written after a scored HotpotQA run was found to be invalid, and before that run is
repeated. The invalid result is reported in §5 rather than discarded.

## 1. What amendment 1 actually did

Amendment 1 added `reasoning: {"enabled": false}` because the payload was arriving in the
response's `reasoning` field with `content` set to `None`, so the parser — which read only
`content` — got nothing. Disabling reasoning made `content` populate and the parse worked.

That was a fix to the model's behaviour for a defect in the parser. The cost was invisible on the
input type that was tested and total on the input type that was not.

| input | reasoning off | reasoning on |
|---|---|---|
| questions, 20 HotpotQA queries | **9 of 20 extract nothing** | **0 of 20 extract nothing** |
| passages, 40 annotated, micro F1 | 0.8255 | 0.8224 |

Same model, same pin, same prompt, same temperature and seed. On passages the setting is worth
0.003 F1 against spaCy's 0.6703 — noise. On questions it is the difference between an extractor
and no extractor.

## 2. The change

**The transport is fixed at its cause.** The parser now reads whichever field carries the
payload: `content` if present, otherwise `reasoning`. That is what amendment 1 should have been.

**Queries are extracted with reasoning ON**, by omitting the field, which is the model's default.

**Documents keep the setting they were bought under**, `{"enabled": false}`. The measurement in §1
is the justification, published rather than asserted: the setting does not move extraction quality
on passages, and re-buying 110,068 extractions to change a figure by 0.003 F1 would spend about
$10 and five hours to make a protocol sentence tidier.

The two settings key differently in the extraction cache, so a query can never be served a
document-side extraction of the same string, or the reverse.

## 3. The asymmetry, stated plainly

Documents and queries are now extracted under different reasoning settings. Amendment 3 exists to
keep the two sides of the extractor consistent, and this is a departure from that principle,
accepted on the evidence in §1 rather than on convenience. A reader who thinks 0.003 F1 is too
much to wave through is entitled to that view, and the number is published so the judgement is
theirs to make.

## 4. What this does not change

The model, provider, quantisation, fallback policy, temperature, seed, batch size, prompt, output
contract, salvage handling, and every registered prediction in §3 of the protocol.

## 5. The invalid run, recorded rather than deleted

A full HotpotQA scoring run completed under amendment 1's setting before the defect was found:
R@2 **0.1974** against 003's spaCy arm at 0.2148, with the arm returning nothing for **51.44%** of
queries against 003's 17.68%.

That number is not a result and is not reported as one. It measures a query extractor that
returned nothing for roughly half its inputs. It is recorded here because it had exactly the shape
of registered failure mode 1 — "the advantage flattens, meaning 003's differential was an
extraction artefact" — and a plausible mechanism was available to explain it ("higher precision
means fewer co-occurrence edges"). It would have been publishable and wrong.

What caught it was not the interpretation but the empty rate: a 3x jump is not what a change in
extraction quality looks like. The gate registered for this class of error,
`assert_queries_cached`, did not fire, because the queries **were** cached — cached with empty
entity lists. The gate checked for missing entries and not for empty ones.
