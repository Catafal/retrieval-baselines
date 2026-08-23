# Protocol 004 — does a better extractor move the graph result

**Status: frozen on tagging as `protocol-004`.** Written and committed before the first passage is
sent to any model and before anything is scored. No extraction call has been made at the time of
tagging, paid or free.

## 1. Why this exists, and what 003 already committed to

003 shipped a deterministic spaCy extractor so the run stayed reproducible on a laptop and so the
result was not dominated by an LLM call. The cost of that choice was named in 003 §11 as a
limitation and filed as NB-20 before 003 ran. The published entry commits to this experiment
twice, in the body and in the closing.

The question is the one 003 could not settle: does the graph arm's failure belong to **this arm**
or to **graphs built this way**? An extractor swap is the test that separates them.

## 2. The ceiling, registered before running, because it bounds every outcome below

003 measured an oracle: corpus titles as entities, a perfect linker, no recogniser and no
whitelist. That is an upper bound on any extractor this graph design can use.

| arm | R@2 | retrieves nothing |
|---|---|---|
| BM25 | 0.5490 | — |
| graph, oracle extractor | 0.3344 | 18.8% |
| graph, spaCy (003's arm) | 0.2148 | 17.7% |

**A real extractor cannot beat a perfect one.** This experiment's R@2 is therefore bounded to
approximately [0.2148, 0.3344], and it cannot overturn BM25. That is registered here so the entry
cannot later be read as though a surprise was available and failed to arrive, and so that "the
graph still loses" is reported as the expected outcome rather than as news.

What the experiment CAN settle, and the only reason it is worth running: whether the differential
003 found is a property of the mechanism or an artefact of a weak recogniser.

## 3. The registered expectation, and the falsifiers

**Registered expectation (not a falsifier).** The graph arm does not beat full BM25 overall, on
either corpus. See §2.

**Prediction A — the curve rises, the concentration survives.** A better extractor raises the
graph arm's whole curve and leaves its concentration in the bridge-absent class intact.

- Statistic: the same class differential 003 registered — graph − BM25 per coverage class,
  difference of class means, percentile bootstrap over queries, B = 10,000, seed 20260820.
- Decision rule: the same as 003 §7. CI95 excluding zero **and** a point estimate of at least
  +0.02. Holm across the family {R@2, R@5, nDCG@10} × {primary, sensitivity} × {HotpotQA, 2Wiki}.
- **Decomposed, not just measured.** 003's prediction A passed on all six cells and then
  decomposed to the baseline: the graph term's CI included zero in all six, meaning the
  differential was carried by BM25's behaviour rather than the graph's. 004 reports the same
  decomposition, and a differential that again decomposes to the baseline is reported as such.
  Passing A without a non-zero graph term is **not** a confirmation of the mechanism.

**Failure mode 1, registered.** The advantage flattens rather than rises. Then 003's differential
was an extraction artefact and 003's finding is weakened. The entry says so, in those words, and
the 003 entry gets a correction linking here.

**Failure mode 2, registered.** The graph's HotpotQA result crosses full BM25. Given §2's ceiling
this is close to impossible, and it is registered anyway because a registered prediction that
excludes its own surprise is not a prediction. If it happens, 003's headline is
extractor-dependent the way 002's corpus ranking turned out to be. That is a result, not a
failure.

**Failure mode 3, registered.** 2Wiki's predictions C and D were refuted in 003 — the arm did its
best work on questions needing no traversal. If a better extractor reverses that, 003's mechanism
claim changes and the entry reports the reversal as the headline.

## 4. The extractor, pinned

| | |
|---|---|
| model | `z-ai/glm-4.7-flash` |
| gateway | OpenRouter |
| provider | DeepInfra, pinned |
| quantisation | **bf16**, pinned |
| fallbacks | disabled |
| temperature | 0 |
| seed | 20260820 |
| output | structured outputs, schema in §5 |
| batch | 10 passages per call |

Sent as `provider: {"only": ["DeepInfra"], "quantizations": ["bf16"], "allow_fallbacks": false}`.

**Why this model and not a flagship.** OpenRouter routes one model id across backends at
different numerical precisions. `deepseek-v4-flash` spans 17 backends from fp4 to fp8;
`kimi-k2.5` spans 10 including int4, and Moonshot's own endpoint serves int4; GLM-4.7's own Z.AI
endpoint serves fp4. bf16 is the only tier where the weights served are the weights named.

§2 is what makes that trade correct rather than merely cheap. The ceiling is already known, so
extractor capability only matters up to 0.3344, and no model can buy past it. Reproducibility is
the scarce property here, not capability. A reader objecting that a larger model might differ is
answered by the oracle row rather than by a bigger bill.

**A pin is best-effort and is not the reproducibility guarantee.** A provider can change its
serving stack under a stable name. §6 is the guarantee.

## 5. The prompt and the output contract, frozen verbatim

004 extracts **entities**, not OpenIE triples.

**This is a deliberate deviation from NB-20's wording**, which says "LLM OpenIE extraction".
Triples would change the graph's node and edge semantics, so the arm would differ from 003 in the
extractor *and* the structure built on top of it, and no result could be attributed to either.
002's amendment 2 claimed "the only variable is the encoder" while three things differed; that
error is not repeated here. Entity extraction keeps the builder, the linking rule, the walk, the
query classes, the analysis and the controls byte-identical to 003. OpenIE is a different
experiment and is not this one.

Output contract, identical to `extractor.extract`: a list of `(surface_form, label)` per passage,
**unfiltered**, with the whitelist applied downstream by `entity_types.filter_entities` to both
the reference set and the predictions, in one place, exactly as in 003.

Labels are the OntoNotes set spaCy emits, so both extractors are scored on the same alphabet:
`PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE`, plus
`DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL` which the whitelist excludes.

Prompt, frozen:

> Extract every named entity from each passage. For each entity return its exact surface form as
> it appears in the passage, and one label from this set: PERSON, NORP, FAC, ORG, GPE, LOC,
> PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL,
> CARDINAL. Use the OntoNotes 5 definitions of these labels. Copy surface forms character for
> character from the passage; do not normalise, expand, or correct them. Return every occurrence
> of a distinct entity once. If a passage contains no named entities, return an empty list.

No few-shot examples, no chain of thought. Both would be free parameters tuned against the
outcome, and neither is registered.

## 6. Reproducibility — the cache is the artifact of record

An LLM is not deterministic even at temperature 0 with a seed. `make reproduce` therefore does
**not** mean re-calling the API.

Every extraction is written to a committed cache keyed by `sha256(passage_text) + model id +
provider + quantisation + sha256(prompt)`. The scored run reads the cache. A cache miss during a
scored run is a hard failure, not a silent API call. Changing the prompt or the pin changes the
key, so a changed prompt cannot silently reuse old extractions.

This mirrors 002's embedding cache. Without it this experiment is not reproducible and does not
ship.

## 7. Extraction quality, measured directly

Both extractors are scored on the extraction step itself with 003's existing
`rb.experiments.graph.extraction_score`, against the same gold and the same normalisation, so the
comparison does not run only through downstream retrieval. Reported: micro precision, recall, F1
with bootstrap intervals, and bridge reachability.

This is what makes "a better extractor" a measured claim rather than an assumption. If the LLM is
not better on extraction, the retrieval comparison answers a different question than the one
asked and the entry says so.

## 8. The pilot gate, and the stopping rule it enforces

**200 passages, sampled with seed 20260820, scored under §7 before the full run is purchased.**

- If the LLM's extraction F1 does not exceed spaCy's by at least 5 points absolute on the pilot,
  the full run is **not** purchased. The entry reports the pilot, states that a stronger extractor
  did not clear the bar on the extraction step, and 003's extractor gap stays open with that
  measurement attached.
- The pilot's per-passage token usage is measured and the full-run cost re-estimated from it
  before spending.

The gate is registered here so that not spending is a pre-committed outcome rather than a decision
made after seeing a disappointing number.

## 9. Everything else unchanged

Same pools (HotpotQA 66,581 passages / 7,405 queries; 2Wiki 43,487 / 9,825), same seeds, same
graph construction, same personalized PageRank, same coverage classes, same metrics, same Holm
family structure, same controls including the §8.3 bridge-reachability gate and the pool
construction control. The extractor is the only variable, and §5 exists to keep that true.

## 10. Cost, reported in the entry

Currency and wall-clock both, per NB-20, because the comparison against a laptop-runnable
extractor is part of the finding. spaCy's cost is reported the same way.

## 11. Stopping rule

Published whatever comes out, including and especially a null, and including the pilot-gate
outcome in §8.

## 12. The null paragraph, written before the run

Required by §11 and written here, before any passage is sent, so a null is reported rather than
spun. If prediction A resolves but again decomposes to the baseline, the entry publishes the
following, adjusted only for the actual numbers:

> **A better extractor did not change the answer.** GLM-4.7-Flash extracted entities at [P/R/F1]
> against spaCy's [P/R/F1], a clear improvement on the extraction step itself. The graph arm's
> R@2 moved from 0.2148 to [X], against BM25's 0.5490 and against the oracle ceiling of 0.3344.
> The class differential was [D] with a 95% interval of [L, U], and decomposing it put the graph
> term's interval at [GL, GU]. [If it includes zero:] As in 003, the differential is carried by
> the baseline rather than by the graph, now with an extractor that is measurably better. That
> moves the finding from "this arm cannot traverse" toward "graphs built this way cannot
> traverse", which is what this experiment was for.
