# Amendment 3 to protocol-004 — the swap covers queries, not only documents

**Status: frozen on tagging as `protocol-004-amendment-3`.** Written before any graph is built and
before anything is scored. Document extraction is still running at the time of tagging; no query
has been extracted by any model.

## 1. The ambiguity, and why it is not a detail

§5 registers the output contract as "identical to `extractor.extract`". That function is used in
**two** places, and the protocol did not say whether both were in scope:

- `GraphRetriever.fit` extracts entities from every passage to build the graph's nodes.
- `GraphRetriever._seed` extracts entities from each query to place the walk's restart mass.

Swapping only the document side would leave GLM-extracted nodes being matched against
spaCy-extracted query entities. `_seed` links them **by exact normalised string**, so any
difference in span boundaries between the two extractors fails to link.

That failure would not be neutral. It would lower the arm's seed rate and raise its
"retrieves nothing" rate, and it would do so for a reason that has nothing to do with whether a
graph can traverse. 003's finding is about reach; an arm crippled by extractor mismatch would
look like a confirmation of it while actually measuring a mistake in this experiment's
construction.

## 2. The change

The extractor swap applies to **both** call sites. Query entities for the LLM arm are extracted
by GLM-4.7-Flash under the same pin, prompt, contract and cache as the document side.

The spaCy arm is untouched: it uses spaCy on both sides, as 003 did. Each arm is therefore
internally consistent, which is the property that makes the comparison about extraction quality
rather than about extractor pairing.

## 3. Cost

17,230 queries — 7,405 HotpotQA and 9,825 2Wiki — at roughly 20 tokens each, batched ten per
call: about 1,723 calls and **$0.25**. Reported with the rest of the extraction cost per §10.

## 4. What this does not change

Not the graph construction, the walk, the damping, the specificity weighting, the coverage
classes, the metrics, the Holm family, the controls, or any registered prediction. It settles
which component the word "extractor" refers to, in the direction that keeps the arms internally
consistent.

## 5. Noted against a prior mistake

002's amendment 2 claimed "the only variable is the encoder" while three things differed, and
that claim had to be corrected in the published entry. This amendment exists because the same
class of error was available here: swapping half a component and describing it as swapping the
component. It was found while wiring the scoring pipeline, before any graph was built, rather
than by a reader afterwards.
