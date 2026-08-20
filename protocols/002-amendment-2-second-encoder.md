# Amendment 2 to protocol-002 — a second encoder, to test the claim I could not support

**Status: frozen on tagging as `protocol-002-amendment-2`.** Written and committed before the
second encoder embeds anything.

## 1. Why this exists

The Experiment 002 draft argued that its finding does not depend on the small encoder it used,
on the grounds that the per-query overlap gradient is a property of the queries rather than of
the model, so a stronger encoder should raise the whole curve without flattening it.

That was an argument where a number belongs, in an entry whose whole complaint is that people
publish arguments where numbers belong. This amendment runs the experiment that settles it.

## 2. The prediction, fixed before anything is embedded

A second, stronger encoder is run through the identical pipeline. Predicted, in order of how
badly each would hurt if wrong:

1. **The gradient stays negative on both corpora.** Dense's advantage over full BM25 still falls
   as query-answer overlap rises, first bin to last.
2. **The gradient does not flatten.** On Quora the first-to-last decline stays within 0.13 to
   0.38, that is the measured 0.255 give or take half. Below 0.13 counts as flattening and the
   draft's argument fails.
3. **The curve rises.** The stronger encoder scores higher overall on both corpora, which is a
   sanity check on the choice of model rather than a claim about the gradient.

**What falsifies the draft's argument:** a first-to-last decline on Quora below 0.13, or a
gradient that inverts on either corpus. Either result gets published and the entry's framing
gets rewritten rather than defended.

Prediction 3 failing without 1 or 2 failing means the second encoder was not actually stronger
here, which invalidates the test rather than the argument, and the entry will say so.

## 3. The second encoder, pinned

- Model: `BAAI/bge-base-en-v1.5`
- Revision: `a5beb1e3e68b9ab7`
- Licence: MIT
- 768 dimensions against MiniLM's 384, 512 max sequence length against 256, and roughly
  11.7 million downloads in the last month, so it is a mainstream choice rather than one picked
  to produce a result.

Measured on this machine before tagging: 252 documents per second against MiniLM's 605.

**Query prefix.** This model family expects retrieval queries to be prefixed with
`Represent this sentence for searching relevant passages: ` while documents are encoded plain.
That asymmetry is part of the model and amendment 1 section 4 already requires the convention
used to be recorded in the manifest. It is applied to queries only, and the manifest records it.

## 4. Corpora

SciFact and Quora. Not HotpotQA: at 252 documents per second it is 5.75 hours to embed, and
Quora is both the corpus where the gradient is cleanest and the one where the draft's claim is
most exposed, since it is the only corpus where dense wins at all.

The omission is stated in the entry rather than left for a reader to notice.

## 5. Everything else unchanged

Same queries, same subsample, same seed, same scoring code, same metrics, same paired bootstrap
and Holm correction, same controls, all inherited from amendment 1. Exact cosine, no approximate
index. The only variable is the encoder.

## 6. Stopping rule

Published whatever comes out, including and especially the case where the gradient flattens and
the draft's central defence turns out to be wrong. That outcome is more interesting than the
confirmation and it is the reason for running this before publishing rather than after.
