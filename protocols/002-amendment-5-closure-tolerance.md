# Amendment 5 to protocol-002 — the BM25 closure tolerance

**Status: frozen on tagging as `protocol-002-amendment-5`.** Written before the artifacts are
regenerated against the new tolerance, and before any number it gates is re-reported. That
ordering is the opposite of amendment 4's and is stated because the difference matters.

## 1. What section 7 registered, and what changes

Section 7 registers the BM25 closure control at "0.10 absolute tolerance" against the published
Anserini figure (Thakur et al. 2021). The tolerance becomes **0.05**. Nothing else about the
control changes: same comparison, same external reference, same halt-on-failure behaviour.

## 2. Why, with the measurement rather than a preference

A review found the gate was 4x to 22x looser than the deltas it was gating, so no realistic
defect could trip it. Rather than pick a smaller number by taste, the defect classes were
measured against the committed scorer on SciFact and Quora:

| class | measured delta in nDCG@10 |
|---|---|
| legitimate tokenisation difference from published | 0.0045 scifact, 0.0212 quora, 0.0272 hotpotqa |
| IDF silently off — a wrong formula | 0.1085 |
| TF saturation off — a wrong formula | 0.1075 |
| length normalisation off | 0.0148 |
| k1 drift, 1.2 to 0.9 or 2.0 | 0.0008 to 0.0054 |
| b drift, 0.75 to 0.4, 0.5 or 1.0 | 0.0006 to 0.0211 |

0.05 sits above the largest legitimate difference with 1.8x headroom and below the wrong-formula
defects with a 2.2x margin. At 0.10 that second margin was 1.08x — the gate barely cleared the
largest defect that exists, which is not a margin but a coincidence.

0.03 was considered and rejected: it clears HotpotQA's 0.0272 by 0.0028, so one corpus revision
would false-fail it on correct code, and a control that false-fails gets loosened until it means
nothing. Per-corpus tolerances were also rejected — a free parameter per corpus is the shape of a
gate tuned until it passes.

## 3. What this does NOT achieve, stated because the opposite is convenient to imply

Tightening this does not let the control catch subtle scoring defects. A k1 or b drift moves
nDCG@10 by up to 0.0211 and the legitimate difference to the published figure is already 0.0272,
so no threshold separates them. The same holds for length normalisation being disabled, at 0.0148.
Both classes are below this control's resolution **by construction**, not because the old number
was too loose.

Those classes are covered elsewhere and this amendment is the record of where: `k1` and `b` are
pinned directly by `tests/test_bm25_constants_pinned.py`, and the factorial's switches by the
equivalence tests in `tests/test_lexical_equivalence.py`. The control's one job is catching a BM25
that is not BM25, against an external reference. It does that, better than it did.

## 4. No decision moves

The measured deltas are 0.0045, 0.0212 and 0.0272 and all three pass at 0.05 as they did at 0.10.
The control passed on all three corpora before this amendment and passes on all three after it.
The artifacts are regenerated so the recorded `tolerance` matches the code, and they gain a
`cannot_detect` field naming the control's blind spots where a reader will actually see them.
No published nDCG@10, no p-value, no Holm decision, and no registered prediction is affected.

## 5. What is not changed

Section 7's other controls — gold-presence, empty-query, embedding shuffle — keep their
thresholds. The anchor remains the published Anserini figure; the in-repo `bm25s` number remains
informational, for the reason the control's docstring already gives.
