# Amendment 5 to protocol 003 — decomposing prediction A's differential

**Status:** registered **after** the results were seen. This is the amendment's most important
sentence and it is placed first deliberately.
**Amends:** `protocols/003-graph-arm.md` §7.
**Classification:** POST-HOC, exploratory. Not a falsifier. Not added to §7's Holm family.

## Why this is not a pre-registration, and what that costs

Every other analysis in 003 was fixed in the `protocol-003` tag before any arm was scored. This one
was not. It was written after prediction A came back supported, because the supported result raised
a question the protocol had not thought to ask.

An analysis chosen after seeing the numbers has weaker evidential standing than one chosen before,
and no amount of care in the procedure repairs that. It is registered here so that a reader can
see the order events happened in, not to launder the ordering.

The registered result stands unchanged: **prediction A is supported on all six cells.** Nothing
below retracts it.

## The question §7 could not answer

Prediction A is a **difference of advantages**:

```
differential = (graph - bm25) on bridge-absent  -  (graph - bm25) on coverage-2
```

A quantity of that shape moves if *either* arm moves, and §7 required nothing that says which. The
mechanism under test — a graph reaching a document the query never names — predicts that the
**graph** improves where the bridge entity is absent. But BM25 declining on bridge-absent queries
produces the identical differential while meaning something entirely different: that a lexical
matcher does better when the answer's title is spelled out in the question, which is not a fact
about graphs at all.

Both readings satisfy the registered decision rule. The protocol was rigorous about the statistic,
the resample count, the seed, the margin, the correction family and the negative control, and left
the one ambiguity that determines what a passing result means.

## The decomposition

Algebraically the same quantity, split into the two class swings it is the difference of:

```
differential = (graph_absent - graph_cov2) - (bm25_absent - bm25_cov2)
               \_______ graph term _______/   \_______ bm25 term _______/
```

Same stratified percentile bootstrap, same B = 10,000, same seed 20260820 as `_contrast`, so the
registered contrast and the decomposition are one resampling procedure read two ways. Reported for
the same six cells, under both class definitions.

**No new decision rule is introduced.** A CI excluding zero is described as such and nothing is
declared supported or refuted on the basis of it.

## Result

| cell | differential | graph term (CI95) | bm25 term | bm25 share |
|---|---|---|---|---|
| `recall_2\|stripped` | +0.0716 | **+0.0065 [-0.0082, +0.0212]** | +0.0651 | 90.9% |
| `recall_5\|stripped` | +0.1182 | **-0.0155 [-0.0334, +0.0020]** | +0.1338 | 113.1% |
| `ndcg_cut_10\|stripped` | +0.0903 | **-0.0110 [-0.0272, +0.0053]** | +0.1014 | 112.2% |
| `recall_2\|exact` | +0.0985 | **+0.0151 [-0.0017, +0.0315]** | +0.0834 | 84.7% |
| `recall_5\|exact` | +0.1568 | **+0.0035 [-0.0165, +0.0228]** | +0.1533 | 97.8% |
| `ndcg_cut_10\|exact` | +0.1216 | **+0.0029 [-0.0156, +0.0207]** | +0.1187 | 97.6% |

**The graph term's interval includes zero in all six cells.** The bm25 term's excludes zero in all
six. In three cells the share exceeds 100%, meaning the graph term is negative — the graph doing
marginally *worse* where the bridge entity is absent.

The graph arm's absolute R@2 is near-flat across classes — 0.1952 at coverage 0, 0.2272 at
coverage 1, 0.2100 at coverage 2 — while BM25 climbs monotonically, 0.4868 / 0.5543 / 0.5969,
which is what a lexical matcher should do as more gold titles appear verbatim in the query.

## What this does and does not license

**Licensed:** prediction A's differential is, on this corpus with this extractor, overwhelmingly a
property of the BM25 baseline's behaviour across query classes rather than of the graph's.

**Not licensed:** any claim that graph retrieval does not work. §8.3's gate passed convincingly —
gold document pairs share an extracted entity 67.8% of the time against a random-pair null of 6.5%
— so the bridging structure is present in this corpus and findable. What is measured here is that
*this* arm did not convert it. The extractor is spaCy NER where the prior art uses OpenIE, 17.7% of
queries link to no node at all, and experiment 004 is the pre-registered test of exactly that.

**Not licensed:** treating the near-zero graph term as a demonstrated null. These are intervals
that include zero, on one corpus, at one hop depth. Absence of a detectable effect is not a
measured absence of effect.

## Consequence for future protocols

A prediction stated as a differential between two arms cannot attribute the differential. Where the
claim under test is mechanistic — *this arm does better under these conditions* — the registered
prediction should be on the arm's own between-condition difference, with the differential reported
alongside as context. Experiment 004 registers its primary prediction in that shape.
