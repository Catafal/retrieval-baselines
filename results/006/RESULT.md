# Experiment 006 — result

Protocol: `protocols/006-graph-memory-answering.md`, tagged `protocol-006` before the first scored call. Every figure below is generated from the committed artifacts.

## The headline

**P1 (primary).** On haiku, `graph-facts` minus `grep` = **−0.2700** EM, 95% CI [−0.3700, −0.1700], p = 0.00019998000199980003, MDE 0.1657 at the realised discordance of 0.35. Decision: **no_advantage**.

**P3 (the prior art's actual thesis, registered underpowered).** The haiku-minus-opus interaction = −0.2000, CI [−0.3200, −0.0800], MDE 0.241. Decision: **no_advantage**.

### The shortfall is coverage, not conversion

Answer-presence is the rate at which the gold answer string is present in an arm's injection: a ceiling on its EM under a copy-only model. EM divided by that ceiling is a crude conversion efficiency — how well an arm used what it actually delivered.

| arm | answer-presence ceiling | EM/ceiling haiku | sonnet | opus |
|---|---|---|---|---|
| `oracle` | 0.98 | 0.582 | 0.622 | 0.633 |
| `dense` | 0.80 | 0.637 | 0.688 | 0.762 |
| `bm25` | 0.55 | 0.727 | 0.891 | 1.036 |
| `graph-facts` | 0.38 | 0.711 | 1.237 | 1.474 |

**This is the caveat the headline needs.** The graph's injection contains the literal gold string for 38% of questions against 55% for BM25, 80% for dense and 98% for the oracle. Within that coverage the graph converts evidence into correct answers about as efficiently as BM25 and more efficiently than dense. **The P1 shortfall is concentrated in what the extractor retained, not in how the answering model used what it kept.** What this experiment falsifies is a graph built cheaply by the weakest tier, not graph retrieval as such.

Efficiency above 1 means an arm answered beyond its own injection — parametric memory filling gaps in a sparse context. It appears for BM25 at opus too, so it is a low-ceiling strong-model effect rather than anything specific to graph structure. Exploratory, unregistered, and carrying no decision.

## Every arm, every tier

| arm | tier | n | EM | EM lenient | EM strict | F1 | abstain | max-turns | turns | ctx tok | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `closed-book` | haiku | 100 | **0.1400** | 0.1700 | 0.1400 | 0.2143 | 0.3000 | 0.0000 | 2.61 | 30263 | 1.89 |
| `closed-book` | sonnet | 100 | **0.3700** | 0.4400 | 0.3700 | 0.5221 | 0.0300 | 0.0000 | 2.89 | 47546 | 4.82 |
| `closed-book` | opus | 100 | **0.4300** | 0.4800 | 0.4300 | 0.5718 | 0.0900 | 0.0000 | 2.34 | 25419 | 7.01 |
| `grep` | haiku | 100 | **0.5400** | 0.5700 | 0.5400 | 0.6853 | 0.0700 | 0.0500 | 7.54 | 92870 | 2.64 |
| `grep` | sonnet | 100 | **0.5400** | 0.6000 | 0.5400 | 0.7263 | 0.0000 | 0.0000 | 5.83 | 80517 | 4.77 |
| `grep` | opus | 100 | **0.6300** | 0.6800 | 0.6300 | 0.7803 | 0.0000 | 0.0000 | 4.62 | 49884 | 8.55 |
| `bm25` | haiku | 100 | **0.4000** | 0.4400 | 0.4000 | 0.5102 | 0.2800 | 0.0000 | 1.13 | 12987 | 1.14 |
| `bm25` | sonnet | 100 | **0.4900** | 0.5300 | 0.4900 | 0.6013 | 0.0100 | 0.0000 | 1.29 | 19807 | 2.42 |
| `bm25` | opus | 100 | **0.5700** | 0.6300 | 0.5700 | 0.7316 | 0.0200 | 0.0000 | 1.13 | 13552 | 4.48 |
| `dense` | haiku | 100 | **0.5100** | 0.5600 | 0.5100 | 0.6608 | 0.1100 | 0.0000 | 1.11 | 12719 | 1.01 |
| `dense` | sonnet | 100 | **0.5500** | 0.5900 | 0.5500 | 0.6961 | 0.0000 | 0.0000 | 1.14 | 17226 | 2.03 |
| `dense` | opus | 100 | **0.6100** | 0.6500 | 0.6100 | 0.7823 | 0.0100 | 0.0000 | 1.11 | 13300 | 4.26 |
| `graph-facts` | haiku | 100 | **0.2700** | 0.3100 | 0.2700 | 0.3553 | 0.2500 | 0.0000 | 2.08 | 24525 | 1.67 |
| `graph-facts` | sonnet | 100 | **0.4700** | 0.5100 | 0.4700 | 0.5916 | 0.0400 | 0.0000 | 1.74 | 27003 | 3.01 |
| `graph-facts` | opus | 100 | **0.5600** | 0.6100 | 0.5600 | 0.7141 | 0.0300 | 0.0000 | 1.57 | 17919 | 5.70 |
| `oracle` | haiku | 100 | **0.5700** | 0.6200 | 0.5700 | 0.7612 | 0.0300 | 0.0000 | 1.0 | 11253 | 0.95 |
| `oracle` | sonnet | 100 | **0.6100** | 0.6600 | 0.6100 | 0.7584 | 0.0000 | 0.0000 | 1.04 | 15409 | 1.76 |
| `oracle` | opus | 100 | **0.6200** | 0.6600 | 0.6200 | 0.8055 | 0.0000 | 0.0000 | 1.0 | 11896 | 3.94 |

Harness overhead, subtracted from the P4 measure: 34409 context tokens per call, measured on the zero-context arm. It is the CLI's own system prompt and tool schemas, not corpus.

## The registered predictions

### P2 — graph-facts vs grep, per tier (Holm family of 3)

| tier | n | graph | grep | Δ | 95% CI | p | p Holm | MDE | discord | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku | 100 | 0.2700 | 0.5400 | −0.2700 | [−0.3700, −0.1700] | 0.00019998000199980003 | 0.0005999400059994001 | 0.1657 | 0.35 | **no_advantage** |
| sonnet | 100 | 0.4700 | 0.5400 | −0.0700 | [−0.1700, +0.0300] | 0.19198080191980801 | 0.19198080191980801 | 0.1401 | 0.25 | **no_advantage** |
| opus | 100 | 0.5600 | 0.6300 | −0.0700 | [−0.1300, −0.0100] | 0.034996500349965 | 0.06999300069993 | 0.0929 | 0.11 | **no_advantage** |

### P5 — the adversarial one: graph-facts vs dense

Registered because the author did not want it to be true.

| tier | n | graph | dense | Δ | 95% CI | p | p Holm | MDE | discord | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku | 100 | 0.2700 | 0.5100 | −0.2400 | [−0.3400, −0.1500] | 0.00019998000199980003 | 0.0005999400059994001 | — | 0.3 | **no_advantage** |
| sonnet | 100 | 0.4700 | 0.5500 | −0.0800 | [−0.1800, +0.0200] | 0.12318768123187682 | 0.23517648235176483 | — | 0.24 | **no_advantage** |
| opus | 100 | 0.5600 | 0.6100 | −0.0500 | [−0.1100, +0.0100] | 0.11758824117588242 | 0.23517648235176483 | — | 0.09 | **no_advantage** |

### P4 — context tokens, grep minus graph-facts

A positive difference means the graph arm read less. The prediction is conjunctive: fewer tokens AT EQUAL OR BETTER EM.

| tier | n | grep tok | graph tok | Δ | 95% CI | p | p Holm | MDE | discord | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku | 100 | 58605.5082 | 2088.5375 | +56516.9707 | [+42991.4797, +71261.3670] | 0.00019998000199980003 | 0.0005999400059994001 | — | 0.95 | **no_advantage** |
| sonnet | 100 | 46486.1687 | 8183.4298 | +38302.7388 | [+28510.3825, +48853.4588] | 0.00019998000199980003 | 0.0005999400059994001 | — | 0.92 | **supported** |
| opus | 100 | 17655.6998 | 1125.9335 | +16529.7663 | [+12347.1332, +21264.7727] | 0.00019998000199980003 | 0.0005999400059994001 | — | 0.8 | **no_advantage** |

## The falsifier checks

### F2 — is the graph moving hops, or just supplying a short clean context?

`oracle` gives the same information without the graph's structure. If it matches or beats `graph-facts`, the mechanism claim is dead however the EM table looks.

| tier | n | oracle | graph | Δ | 95% CI | p | p Holm | MDE | discord | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku | 100 | 0.5700 | 0.2700 | +0.3000 | [+0.1900, +0.4100] | 0.00019998000199980003 | — | — | 0.38 | **—** |
| sonnet | 100 | 0.6100 | 0.4700 | +0.1400 | [+0.0500, +0.2300] | 0.0023997600239976003 | — | — | 0.24 | **—** |
| opus | 100 | 0.6200 | 0.5600 | +0.0600 | [+0.0000, +0.1200] | 0.0653934606539346 | — | — | 0.1 | **—** |

### F4 — is the grep arm a strawman?

`grep` against the `oracle` ceiling. If grep on opus cannot approach oracle on opus, the control is broken and no arm comparison stands.

| tier | n | grep | oracle | Δ | 95% CI | p | p Holm | MDE | discord | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku | 100 | 0.5400 | 0.5700 | −0.0300 | [−0.1100, +0.0500] | 0.5321467853214679 | — | — | 0.15 | **—** |
| sonnet | 100 | 0.5400 | 0.6100 | −0.0700 | [−0.1400, +0.0000] | 0.05599440055994401 | — | — | 0.13 | **—** |
| opus | 100 | 0.6300 | 0.6200 | +0.0100 | [−0.0500, +0.0800] | 0.8853114688531147 | — | — | 0.11 | **—** |

### F6 — is `graph-facts` a lexical retrieval arm wearing arrow notation?

Answer presence is the rate at which the gold answer string appears in the injection: a necessary condition for answering by copying, so a ceiling on the arm's EM under a copy-only model. Computed with no model calls.

| measure | value |
|---|---|
| seed rate (question matched ≥1 entity) | 0.94 |
| questions with empty recall | 6 |
| `a0` seed neighbourhood only | 0.11 |
| `a3` the shipped walk | **0.39** |
| `a3_shuf` edge-permuted placebo | 0.18 |
| `a3` triples only, no entity prose | 0.22 |
| walk over seed match | 0.28 |
| walk over placebo | 0.21 |

**Verdict: walk contributes beyond seed match and beyond placebo.**

### Does the walk actually walk?

Injected-triple depth histogram at the registered configuration (`hops=3, top_k=8`, no depth reservation): `{'0': 663, '1': 71, '2': 9}`. 10.8% of injected facts are depth ≥ 1 and 1.2% are depth ≥ 2.

## Extraction yield, which gates interpretation

| | |
|---|---|
| documents attempted | 990 |
| completed | 990 |
| parsed into the graph | 990 |
| yield | **100.0%** |
| questions with 2 gold docs in the graph | 97 |
| with 1 | 3 |
| with 0 | 0 |

Interpretable at the registered 90% threshold: **True**.

## What the graph cost to build

A method that wins at query time by spending more at build time has not won until both are on the page.

| | |
|---|---|
| extractor | haiku (the weakest tier under test) |
| documents | 990 |
| cost | $28.45 |
| tokens in / out | 6,517,329 / 4,694,191 |
| wall time | 69 min |
| entities / edges / aliases | 7,209 / 8,305 / 1,837 |
| amortised per question | $0.284 |

## The corpus

100 HotpotQA bridge-hard questions, seed 20260820, over a 990-document pool of 577,517 characters. Questions sha256 `47397643a6ca3e64`, pool sha256 `953c12ea4f8bd49c`.

Strata sizes: all = 100, resistant-haiku = 86, resistant-sonnet = 63, resistant-opus = 57.
