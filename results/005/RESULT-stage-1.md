# Experiment 005, Stage 1 — typed identity, scored

**Protocol:** `protocols/005-typed-identity.md`, tagged `protocol-005` before the first typed arm
was scored. Two amendments, both tagged before the work they govern except where stated:
`protocol-005-amendment-1` (an estimand error, recorded after the fact with its blast radius
established) and `protocol-005-amendment-2` (a confirmatory test, frozen before the arms it
predicts existed).

**Stage 0** (`RESULT.md`) built the identity registry and measured its coverage. This is the
scored 2×2.

---

## The registered result

| corpus | extractor | string | typed | A: Δ R@2 | 95% CI | A | B: affected − unaffected | 95% CI | B |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | spaCy | 0.2148 | 0.2236 | +0.0088 | [0.0064, 0.0113] | supported | 0.0519 | [0.0391, 0.0646] | supported |
| HotpotQA | GLM | 0.3699 | 0.3756 | +0.0057 | [0.0027, 0.0086] | supported | 0.0328 | [0.0187, 0.0471] | supported |
| 2Wiki | spaCy | 0.2734 | 0.2896 | +0.0161 | [0.0141, 0.0182] | supported | 0.3501 | [0.3182, 0.3820] | supported |
| 2Wiki | GLM | 0.3770 | 0.4196 | +0.0426 | [0.0393, 0.0462] | supported | 0.3458 | [0.3239, 0.3678] | supported |

Holm-adjusted p = 0.0008 on all eight.

**⚠ Read literally this is eight of eight supported. Both readings below cut it down: prediction B
is close to circular (§4), and the effect sizes are small against the baselines (§3). Do not quote
the table without them.**

## Where this sits against the baselines

The graph arm is not competitive, and the relevant comparison is not BM25.

| | HotpotQA R@2 | 2Wiki R@2 |
|---|---|---|
| dense (bge-base-en-v1.5) | **0.6957** | **0.6557** |
| BM25 | 0.5490 | 0.5164 |
| dense (all-MiniLM-L6-v2) | 0.5136 | 0.4766 |
| graph, GLM + typed identity (this entry's best) | 0.3756 | 0.4196 |
| graph, GLM + string identity (004) | 0.3699 | 0.3770 |
| graph, spaCy + string identity (003) | 0.2148 | 0.2734 |

Dense retrieval has beaten BM25 and every graph arm since 003, and had dropped out of the framing
in the last two entries. Reinstated here: after an extractor swap *and* identity resolution, the
graph arm remains **0.28 behind dense** on HotpotQA. §3 of the protocol registered that the graph
would not beat BM25; it does not, and it is further still from the arm that actually wins.

## The near-miss

The Holm loop read each cell's p-value under the key `p`. `paired_bootstrap` returns it under
`p_value`. Every prediction-A cell fell through to a `1.0` default, Holm ran on [1, 1, 1, 1], and
four cells significant at p = 0.0008 were labelled `no_advantage`.

That is falsifier F1 firing spuriously. The analysis was one keystroke from reporting *typed
identity does nothing* — a clean, publishable null that no test and no reviewer would have
contradicted, because the analysis had not run when the reviews were written. It surfaced from
reading the output rather than the exit code.

The default is gone: a cell with no p-value now raises, because a missing p-value must not be able
to present itself as evidence of absence.

## Prediction B is weaker than its p-value suggests

| corpus/extractor | affected: n, non-zero, mean | unaffected: n, **non-zero**, mean |
|---|---|---|
| HotpotQA/spaCy | 1,217, 20.4%, 0.0522 | 6,188, **1.13%**, 0.0003 |
| HotpotQA/GLM | 1,316, 22.4%, 0.0327 | 6,089, **1.87%**, −0.0002 |
| 2Wiki/spaCy | 414, 68.4%, 0.3515 | 9,411, **0.49%**, 0.0014 |
| 2Wiki/GLM | 1,085, 69.6%, 0.3502 | 8,740, **1.44%**, 0.0045 |

**The unaffected stratum is 98–99.5% exactly zero.** That is structural, not empirical: the walk is
a *personalized* PageRank seeded from the query's own entities, so a query whose entities'
reachability did not change is walking a nearly unchanged local subgraph and can barely move.

So B largely restates the definition of its own stratification variable. "The gain concentrates
where identity changed reachability" is close to circular, and it was the sentence this entry was
going to lead with. The 0.5–1.9% residual is real — global specificity and degree reweighting
leaking in, positive in sign — but it is a small finding, not the mechanism confirmation §5 wanted.

## The non-circular test, and what it took to interpret it

Among affected queries only, does the effect scale with the *size* of the reachability change?
Dose = documents entering or leaving a query's entities' reachable sets.

| corpus | extractor | per-query dose min / median / max | ρ (dose vs Δ) | 95% CI |
|---|---|---|---|---|
| HotpotQA | spaCy | 1 / 3 / 13,844 | −0.2461 | [−0.2942, −0.1984] |
| HotpotQA | GLM | 1 / 2 / 6,845 | −0.1337 | [−0.1795, −0.0830] |
| 2Wiki | spaCy | 1 / 1 / 29 | −0.2898 | [−0.3804, −0.1916] |
| 2Wiki | GLM | 1 / 1 / 67 | −0.2005 | [−0.2594, −0.1376] |

Negative on all four, every interval excluding zero. **A larger identity change predicts a worse
outcome** — the opposite of the mechanism.

*(Dose is per query and sums over that query's entities; it is not the blast radius of any single
alias. The two are different quantities and only coincide when one entity dominates.)*

### Explanation 1: hub merges — proposed, tested, and withdrawn

The registry contains a few aliases whose resolution collapses an enormous document set onto one
node. `america → united states` moves 6,669 documents; the merged `united states` node covers
6,846 of 66,581 passages, 10.3% of HotpotQA. Node specificity is 1/document-frequency and
HippoRAG's own ablation shows it is load-bearing, so such a node should be poison.

`protocol-005-amendment-2` registered the test **before running it**: cap any canonical whose
merged document frequency exceeds 1% of the corpus, re-score, and predict that HotpotQA improves
while 2Wiki (whose largest merge is far under its cap) does not. It also registered what a null
would mean: *the hub explanation is then withdrawn, not softened.*

| HotpotQA / GLM | R@2 |
|---|---|
| typed, uncapped | 0.3756 |
| typed, capped at 1% (24 canonicals removed, 347 aliases reverted) | **0.3757** |

**+0.0001. The hub explanation is withdrawn.**

The 2Wiki arm is the control the amendment predicted. Its cap reverts **zero** aliases — 19
canonicals exceed the threshold but none has an alias pointing at it — so the capped and uncapped
arms are identical by construction. They score identically, 0.4196 both, which also confirms the
pipeline is deterministic end to end.

It failed for a reason visible in its own numbers: the cap removed 347 of 183,513 aliases — **0.19%
of the registry**. The dose distribution runs from 1 to 6,845, and almost all of its mass is far
below the cap. Removing the extreme tail was never going to move an effect carried by the middle.

### Explanation 2: difficulty and ceiling — measured, and it is not the answer either

Affected queries are harder than average (HotpotQA 0.3446 baseline R@2 against 0.3699 overall;
2Wiki 0.1991 against 0.3770). And there is a strong ceiling effect: a query already scoring well
can only fall.

| | HotpotQA/GLM | 2Wiki/GLM |
|---|---|---|
| ρ, dose vs **baseline** R@2 | +0.0150 [−0.038, 0.070] | −0.0401 [−0.097, 0.020] |
| ρ, baseline vs Δ | **−0.4110** [−0.448, −0.371] | **−0.4559** [−0.504, −0.406] |
| ρ, dose vs Δ | −0.1337 | −0.2005 |

The ceiling effect is large and real. **But it does not explain the dose-response, because dose is
uncorrelated with baseline** — both intervals include zero. The two effects are independent.

### What survives

The merge-size effect is genuine, it is not the famous hubs, and it is not difficulty. **Bigger
merges are monotonically worse across the ordinary middle of the distribution**, and capping the
spectacular tail does nothing. That is a narrower and better-supported claim than the one this
document carried a draft ago.

Why 2Wiki gained seven times more per affected query than HotpotQA is then mostly **headroom**: its
affected queries start at 0.1991 R@2 against HotpotQA's 0.3446, so there is far more room to
recover. That is a partial account, not a closed one.

## Effect sizes, stated plainly

+0.0057 on HotpotQA/GLM is about 42 queries in 7,405, a 1.53% relative lift. It is
distinguishable from zero because n is large and the sign is consistent, not because it is large.
004's extractor swap moved the same cell by +0.155 — **27× more**.

The four cells are not one finding. 2Wiki/GLM (+0.0426) and HotpotQA/GLM (+0.0057) differ by a
factor of seven and must not be reported under a single "4/4 supported" headline.

## F3, and where the precision report is

Every query in both corpora has **exactly two gold documents** (7,405 and 9,825 queries, all size
2). At k = 2, precision@2 and recall@2 are the same number, so the primary measure already is the
precision report. Secondary measures move the same way — 2Wiki/GLM R@5 0.4479 → 0.5051, R@10
0.4784 → 0.5382, R@100 0.5294 → 0.5908, nDCG@10 0.4573 → 0.5084. **F3 did not fire.**

## Generalisation: everything here is Wikipedia

**The corpora are cut from Wikipedia, the entities are Wikipedia article titles, and the alias
registry is Wikipedia's own redirect graph.** Three legs of one source.

That cuts both ways and neither direction is quantified here.

**It flatters coverage.** 56–61% of pool titles carry a redirect *for free*, because the source
ontology and the target ontology are the same ontology. No enterprise wiki, code repository, legal
corpus or chat archive has an equivalent table. A practitioner there must build one, with an
entity-resolution pipeline carrying its own error rate — which is precisely the cost
`protocols/005-identity.md` §7 named in advance as the one nobody prices.

**It also suppresses the effect.** These passages are already keyed by canonical titles, so the
text contains less alias variation than ordinary prose, where "Obama", "the president" and "Barack
Obama" would all need resolving.

So read this as an **upper bound on what free, curated, perfectly-scoped identity resolution buys a
specificity-weighted graph** — not as evidence about typed identity where curation must be paid
for. 004's entry carried a generalisation caveat of this kind; an earlier draft of this document
dropped it, while being more Wikipedia-coupled than 004 ever was.

## Limitations

**No multiple-testing correction across the programme.** Holm is applied within each experiment's
family and never across experiments. This is the fifth registered analysis run against the same two
corpora, with the freedom each time to design a post-hoc follow-up after seeing the last result.
That is defensible if each experiment answers an independent pre-registered question, which is the
intent — but it is asserted here rather than corrected for.

**The analysis implementation postdates three of the four numbers.** The protocol prose — contrast,
estimator, seed, Holm procedure, decision rules — was tagged at 13:46, before the first typed arm
scored at 13:55, and `git diff protocol-005 HEAD -- protocols/` is empty. `analysis005.py` was
written at 14:39, when three of four R@2 values existed on disk. No tuning is visible and every
threshold traces to Stage 0's table timestamped 12:44. The sequencing is still a fair criticism;
the fix for next time is to commit the analysis code before scoring, not just the prose.

**The `underpowered` rule used the wrong estimand** (`protocol-005-amendment-1`). It compared a CI
width against a two-proportion Bernoulli MDE. The branch was never reached — the largest
Holm-adjusted p across all eight decisions is 0.0008 — so no decision depends on it. That is a
property of this dataset, not a fix.

**Dose-response, hub analysis and the difficulty check are exploratory.** All were designed after
the registered result was known, in response to review. Only the hub cap was pre-registered before
running, in `protocol-005-amendment-2`, and it falsified the explanation it was written to test.

**Two corpora, one alias source.** Nothing here speaks to a corpus without a curated redirect graph.

## The cross-commit question, closed

String arms were scored days earlier at older commits, and `build.py`/`retriever.py` gained the
`link=` and extractor-injection parameters in between. Two reviewers said only a re-run could rule
out a behavioural change. `graph-glm` was re-scored on 2Wiki at HEAD: **R@2 = 0.3770 with a
byte-identical `per_query.jsonl`** (`results/005/cross-commit-reproduction.json`). The published
`summary.json` was restored so the artifact still records its original run.

## Reproducing

    make reproduce-005-affected     # alias-affected query ids
    make reproduce-005-analysis     # both predictions, four cells, Holm per family

Offline, no API key, no network. Scored arms replay from committed `per_query.jsonl`; coverage and
dose work replays from the committed redirect snapshots and 004's committed extraction cache.
