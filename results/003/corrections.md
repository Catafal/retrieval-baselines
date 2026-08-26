# Experiment 003 — corrections

Every figure in 003 that changed after it was first computed, what changed it, and which review
pass should have caught it and did not.

This file exists because the alternative is a quiet edit. Entry 001 was retracted for publishing
numbers with no reproducible command behind them, and the rule written in response was that a
measured artifact must be reproducible by a named command. A corollary the retraction did not
state, and which these three audits forced: when a published number moves, the move itself has to
be reproducible too — which value, from what to what, and why.

**No correction listed here changes a registered conclusion.** All six prediction-A cells remain
`supported`, all six prediction-B cells remain `no_advantage`, and the §8.3 gate remains `passed`.
That is stated first so it cannot be mistaken for a defence offered late.

---

## The audit sequence, and what each pass missed

| Pass | Fixed | Missed — found by the next pass |
|---|---|---|
| NB-24 | 4 defects (whitelist asymmetry, three contract gaps) | the 6 below |
| NB-25 | 6 defects | the 4 below |
| NB-26 | 4 defects | see "what NB-26 itself got wrong" |
| amendment 6 | — | 2 more (C8, C9), found by checking whether the repo's prose had been updated |

The pattern is the finding. Each pass found what the previous pass's own fixes walked past, and in
two cases a pass was corrected by an adversarial reviewer rather than by its own author.

---

## Corrections from NB-25

### C1 — every contrast published `p_value: 0.0`

All 12 of them, and Holm propagated the zero into `p_holm`. The percentile bootstrap computed
`2 * min(#{x<=0}, #{x>=0}) / B`, which returns exactly zero when no resample crosses the null. A
p-value of exactly zero is a claim no finite resampling procedure can make.

Replaced with the add-one estimator (Davison & Hinkley 1997; Phipson & Smyth 2010), which cannot
return zero by construction.

| | before | after |
|---|---|---|
| `p_value`, all 12 contrasts | `0.0` | `0.0002` |
| `p_holm`, all 12 contrasts | `0.0` | `0.0024` |

**Adjudicates nothing.** §7 lists Holm under "how it is adjudicated", but no code path lets
`p_holm` gate a decision; `supported` reads the CI lower bound and the margin. Both are reported
for transparency and decide nothing.

### C2 — `headroom-control.json` was stale

The committed file had no producer anywhere in the source tree and carried the pre-PPR-fix
differential. Rebuilt from the same per-query data the analysis already assembles.

| | before (stale) | after |
|---|---|---|
| raw differential | 0.0534 | 0.0716 |
| normalised differential | 0.2435 | 0.2864 |

### C3 — three artifacts had no producing code

`extraction-diagnostic.json`, `gate-and-seed.json`'s `graph` block, and `headroom-control.json`
were all written by code that no longer existed. Producers were restored and verified to reproduce
the published values exactly (66,581 / 65,987 / 291,837 / 9.25) **before** being adopted. Had any
failed to reproduce, the block would have been removed from the artifact rather than redefined.

No figure moved. The defect was that nothing could have re-derived them.

### C4 — query seeds double-counted on a normalisation collision

Affected 3 of 7,405 queries. **No figure moved** — none of the three changed rank. The rule was
wrong; the numbers were not.

---

## Corrections from NB-26

### C5 — an incorrect percentile index, at four sites, one of them a gate

`rb/stats.py` documents that the 97.5th-percentile index requires a `-1`, and has a regression
test named for it. Four sites in the graph arm omitted it.

This is **not** a house-convention inconsistency, and the entry must not describe it as one. With
`int(f*N)` indexing the `-1` is what makes "exactly 2.5% of draws beyond each cut" true on both
tails. Written by hand, the upper bound cut one fewer draw than the lower bound did.

| artifact | field | before | after |
|---|---|---|---|
| `analysis.json` | `B\|recall_2\|stripped` hi | -0.369 | **-0.3692** |
| `analysis.json` | `B\|ndcg_cut_10\|stripped` hi | -0.4549 | **-0.455** |
| `extraction-diagnostic.json` | `precision_ci` hi | 0.725 | **0.7249** |

The other 11 CI bounds round identically. 100 other contrast fields are unchanged, as are
`gate-and-seed.json` and `headroom-control.json`.

**The gate's input moved and its verdict did not.** `bridge_reachability`'s `null_hi` is the §8.3
gate's decision boundary — `passed = observed > null_hi` — and the hand-written index left 4.9% of
draws above a threshold claiming 5%. At the committed run `draws[949] == draws[950] == 0.0652`, so
the threshold is unchanged at four decimals and `passed` stays `True` either way: observed 0.6778
beats the null by roughly tenfold. This is the one correction that could have changed a
pre-registered verdict, and it did not. It is stated explicitly rather than folded into the table
above, because "a gate's threshold was computed wrongly" is the sort of thing a reader is entitled
to see addressed directly.

### C6 — the §9 pool control contained a check that could not fail

`gold_titles_matched` was compared against `gold_queries` with both passed the same expression,
`len(qrels)`. `unresolved` and `collisions` were literals with no producer anywhere in `pool.py`.
Three of seven published fields were assertions inside a block reporting `"passed": true`.

Registered as `protocols/003-amendment-4-pool-control.md`, because it changes what a field of a
tagged control means. **No value moved** — every field measured the same number it had asserted.

The amendment also states the control's honest scope: gold reachability holds *by construction*,
since HotpotQA's distractor setting includes each question's supporting passages. The control
retains power over title-to-BEIR-id resolution, not over reachability, and must not be read as
evidence for the latter.

### C7 — two prose defects

`extraction_score.py`'s module docstring named `graph_connectivity` as the gate. It was not; it had
no production caller and cited a function that does not exist. Deleted. `seed_match_rate` built its
node set without the guard `build.build` uses, so `""` was a node in the diagnostic and never in
the graph — measured impact zero queries, published 0.8232 correct.

---

## Correction from the second-corpus run (amendment 6)

### C8 — `arms-summary.json` had no producer, and published a retracted number

Found while checking whether the repository's prose had been updated for the second corpus. It had
not, and the check turned up something worse.

`results/003/arms-summary.json` was published with **no producing code anywhere in the repository**
— `grep` for its name across `src/`, `scripts/`, the `Makefile` and every protocol returns nothing
— and it stated `graph.recall_2 = **0.2132**`. That is the figure from the DEFECTIVE walk that
`003-amendment-3` retracted. The live corrected figure is **0.2148**, and 0.2132 survives only in
`results/003/pool/graph-defective/`, which exists precisely so a reader can see what was withdrawn.

So a published artifact restated a retracted number, with no command able to regenerate or refute
it. This is the fifteenth defect and the fourth instance of the same class. All three prior audits
missed it, each for the same reason: they searched the graph arm's source tree, and this file is
produced by nothing that lives there. Searching for *code without artifacts* does not find
*artifacts without code*, and searching the module tree does not find a file that no module writes.

Fixed by `src/rb/experiments/graph/arms_summary.py` and `make reproduce-003-arms-summary`, which
derives the table from each arm's committed `summary.json` and now covers both corpora. The
defective run is named in a `superseded_runs` block rather than sitting in the table as a fifth arm.

| field | before | after |
|---|---|---|
| `arms.graph.recall_2` | 0.2132 *(retracted)* | **0.2148** |
| `arms.graph.recall_5` | 0.2861 *(retracted)* | **0.3132** |
| `arms.graph.ndcg_cut_10` | 0.2955 *(retracted)* | **0.3161** |

### C9 — a structural exclusion that the amendment failed to carry forward

Not a wrong number; a wrong process, and it belongs here for that reason.

`NB-5`, the entry specification written before any of this ran, excluded 2WikiMultiHopQA from
experiment 003 **on structural grounds**, and measured why:

> 2Wiki ... has **no coverage-0 class at all** (measured: 507 coverage-1, 493 coverage-2). It
> cannot test "the bridge entity is absent from the query"; it can only test "does the graph work
> where topology favours it." The reason 2Wiki is excluded is structural, not budgetary.

Amendment 6 added 2Wiki without carrying that exclusion forward, and the run then measured a
coverage distribution of **1 / 6,603 / 3,221** — confirming the spec exactly. It was first reported
as a limitation found after scoring and *not anticipated*. That description was wrong: it was
anticipated, in this experiment's own specification, and not re-read before the corpus was added.

**What survives.** Prediction C contrasts coverage ≤ 1 against coverage 2, which on 2Wiki is
coverage-1 against coverage-2 — one gold title named and one hidden, against both named. That is
the classic bridge contrast and it remains testable. What is lost is the coverage-0 stratum, the
hardest case, where neither title anchors the query.

**What did not survive.** NB-5's own expectation that "2Wiki is where the graph wins" was refuted
by the run: 0.2734 against BM25's 0.5164. The topology argument does not rescue an arm whose
binding constraint is that it cannot link 41.1% of queries to any node at all.

### C10 — an orphaned artifact that already answers the question the entry defers

Found by a pre-publication review seat that cloned the repository and read `results/003/` before
running anything, which is exactly the reader the entry invites.

`results/003/oracle-entity-graph.json` was committed with **no producing code anywhere in the
repository**, no Makefile target, and no mention in any protocol, amendment, correction record or
entry. Seventeenth defect, fifth of this exact class, and the first one found by someone other than
the author.

The substance is worse than the bookkeeping. The artifact reports the graph arm's ceiling under a
**perfect extractor and a perfect linker** — corpus document titles as entities, no NER, no
whitelist, no normalisation mismatch, no span-segmentation failure:

| | oracle extractor | spaCy arm | BM25 |
|---|---|---|---|
| R@2 | **0.3344** | 0.2148 | 0.5490 |
| queries retrieving nothing | **1,397 (18.9%)** | 1,309 (17.7%) | — |

So solving extraction entirely moves the arm from 0.2148 to 0.3344 and leaves it roughly 21 R@2
points behind BM25, still returning nothing for about one query in five. The entry's forward
pointer to experiment 004 — swap the extractor — was written as though extraction were the binding
constraint. On this evidence it is not the only one, and publishing that pointer without this
number beside it would have overstated what 004 can buy.

`src/rb/experiments/graph/oracle.py` and `make reproduce-003-oracle` now produce it. **It does not
fully reproduce the committed values, and that is recorded rather than smoothed over.**

| field | committed | reproduced | |
|---|---|---|---|
| R@2 | 0.3344 | **0.3344** | matches |
| mean entities per document | 3.08 | **3.08** | matches |
| R@5 | 0.4464 | 0.4465 | differs |
| nDCG@10 | 0.455 | 0.4554 | differs |
| R@10 | 0.517 | 0.5178 | differs |
| R@100 | 0.6027 | 0.6034 | differs |
| queries retrieving nothing | 1,397 | 1,394 | differs |
| nodes | 66,304 | 66,568 | differs |

The reconstruction matches titles as word n-grams; whatever produced the original matched slightly
differently and found 264 fewer nodes. That rule is not recoverable, because no code for it exists —
which is the defect. The artifact has therefore been **replaced by what the producer emits**, not
kept and annotated, on the same principle the other reconstructions followed: an artifact that does
not reproduce is removed rather than quietly redefined.

The one figure the entry rests on, R@2 0.3344, is identical under both rules, and the conclusion is
unchanged: solving extraction leaves the arm roughly 21 R@2 points behind BM25.

### A false verification claim, published, and corrected here

Commit `5c4034f` on this branch asserted "All nine reproduce exactly. Had they not reproduced, the
artifact would have been removed rather than quietly redefined." **That claim was false when it was
written.** The check behind it read `results/003/oracle-entity-graph.json` while the run that was
supposed to rewrite that file had not finished — so it compared the committed file against a copy of
the committed file and reported eight matches. The producer was also broken at the time: it labelled
oracle query entities `"ORACLE"`, which `extractor.node_strings` drops because the label is not in
the whitelist, so every seed vector was zero and the module would have retrieved nothing for all
7,405 queries had it ever finished.

Neither fault was caught by the author. Both were caught by a code-review seat that read the module
instead of trusting the output, and that noticed the commit timestamp preceded the run it described.

The re-run above was done with the file **deleted first**, so a stale read was impossible rather
than merely unlikely.

### C11 — five more artifacts with no producer, one of them the §8.1 gate

The oracle ceiling was not the only one. A code-review seat grepped every filename in
`results/003/` against the source tree and found five more, which is the same defect for the sixth,
seventh, eighth, ninth and tenth time:

| artifact | what it is | now produced by |
|---|---|---|
| `closure-8-1.json` | **the §8.1 harness-closure GATE** | `make reproduce-003-closure` |
| `annotation-agreement.json` | inter-rater agreement, three-model panel | `make reproduce-003-diagnostics` |
| `extraction-error-analysis.json` | false-positive/negative characterisation | `make reproduce-003-diagnostics` |
| `scoring-ablation.json` | summed against mean document scoring | `make reproduce-003-ablation` |
| `graph-arm-defect-report.json` | narrative record of the PPR defect | labelled as a record, not a measurement |

**The gate is the serious one.** §8.1 halts the run if our BM25 cannot land within 0.05 of
HippoRAG's published BM25 under matching conditions. A gate nobody can re-run is not a gate, and
this one could not be re-run by anyone, including its author.

Its subset rule was also unrecoverable: the committed file reports 9,811 pooled passages, while the
first 1,000 questions in file order give 9,769 and sorted by id give 9,755. No code recorded the
original draw. The rule is now **stated in the module** — first 1,000 question ids in sorted order —
and the gate re-run under it. It passes, and by a wider margin than the artifact it replaces:

| | ours | HippoRAG published | delta | tolerance |
|---|---|---|---|---|
| R@2 | 0.5520 | 0.554 | **0.002** | 0.05 |
| R@5 | 0.7225 | 0.722 | **0.0005** | 0.05 |

The previous artifact recorded deltas of 0.013 and 0.009 against the same reference. Both pass; the
reproducible one passes more convincingly.

`annotation-agreement.json` reproduces exactly. `extraction-error-analysis.json` reproduces five of
six fields, with the count of false positives containing the prepended title at 34 against 36 —
a title-matching rule that differs slightly and, again, no code for the original.

### C12 — the false note published inside the extraction diagnostic

`results/003/extraction-diagnostic.json` carried, as a field a reader is meant to rely on:

> "Gold annotated by one rater who is also the author of the extractor; no inter-annotator
> agreement exists."

Every clause of that was false by the time it shipped. §8.2's second revision replaced the single
annotator with **three independent language-model raters** and majority adjudication;
`extraction-sample.jsonl` records `annotator: llm-panel-3x-majority` with a per-passage
`rater_jaccard`; and `annotation-agreement.json` publishes a mean pairwise Jaccard of **0.9356**.
Two committed files in this repository contradicted each other, and a reader diffing them would
have found it before I did.

It was a hardcoded sentence about the data sitting next to the data. It is now **derived from the
sample** by `extraction_score.provenance()`, so it cannot drift again, and the note states plainly
that the reference is model-annotated rather than human, with what that costs and what it buys.

## What NB-26 itself got wrong

Listed because a corrections file that only records the code's errors and not the process's would
be the same kind of flattering omission it exists to prevent.

1. **The audit that found C5 missed one of its four sites** — the gate, the only one on a decision
   path. Found by an adversarial verifier briefed to refute rather than confirm.
2. **A claimed "7/7 mutations killed" did not survive checking.** An independent 18-mutation sweep
   found 5 survivors, including all three reporting call sites — the ones whose figures moved. The
   sweep had mutated the percentile *rule* and the gate, never the sites calling it.
3. **The first repair of those tests was itself vacuous.** At B = 10,000 the 9749th and 9750th
   draws are equal to four decimals on every fixture tried, so the pins passed under both indices.
   They now lower B to 1,000, where the difference is observable.
4. **Two Makefile targets were added and never run.** Both used bare `python` without
   `PYTHONPATH=src` and failed on invocation — a newly published artifact with a named command that
   did not work, in the round whose subject is exactly that.
5. **A false claim reached a registered amendment.** The measurement moved before `build()`; the
   control *evaluation* did not, so `build()` still raised first and only one of the three fields
   could reach the failure path. Corrected in `3c556ef`.

Items 2–5 were found by a code-quality review that returned BLOCK on the first commit.

### The pre-existing vacuous test

`test_percentile_bounds_pin_exact_values_on_a_known_distribution` used constant-valued arms, so
every bootstrap resample produced the same mean and both candidate indices selected the same draw.
It passed with and without the off-by-one it was named for, **for its entire life**, while
`rb/stats.py` cited it by name as the regression test for that bug. Repaired, with a guard that
fails loudly if the fixture ever goes degenerate again.

This is the third vacuous test found across three audits. It is the strongest evidence in the
sequence that a passing suite is not evidence.

---

## Reproducing every figure above

```
make reproduce-003-analysis        # analysis.json, headroom-control.json
make reproduce-003-controls        # extraction-diagnostic.json, gate-and-seed.json
make reproduce-003-pool-control    # pool-control.json
```

Delete `data/003-pool-entities.json` before verifying the controls: a warm cache replays its own
contents and would pass even against a broken extractor.

The scored arms are not behind a target, deliberately — they take hours and are gated on the
pre-registration tag. The graph arm was nonetheless re-scored from scratch at `3c556ef` to confirm
none of these corrections touches a retrieved ranking: `per_query.jsonl` came back **byte-identical**
(sha256 `725c40c2…`), R@2 0.2148 unchanged.

## C10 — a hard-coded number inside a generated block (2026-08-23)

Found while verifying figures for protocol 004, not by any review pass.

`scripts/sync-003-results.mjs`'s oracle table computed BM25's R@2 and the oracle's R@2 and empty
rate from committed artifacts, and carried the graph arm's empty rate as a **literal `17.7%`**.

**The value was correct** — the arm's own `per_query.jsonl` gives 1309/7405 = 17.68%. That is
precisely what makes it worth recording. It was right by luck rather than by construction, sitting
inside a `RESULTS:003:oracle` block, which is worse than a number in prose: a reader sees it in a
table marked generated and reasonably assumes it traces to data. Had the arm been re-scored, every
other cell would have moved and that one would not, and `--check` would have passed.

The overall empty rate was not derivable from any committed JSON, only from `per_query.jsonl`,
which is why it was typed. `arms_summary.py` now counts it for every arm on both corpora, so the
table reads it like every other cell. The published entry is byte-identical after the fix.

Two things the count surfaced that were not visible before: BM25 and both dense arms return
results for every query on both corpora, and the graph arm returns nothing for **41.07%** of 2Wiki
queries against 17.68% on HotpotQA — the figure the entry quotes in prose as "one query in five"
and "49.2% of bridge-hidden questions" now has its corpus-level counterpart in the artifact.

## C11 — experiment 004 removes 003's central finding (2026-08-26)

Not a defect in 003's code. A later experiment, run as 003 committed it would be, contradicted
what 003 concluded. Recorded here because a reader arriving at 003 must not have to find 004 to
learn that its headline no longer stands.

### What 003 claimed

Prediction A was supported on all six cells: the graph arm's advantage over BM25 was greater on
bridge-absent queries than on coverage-2 queries, by +0.0716 to +0.1568, Holm p 0.0024. The entry
read that as the graph reaching documents a query does not name.

003 already qualified it once: decomposing the differential put the graph term's interval across
zero on all six cells, so the movement was carried by BM25's behaviour rather than the graph's.
The entry says so.

### What 004 found

Swapping spaCy for GLM-4.7-Flash on both corpora, changing nothing else:

| cell | 003 (spaCy) | 004 (GLM) |
|---|---|---|
| ndcg@10 exact | +0.1216 supported | −0.0110 not supported |
| recall@2 exact | +0.0985 supported | +0.0033 not supported |
| recall@2 stripped | +0.0716 supported | −0.0102 not supported |

Zero of six cells supported, one significantly negative. Meanwhile the arm's overall R@2 rose from
0.2148 to 0.3699 and its empty rate fell from 17.68% to 9.89%.

**The differential was an extraction artefact.** It measured how badly spaCy extracted the queries
whose gold documents were named — not how well a graph traverses. 004's protocol registered this
outcome in advance as failure mode 1 and required it be said plainly, which is what this is.

### The oracle sentence, and what it cost

003's closing leaned on the oracle: a perfect extractor and linker gave R@2 0.3344 against BM25's
0.5490, and the entry concluded that "the extractor is not the whole of it" and that it was "less
sure than I was this morning" that the extractor was the interesting question.

**004 scored 0.3699 and passed that ceiling.** The oracle used corpus document titles as entities:
perfect linking over a *fixed entity set*. It bounded that definition of entity, not extraction.
GLM returns about ten entities per passage where titles give one.

So the closing argued from a bound that was not one, and reached a conclusion — extraction is not
the binding constraint — that 004 reverses. Extraction was the binding constraint on overall
performance. It simply had nothing to do with the class differential.

I registered that same false ceiling again in protocol 004 §2, in the sentence "a real extractor
cannot beat a perfect one", specifically so it could not be spun later. The experiment falsified
it. That is the system working, and it is the second time an oracle has been mistaken for an upper
bound in this sequence.

### What still stands in 003

The negative control (prediction B, `confirmed_no_advantage` on all six cells) holds under both
extractors. Predictions C and D on 2Wiki remain refuted, and C is refuted harder under GLM
(−0.3416 against −0.2587). The corpus-level result — the graph arm loses to BM25 on both corpora —
stands, and 004 widens rather than closes that gap's explanation.

### Where the entry is corrected

`content/experiments/003-my-graph-retriever-was-doing-entity-lookup.mdx` carries a correction
section pointing here. The published numbers in 003 are unchanged; what changed is what they mean.

## C12 — the 2Wiki crossover compared two different retrievers (2026-08-26)

Found while running 004's registered analysis, before its result was written up.

`analysis2wiki` quotes HotpotQA's figures as module constants — `HOTPOT_GRAPH_R2 = 0.2148` — so a
2Wiki run cannot silently alter HotpotQA's published numbers. That is a good property, and it
became a defect the moment the module was pointed at a second arm: the constants are 003's spaCy
figures, so running the GLM arm computed prediction D's crossover from **GLM on 2Wiki against
spaCy on HotpotQA**.

It reported `supported: true`, 19.48 shrink points against a registered threshold of 10 — a
prediction 003 had refuted, apparently now confirmed. Corrected to use each arm's own HotpotQA
figure, the same run reports **3.97 points and is refuted.**

The failure is invisible in the output. Both numbers are real, both are published, and the
crossover reads as a finding either way. `tests/test_crossover_uses_one_arm.py` now pins that each
arm quotes its own figure, and that the two differ — an assertion that would pass vacuously if
they did not.
