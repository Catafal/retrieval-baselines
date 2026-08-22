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

Reconstructed by `src/rb/experiments/graph/oracle.py` and `make reproduce-003-oracle`, and checked
against the committed values before adoption. All nine reproduce exactly: R@2 0.3344, R@5 0.4464,
nDCG@10 0.455, R@10 0.517, R@100 0.6027, 1,397 empty, 66,304 nodes, 3.08 entities per document.
Had they not reproduced, the artifact would have been removed rather than quietly redefined.

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
