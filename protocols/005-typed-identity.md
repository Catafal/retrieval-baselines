# Protocol 005 — is the graph's problem the walk or the identities it walks over

**Status: frozen on tagging as `protocol-005`.** Written and committed before the first typed-identity
arm is scored. Stage 0 is already measured and published at `results/005/RESULT.md`; the coverage
figures below are quoted from it rather than anticipated, and the alias-affected subsets it fixed
are the populations this protocol decomposes over.

## 1. What this settles, and what it cannot

003 and 004 both key entities by exact normalised string. "Cleveland State" and "Cleveland State
University" are two nodes with no edge between them, and no walk crosses what the node set never
joined.

004 established that this arm's performance lives in the node set rather than the walk: a better
extractor moved HotpotQA R@2 from 0.2148 to 0.3699 and passed the 0.3344 oracle ceiling that
protocol 004 §2 registered in advance as unbeatable, because that oracle bounded one definition of
entity rather than extraction itself.

The question left over is whether the node set's remaining weakness is *identity* — the same thing
under two names occupying two nodes. This experiment holds the walk, the pools, the metric, the
coverage classes and the Holm family fixed, and changes only what counts as one entity.

**It cannot settle whether graphs beat BM25.** See §3.

## 2. The design, and the confound it exists to prevent

A 2×2. Extractor on one axis, identity on the other.

| | string identity | typed identity |
|---|---|---|
| spaCy | `graph` (003, scored) | `graph-typed` |
| GLM-4.7-Flash | `graph-glm` (004, scored) | `graph-glm-typed` |

The string column is already published and is not re-run. Only the typed column is new.

**Why a 2×2 and not a swap.** Stage 0 measured the two components interacting: the alias-affected
population is 4.2% under spaCy and 11.0% under GLM on 2Wiki, and 16.4% against 17.8% on HotpotQA.
So on one corpus the better extractor roughly doubles what identity has to work with, and on the
other it changes almost nothing. An experiment that swapped both at once would report a combined
effect and attribute it to whichever component the author had in mind. The cells make that
impossible.

## 3. The registered expectation, before any number

**The graph arm does not beat full BM25 on either corpus, under either identity.** BM25 scores
0.5490 R@2 on HotpotQA against the graph's best published 0.3699. Stage 0 measured the intervention
at **under 5% of nodes merged** on every cell. An edit that size does not close a gap of 0.18, and
saying so here means "the graph still loses" cannot later be reported as news, nor its absence as
disappointment.

This is registered as an expectation, not a falsifier. Nothing below depends on it.

## 4. Prediction A — typed identity raises the graph arm's R@2

**Statement.** On both corpora and under both extractors, the typed arm's R@2 exceeds the string
arm's.

- **Statistic:** paired difference in R@2 per query, typed − string, same arm otherwise.
- **Estimator:** percentile bootstrap over queries, B = 10,000, seed 20260820, the add-one p-value
  from `stats.paired_bootstrap`.
- **Family:** four cells (2 corpora × 2 extractors), Holm–Bonferroni at α = 0.05 across the four.
- **Decision per cell:** `supported` when the interval excludes zero and lies above it after
  correction; `no_advantage` otherwise.

**Registered magnitude.** Stage 0 puts the alias-affected population between 4.2% and 17.8% of
queries. If typed identity helped every affected query and hurt none — which it will not — the
ceiling on the overall gain is that share. **An overall gain above 0.05 R@2 would be larger than
the coverage supports and should be treated as a bug to be found, not a result to be reported.**

## 5. Prediction B — the gain is concentrated where identity changed something

This is the mechanism claim, and it is the one worth the experiment. A number that moves without
localising here does not show that identity resolution is what moved it.

**Statement.** The typed − string difference is larger on the alias-affected subset than on its
complement.

**The subsets are already fixed.** Stage 0 wrote them before any score existed, per
`protocol-005-identity-amendment-1`: a query is alias-affected when at least one of its entities
reaches a different set of documents under typed identity than under string identity.

| corpus | extractor | affected | MDE at 80% power |
|---|---|---|---|
| HotpotQA | spaCy | 1,217 | 0.0568 |
| HotpotQA | GLM | 1,316 | 0.0546 |
| 2Wiki | spaCy | 414 | 0.0974 |
| 2Wiki | GLM | 1,085 | 0.0601 |

**They may not be reselected.** Recomputing "affected" after seeing which queries improved would
confirm any result whatsoever, which is why it was fixed a stage early.

- **Statistic:** difference of subset means — (typed − string) on affected, minus (typed − string)
  on unaffected.
- **Estimator and family:** as §4, same four cells, its own Holm family.

**The 2Wiki spaCy cell is underpowered by design and is reported as such.** Its MDE is 0.0974, so
a real effect of 0.05 there would be invisible. That cell reports `underpowered` rather than
`no_advantage` when its interval includes zero and its width exceeds the MDE — silence about a
sample that could not have spoken is not evidence of absence.

## 6. Falsifiers, named before running

**F1 — the gain is flat.** Prediction A returns `no_advantage` on all four cells. Then
alias-merging is not what this arm was missing, 004's extractor result is the whole story, and the
entry says so.

**F2 — the gain arrives but does not localise.** A returns `supported` and B does not. Then
something in typed identity is doing work other than resolving identities — the likeliest
candidate being that merging shrinks the node set and changes the specificity weighting, which is a
different mechanism wearing this one's clothes. The mechanism claim is wrong even though the number
moved, and the entry leads with that.

**F3 — recall rises while precision falls.** Merging two entities that share a name invents a path
that does not exist. Stage 0 measured R6 dropping only 98 ambiguous aliases on HotpotQA and 32 on
2Wiki, so this is registered as unlikely rather than impossible. **Both precision and recall are
reported per cell whatever happens**, so a recall gain bought with precision cannot be shown as a
clean win.

## 7. What is held fixed

Pools, queries, qrels, coverage classes, `GRAPH_MEASURES`, damping 0.5, node specificity, the
whitelist, the extraction caches, the bootstrap seed, and both baselines. The registry is the
committed Stage 0 snapshot, unmodified — its sha256 is recorded in
`results/005/redirects-*-manifest.json` and any re-fetch invalidates this protocol.

The linker enters through the seam registered in `005-identity.md` §5: one linker stored once and
passed to `build()` from `fit()`, so a graph keyed by one identity and seeded by another is
unrepresentable rather than merely forbidden.

## 8. Reporting obligations

- Every cell reports R@2, precision, recall and its decision, whichever way it goes.
- The underpowered cell is labelled, not silently folded into the nulls.
- If F2 fires, the entry leads with the mechanism failing, not with the R@2 that moved.
- The 2×2's interaction is reported even though no prediction is registered on it, because Stage 0
  already showed the components interacting and omitting it would be a choice made after seeing the
  number.
