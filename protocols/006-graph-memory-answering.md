# Protocol 006 — does a precomputed graph walk move the hops out of the model

**Status: frozen on tagging as `protocol-006`.** Written and committed before any answering arm
is scored. The question sample, the corpus, the arm definitions, the token budget, the turn cap,
the abstention rule and the Holm families below are all fixed here. The graph was built before
this tag; building it reveals no outcome, and its cost is reported in §9.

## 1. What this settles, and what it cannot

001 through 005 measured RETRIEVAL: whether an arm ranks the right documents. All five put the
graph well behind dense retrieval — 0.3756 against 0.6957 R@2 on HotpotQA at the graph's best
configuration — and 005 additionally falsified the author's own explanation for the graph's
behaviour with a test he had registered for that purpose.

None of them measured ANSWERING. The claim this experiment responds to is a different one:

> "the graph walks the hops, never the model" — graph-memory-starter, Glitch Cat Club, 2026

That is a capability and cost claim. Its published evidence is ONE question, asked under two
conditions across three model tiers, over eight hand-modelled documents whose entities, aliases
and typed relations were authored by hand. Its reported finding is that a weak model reaches 1 of
3 hops when driving its own search and answers wrongly, and 3 of 3 when handed graph facts.

**This experiment cannot settle whether graphs beat dense retrieval at ranking.** 001-005 already
answered that, against the graph. It also cannot settle whether hand-modelled corpora beat
extracted ones: the prior art's graph is hand-authored and this one is extracted, and that
difference is held fixed rather than tested. Nor can it settle anything about personal or
agent-memory corpora, which is the setting the author actually uses a graph in. HotpotQA is
encyclopaedic. These limits are registered here so they cannot be claimed later as findings.

**And it measures ANSWERING, not HOPPING.** The prior art's claim is about a trace — 3 hops of
3 — and exact match cannot see a trace. A model can reach the right answer from parametric
memory, from one lucky lexical hit, or from a genuine bridge, and EM scores all three the same.
§9 registers two partial proxies (answer-presence in the injection, and turn counts for the
grep arm), and neither is a hop count. No arm's hop count is observable for the injected arms
by construction.

## 2. The corpus, frozen

100 questions drawn from HotpotQA distractor validation, `type == "bridge"` and `level == "hard"`,
seed 20260820, sorted by id before sampling so parquet order cannot affect the draw.

The pool is the union of those questions' own 10-passage contexts, deduplicated by title: **990
documents, 577,517 characters**. Every question's 2 gold passages are in it, and every other
question's 8 distractors are in it too, so a question's haystack is 990 documents rather than 10.

| | frozen value |
|---|---|
| questions sha256 | `47397643a6ca3e643d541700650d6035002670c5eae3613214764b7c83673918` |
| pool sha256 | `953c12ea4f8bd49cb234ff07fcd5155666f92d9776644d9f16196c6a67a00cbc` |
| gold docs per question | exactly 2, verified |

**Why a small pool is not a soft pool.** BM25 scores R@2 = 0.550 on these 990 documents against
0.549 on 003's 66,581-passage pool, and dense scores 0.742 against 0.696. The corpus is three
orders of magnitude smaller and very slightly easier; it is not a different problem.

**Single-hop solvability is measured, not assumed.** Under 003 §4's frozen `coverage` definition,
8 of the 100 questions name both gold titles in the query text and are therefore solvable without
a second hop. They are reported as their own stratum. They are not removed, because removing them
after seeing scores is the error 005's amendment 1 exists to prevent.

## 3. Contamination, which is large and is measured per question per tier

A closed-book probe — no corpus, no tools — is a scored arm, not a diagnostic. It must be,
because these models have read Wikipedia. A 20-question pilot measured 30% EM on haiku and 40% on
sonnet and opus, and that pilot is discarded: it ran before the harness leak in §8 was closed.

Every analysis is reported twice: over all 100 questions, and over the **contamination-resistant
subset** — the questions that tier answered wrongly closed-book. Membership is per tier, is
defined here before any arm runs, and is never recomputed after seeing a retrieval arm's score.

The resistant subset is not a random sample of the corpus. It is selected on being hard relative
to parametric memory, which plausibly correlates with genuine multi-hop structure, and that
selection probably flatters the graph. Registered as a limitation, in advance.

## 4. The arms

Six arms, one question sample, one pool. Every prompt is built by `arms.py` and every arm shares
`ANSWER_RULE` verbatim; arms differ in their context block and their tool grant and in nothing
else. A word-for-word diff of any two arms' prompts shows the context block alone.

| arm | context | tools |
|---|---|---|
| `closed-book` | none | none |
| `grep` | none — the agent searches the corpus itself | `Grep,Read,Glob`, cap 20 turns |
| `bm25` | top passages by BM25 | none |
| `dense` | top passages by bge-base-en-v1.5 | none |
| `oracle` | the 2 gold passages | none |
| `graph-facts` | the walk's typed triples and entity notes | none |

**The token budget is equalised at 400 tokens for every injected arm**, which is the prior art's
own reported injection size. Whole blocks only, so no arm receives half a passage. Without this,
a passage arm could be handed 3,000 tokens against the graph's 400 and the difference in context
VOLUME would be reported as a difference in context STRUCTURE.

**`oracle` is the arm that makes the mechanism claim falsifiable.** It supplies the same
information without the graph's structure. If `graph-facts` merely matches `oracle`, then any win
over `grep` is short clean context, not precomputed hops, and the mechanism claim is dead
regardless of what the headline table looks like. This arm is also the strawman guard: if `grep`
on opus cannot approach `oracle` on opus, the grep arm is broken and no comparison is reported.

**The grep arm is given every advantage that does not decide the outcome.** Corpus documents are
one markdown file each, named for their title, with the title as the first line — the same
affordance the prior art's corpus has. The turn cap is 20, roughly five times the four calls a
two-hop chain needs. It gets the full read-only toolkit, not a hobbled subset.

**And it gets a two-hop strategy hint that no injected arm receives**, quoted here verbatim
because an earlier draft of this protocol claimed the arms differed only in context and tools,
and that was not true:

> "The corpus is the directory {path} — markdown files, one document per file, each beginning
> with its title as a heading. Search it with Grep and read files with Read. Answering usually
> requires finding one document, extracting a name or term from it, and then searching for a
> second document about that term. Search as many times as you need."

This is registered as an advantage granted to the CONTROL. It can only make a graph win harder
to obtain, never easier, which is why it is kept rather than removed. Everything else in the
prompts — including `ANSWER_RULE` — is shared verbatim.

**F4 was checked before tagging, and it fired.** Three questions were run through `grep` and
`oracle` on haiku and discarded. `grep` scored 0 of 3; `oracle` scored 2 of 3. There were no
permission denials and the turn cap was never reached, so the arm was not being blocked or cut
short — it was searching an empty directory. Every call runs in a fresh temporary directory so
no repository content can leak in, the corpus is attached with `--add-dir`, and **nothing told
the agent where it was**. With the path supplied, the same three questions score 2 of 3: exactly
the oracle's score.

The probes are not scored, are not written to `calls.jsonl`, and no arm was retuned on their
outcome beyond supplying the missing path. Had this not been checked, the published grep arm
would have been a strawman and every graph win in the table would have been an artifact of a
harness defect.

**Extracting the graph from the whole pool, gold passages included, is legitimate** and is
stated rather than left to be derived: it is the same information every other arm can reach,
and the grep arm has the raw text of all 990 documents.

**Budget caveat, stated because it is not exact.** `fit_budget` never splits a block, and every
arm keeps its first block even when that block alone exceeds 400 tokens. The realised per-arm
token distribution is therefore reported rather than assumed. Measured before any scored call:
the graph arm's injection is mean 268 and max 398 estimated tokens.

**The extractor is pinned to haiku, the weakest tier under test.** The graph is built by an LLM
reading all 990 documents, which is inference-time compute moved to build time. Were the builder
stronger than the answerer, the graph arm would be carrying a smarter model's reasoning into a
weaker model's answer and reporting it as structure. Pinning extraction to the weakest tier makes
that impossible. It costs the graph arm its best case and is registered for that reason.

## 5. Predictions

Per-query outcome is exact match against the gold answer string, SQuAD-normalised. No judge model
appears anywhere in the scoring path.

**The primary metric is normalised EQUALITY, and that is a correction made before tagging.** An
earlier draft made containment primary — a prediction counted correct if it contained the gold
span within six words. Review broke it in seven ways, including `Iron Man 3` scoring correct
against gold `Iron Man`, `no` matching inside `North Dakota`, and
`the answer is not France, it is Germany` scoring correct against gold `France`. Every failure
requires the prediction to be LONGER than the gold, so the rule inflated whichever arms narrate
most — which are both arms in the primary contrast, by an unknown net amount. Three variants now
ship: `em` (normalised equality after stripping stock openers, PRIMARY), `em_lenient`
(token-subsequence containment with a negation guard), and `em_strict` (no narration strip).

**Decisions require BOTH a bootstrap interval clear of zero AND a Holm-adjusted p at or below
0.05.** An earlier draft computed Holm and then decided on the interval alone, which made every
declared family decorative. Estimator: `stats.paired_bootstrap`, B = 10,000,
seed 20260828, add-one p-value, over per-query differences.

Model tier is a WITHIN-question factor: the same question is put to all three tiers in all six
arms, so the design is fully crossed and repeated-measures, and every contrast is paired at the
query level.

**P1 — PRIMARY. The graph rescues the weak model.**
On haiku, `graph-facts` EM exceeds `grep` EM. Statistic: mean of
`EM(graph-facts, haiku, i) − EM(grep, haiku, i)`. One test, α = 0.05, one-sided.

**P2 — SECONDARY, Holm family of 3.** The same contrast on each of haiku, sonnet, opus.

**P3 — SECONDARY, REGISTERED UNDERPOWERED.** The interaction: the graph-minus-grep difference is
larger on haiku than on opus. Per-query statistic
`d_i = [EM(gf,haiku,i) − EM(grep,haiku,i)] − [EM(gf,opus,i) − EM(grep,opus,i)]`.
This is the prior art's actual thesis and it is the quantity this n cannot properly test. See §6.

**P4 — SECONDARY, Holm family of 3.** `graph-facts` uses fewer context tokens than `grep` at
equal or better EM, per tier. Context is `input + cache_read + cache_creation` minus the measured
zero-context harness baseline, so that CLI overhead is not counted as corpus the model read.
**The conjunction is tested, not just the token half**: a cell is `supported` only when the token
difference is significant AND the paired EM difference's interval does not exclude zero from
below. Without that, an arm could win P4 by being cheap and wrong.

**P5 — ADVERSARIAL, registered because the author does not want it to be true.**
`graph-facts` does NOT exceed `dense` on EM on any tier. If the graph loses to dense at answering
as well as at ranking, then "graphs win on some architectures" remains unsupported on this corpus
and the entry leads with that sentence.

Everything else — `bm25`, `oracle` and `closed-book` contrasts, the resistant-subset splits, the
coverage stratum, latency, abstention and turn counts — is EXPLORATORY, uncorrected, and labelled
as such in every table cell.

## 6. Power, stated before the data

EM is binary and paired, so the minimum detectable effect follows the discordant-pair rate `d`,
not a two-proportion formula. Using a two-proportion formula on a differenced statistic is
precisely the estimand error recorded in `005-amendment-1-mde-estimand.md`.

`MDE = 2.8016 × sqrt(d/n)` for a within-tier contrast; `sqrt(2)×` that for the interaction.

| n = 100 | d = 0.30 | d = 0.40 |
|---|---|---|
| within-tier contrast (P1, P2) | 0.153 | 0.177 |
| interaction (P3) | 0.217 | 0.251 |

**P1 is falsifiable at n = 100 against an effect of 0.20.** P3 is not: its MDE exceeds any effect
this experiment plausibly contains. **P3 therefore reports `underpowered` rather than
`no_advantage` whenever its interval includes zero**, and this is registered now, before the
number exists, so that a silence which could not have spoken is never read as evidence of absence.
The realised `d` is reported per contrast so a reader can check which column applies.

## 7. Falsifiers, named before running

**F1 — P1 nulls.** The prior art's single-question result does not survive 100 questions. The
entry leads with that and the sequence closes where 005 left it.

**F2 — P1 holds but `oracle` matches or beats `graph-facts`.** Then the graph is not moving hops;
it is supplying a short clean context, which a list of two passages does equally well. The
mechanism claim is wrong even though the number moved, and the entry leads with the mechanism
failing rather than with the EM that rose. This is the outcome the author would most be tempted to
omit, so it is named here as a lead-with obligation.

**F3 — P1 holds and P3 nulls.** The graph helps every tier equally, so the effect is context
injection rather than hop-shifting, and "the graph rescues weak models" is unsupported.

**F4 — the grep arm is a strawman.** `grep` on opus falls far below `oracle` on opus, or its
max-turn exhaustion rate is high. Then the control is broken and NO arm comparison is reported at
all until it is fixed.

**F6 — the walk is doing nothing and the seed match is doing everything.** `recall()` seeds by
matching entity names and aliases against the question text, so `graph-facts` could be lexical
retrieval wearing arrow notation. §9's decomposition adjudicates it: **if answer presence under
the shipped walk exceeds the seed-only neighbourhood by 0.05 or less, OR exceeds the
edge-permuted placebo by 0.05 or less, then any P1 win is labelled SEED-MATCH, NOT WALK and the
entry leads with that whatever the EM table says.** Measured on the partial graph before
tagging, for reference: seed-only 0.07, shipped 0.27, placebo 0.10.

**F5 — P5 fires the other way**: `graph-facts` beats `dense` on EM while losing to it on R@2.
Ranking quality and answering quality would have come apart, which would be the most interesting
result in the whole sequence. It gets its own section and an explicit statement that it needs
independent replication before anyone builds on it.

## 8. The harness is part of the experiment, and two leaks were found and closed

Both were measured, not assumed, and both would have invalidated the run.

**The author's global `~/.claude/CLAUDE.md` reached the model in every call.** Claude Code loads
user-level settings regardless of working directory, so an empty temp directory does not stop it;
probed directly, the model quoted that file back verbatim. Closed with `--setting-sources ""`,
re-probed, and the model now replies `NONE`.

**`--add-dir` grants corpus access but does not confine the agent to it.** A grep agent given the
corpus directory read a canary file outside it. Closed with `--restricted`, re-probed: the read is
refused with a permission denial and corpus access still works.

Every call runs with `--output-format json --setting-sources "" --strict-mcp-config --restricted`
in a fresh temporary directory, and the resolved model snapshot id, session id, service tier,
stop reason, turn count and permission-denial count are recorded per call.

**Outcomes are typed, because three different things were previously one empty answer scored as
wrong.** `max_turns` is scored WRONG and reported separately, since exhaustion is a real
capability difference and the grep arm is mechanically more exposed to it. `timeout` and
`api_error` are retried once, then excluded from the denominator and reported as a rate; scoring
infrastructure flakiness as a wrong answer would penalise the arm that takes longer.

**`no_json` is a registered outcome**, not an untracked one. It covers a process that produced
no parseable result and any unexpected harness exception. It is retried once, then excluded from
the denominator like the other infrastructure outcomes, and its rate is reported per arm — the
grep arm, being the only one with long transcripts and a turn cap, is the most exposed to it.

**Jobs are interleaved across arms with seed 20260828** before submission. A run of thousands of
calls spans hours, service conditions drift over hours, and emitting one arm at a time would let
that drift land on one arm and be read as an arm effect.

## 9. Reporting obligations

- Every arm reports EM, strict EM, token F1, abstention rate, max-turn exhaustion rate, context
  tokens, cost and latency, per tier, whichever way the result goes.
- **Both EM variants ship.** The lenient rule credits a prediction that contains the gold span,
  which can only ever help an arm that narrates, and the arms differ in narration by
  construction. Publishing one without the other would let a scoring choice decide the result.
- **The graph's build cost is reported in the same table as the query-time results**: extractor
  calls, tokens, dollars, wall time and model tier, plus cost amortised per question. A method
  that wins at query time by spending more at build time has not won until both are on the page.
- The underpowered prediction is labelled, never folded into the nulls.
- If F2, F4 or F6 fires, the entry leads with it.

**The seed-versus-walk decomposition, registered here and computed with no model calls:**
answer presence at hops=0 (seed neighbourhood only), at the shipped configuration, over an
edge-permuted placebo graph with the same nodes and degrees, and with the extractor's free-text
entity descriptions removed. Answer presence is the rate at which the gold answer STRING appears
in the injection: a necessary condition for answering by copying, and therefore a ceiling on the
arm's EM under a copy-only model. **If measured `graph-facts` EM exceeds it, the arm is answering
from parametric memory rather than from the injection, and that is reported explicitly.**

**The realised depth histogram of injected triples is a mandatory diagnostic.** The walk is run
at `hops=3, top_k=8` with NO depth reservation, which is the prior art's own configuration.
Review found a flat top-k cut makes `hops` nearly decorative, and a depth-reserved variant was
written and measured before this was fixed. Both numbers are disclosed rather than buried:
reserve=0 gives 28% of injected triples at depth ≥ 1 and answer presence 0.26; reserve=4 gives
55% at depth ≥ 1 and answer presence 0.18. **Fidelity decided this, not the score** — improving
on the prior art's mechanism would be a different experiment — and the reserved variant is
reported as an exploratory sensitivity. If the shipped walk delivers almost nothing beyond depth
1, then "the graph walks the hops" is a claim the design does not keep, and that is a finding.

**Extraction yield is reported in the headline table, and gates interpretation.** The graph is
only as real as the documents that parsed. `gold_docs_in_graph ∈ {0,1,2}` is computed per
question and `graph-facts` is reported stratified on it. **If yield falls below 90%, F1 reports
as "unfalsifiable at this extraction yield" rather than as a graph loss** — a null would
otherwise be equally consistent with "graphs do not move hops" and "haiku could not emit valid
JSON for much of the corpus".

**Two honesty notes on the contamination stratum.** The closed-book probe measures answer-level
parametric recall only; dataset-level memorisation — recognising the item as HotpotQA — is not
separable at this n. And stratum membership is a ONE-DRAW measurement of a stochastic model, so
a question answered correctly by luck is excluded and one missed by luck is included. Regression
to the mean makes every arm look better on the resistant subset, equally. Resistant-subset
results are exploratory only.
