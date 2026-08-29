# 006 — does the graph move the hops out of the model?

Last entry in the sequence. 001-005 measured **retrieval**: does an arm rank the right
documents. 006 measures **answering**: does the model get the question right, and what did
it cost. That is the axis graph-memory-starter actually argues on, and the one the sequence
has never touched.

Prior art: https://github.com/Glitch-Cat-Club/graph-memory-starter — 8 hand-modelled docs,
ONE question, three model tiers. Its finding: Haiku reaches 1/3 hops driving its own search
and gets the answer wrong; with graph facts injected it reaches 3/3 and gets it right.
Fable and Sonnet are correct either way. We are testing that at n>1 with real baselines.

## 1. The claim

A precomputed graph walk moves multi-hop traversal from inference time to build time. So the
weaker the model, the more the graph is worth — not because it retrieves better, but because
the model no longer has to do the hopping.

This is a **capability/cost** claim, not a ranked-retrieval claim. 005 already settled that
the graph ranks worse than dense on both corpora. Both can be true.

## 2. Design

Corpus and pool unchanged: the 66,581-passage HotpotQA pool from 003. Questions restricted to
`type == "bridge"` (a genuine chain), stratified sample n = 500, seed 20260820, drawn and
frozen before any arm runs.

Five arms, same questions, same pool:

| arm | what the model sees |
|---|---|
| `grep` | nothing; the model drives grep + read itself, max 15 tool calls |
| `bm25` | top-5 passages from BM25 |
| `dense` | top-5 passages from bge-base-en-v1.5 |
| `graph-docs` | top-5 passages from 005's best arm (GLM + typed identity) |
| `graph-facts` | the starter's shape: the PPR walk's entity triples, no passages |

Two model tiers: **Haiku 4.5** and **Sonnet 5**. Optionally a third, GLM-4.7-Flash, which is
near-free and pushes the weak end further — the interaction is the finding, so more tiers is
more signal.

Answer scoring: SQuAD-normalised exact match against the gold `answer` string, plus token F1.
No judge model anywhere in the scoring path.

Also recorded per query: input tokens, output tokens, tool calls, wall latency.

## 3. Predictions, registered before any call

**P1 — the graph rescues the weak model.** On Haiku, `graph-facts` EM exceeds `grep` EM.
Paired bootstrap over queries, B = 10,000, seed 20260820.

**P2 — the advantage shrinks with capability.** The `graph-facts` − `grep` difference is
larger on Haiku than on Sonnet. This is the interaction and it is the actual thesis; P1
without P2 is just "injecting context helps", which is not news.

**P3 — the adversarial one.** `graph-facts` does NOT exceed `dense` on EM for either model.
Registered as an expectation the author does not want to be true. If the graph also loses to
dense on *answering*, then "graphs win on some architectures" remains unsupported on this
corpus and the entry says so plainly.

**P4 — cost.** `graph-facts` uses fewer input tokens than `grep` at equal or better EM.

Holm-Bonferroni within each prediction family.

## 4. Falsifiers

**F1** — P1 nulls. The starter's single-question result does not survive n=500; the entry
leads with that and the sequence closes where 005 left it.

**F2** — P1 holds but P2 nulls. The graph helps every tier equally, so it is context
injection doing the work, not hop-shifting. The mechanism claim is wrong even though the
number moved. Same shape as 005's F2, and the entry must lead with the mechanism failing.

**F3** — P3 fires (graph beats dense on EM while losing on R@2). That would be the most
interesting outcome in the whole sequence: ranking quality and answering quality coming
apart. It gets its own section and an explicit "this needs replication before anyone acts
on it".

**F4** — the `grep` arm is incompetently implemented. Guard: its EM on Sonnet must be within
the range a Sonnet reading the gold passages achieves, or the arm is a strawman and nothing
is reported. An oracle arm (gold 2 passages injected) is run first to set that ceiling.

## 5. What this cannot settle

Not whether hand-modelled corpora beat extracted ones — HotpotQA entities are extracted, and
the starter's are hand-authored with hand-written aliases. That difference is real and this
experiment holds it fixed rather than testing it. Named here so it cannot be claimed later.

Not whether graphs help on a personal/agent-memory corpus, which is Cortex's actual setting.
HotpotQA is encyclopaedic. Registered as a limit, not a finding.

## 6. Cost

~5 arms x 2-3 tiers x 500 questions = 5,000-7,500 model calls, and the `grep` arm is
multi-turn so it dominates. Rough estimate $40-70 on the Anthropic API. **Needs Sir's
approval before the first scored call.** A 50-question smoke run comes first to price it
exactly and to validate the harness; the smoke run is discarded, not scored.

## 7. Order of work

1. oracle arm + `grep` arm on 50 questions — validate F4's guard, price the run
2. freeze the question sample, tag `protocol-006`
3. run the five arms x tiers
4. analysis, entry, plate
