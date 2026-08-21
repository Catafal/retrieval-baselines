# Amendment 4 to protocol 003 — the pool control measures what it reports

**Status:** registered after `protocol-003`, before the corrected control is published.
**Amends:** `protocols/003-graph-arm.md` §9.
**Reason:** §9's control contained a check that could not fail and three fields that were never
measured. Correcting it changes what a field of a tagged control MEANS, which is an amendment
rather than a fix, and is registered here rather than slipped in.

## What was wrong

`run.py` called `controls.pool_construction` with:

```python
unresolved=0, collisions=0,
gold_titles_matched=len(qrels), gold_queries=len(qrels)
```

`controls.pool_construction` compares `gold_titles_matched` against `gold_queries`. Both were the
same expression, so that check was `len(qrels) == len(qrels)` — a tautology. `unresolved` and
`collisions` were literals, and no function in `pool.py` computed either: both quantities were
raised on and never returned, so there was no path on which a nonzero value could reach the
control.

Three of the control's seven published fields were therefore assertions, inside a block reporting
`"passed": true`, published in all five arms' `summary.json`.

## What changed

`pool.construction_counts()` measures all of them, without raising, and runs BEFORE `pool.build()`
— the ordering matters, because `build()` raises on an unresolved title and a control run after it
could only ever observe zero.

`gold_titles_matched` is redefined from "`len(qrels)`" to **the number of judged queries whose
every gold document id is present in the pooled corpus.** That is the property the field's name
always claimed and the property that makes the pool an exactly-identified subset of BEIR's corpus.
Nothing in the repository tested it before.

`title_index()` keeps its raise. `build()` keeps its raise. Those remain the enforcement; the
amendment only makes the counts observable so the control can fail on its own terms.

## Effect on published numbers

None. Measured after the change:

| field | before (asserted) | after (measured) |
|---|---|---|
| `questions` | 7405 | 7405 |
| `passages` | 66581 | 66581 |
| `title_slots` | 73700 | 73700 |
| `unresolved_titles` | 0 *(literal)* | 0 *(measured)* |
| `title_collisions` | 0 *(literal)* | 0 *(measured)* |
| `gold_titles_matched` | 7405 *(tautology)* | 7405 *(real intersection)* |
| `passed` | true | true |

Every value is unchanged. What changed is that they are now facts about the data rather than
restatements of the call site. Independently checked while auditing: 0 of 13,783 distinct gold
document ids are absent from the pool, and 0 queries have an unreachable gold document.

## Where the corrected control is published

`results/003/pool-control.json`, produced by `make reproduce-003-pool-control`.

The five existing `results/003/pool/<arm>/summary.json` files are **not modified.** A command that
rewrote their `pool_control` blocks in place was considered and rejected:
`results/003/pool/graph-defective/README.md` states that artifact is "kept, not deleted... a
reader must be able to see what was published", and rewriting a deliberately-preserved defective
run's control block would contradict the reason it exists. The corrected control is therefore
published beside the old ones, and this table is the diff.
