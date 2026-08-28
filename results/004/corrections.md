# Experiment 004 — corrections

Every figure or claim in 004 that changed after publication, what changed it, and which review
pass should have caught it and did not.

Same contract as [`results/003/corrections.md`](../003/corrections.md): when something published
moves, the move is reproducible too — what it was, what it is, and why.

**No correction listed here changes a measured number.** The ablation's figures are unchanged and
recompute exactly from the counts committed beside them, which `make verify-004-ablation` now
demonstrates offline.

---

## C1 — the entry named a paid command and promised it was free

**Found:** 2026-08-28, while promoting `wilson_interval` and `mde_two_proportion` out of this
experiment's producer and into `stats.py` for experiment 005. The promotion was to be verified by
recomputing this artifact byte-identically; running the entry's own named command to do that is
what surfaced the defect.

**What the entry said.** Under "How to check this instead of trusting me":

> `make setup && make reproduce-004-ablation`
>
> The extraction cache is committed, so the scored run replays from it rather than re-calling a
> paid endpoint.

**What is true.** `reproduce-004-ablation` bypasses the cache by design. The cache is keyed by
reasoning setting, so it cannot answer a question about the setting; the ablation must re-call the
endpoint at both settings or measure nothing. A reader who ran the command got
`RuntimeError: OPENROUTER_API_KEY is not set`.

Both halves were individually true. The scored run does replay from the cache. The ablation never
did. The reassurance was attached to the wrong command, and the result was an entry whose stated
way of checking it could not be followed.

**Severity.** This is the entry's central promise, not a detail inside it. BRAND.md prohibition 5
requires a `measured` artifact be reproducible by a single named command; the named command
required a key and a payment the entry said were unnecessary.

**Fixed by:**

- `verify-004-ablation`, a free offline target that recomputes every derived figure in
  `reasoning-ablation.json` from the counts committed beside it — empty rates, Wilson intervals,
  the MDE, per-arm precision/recall/F1, the pooled-vs-per-passage consistency, and the seeded
  paired bootstrap.
- The check shares `run()`'s arithmetic rather than reimplementing it. A checker carrying its own
  copy of the formula agrees with itself, not with the producer.
- `reproduce-004-ablation` now carries a comment saying it costs money and why it cannot use the
  cache.
- The entry text now names the free command, and states plainly that the measurement itself is
  paid.

**Verified non-vacuous.** Four figures were each perturbed in the committed artifact and the check
was required to fail: `empty_rate_ci95`, `mde_at_80_power`, `paired_f1_difference.ci95` and a raw
`tp`. All four killed. `tests/test_ablation_check.py` keeps six such mutations plus a
pooled-counts consistency mutation as regressions.

**Which pass should have caught it.** The pre-publish check confirmed the target existed — it had
been added precisely because an earlier validator found the entry naming a target that did not
exist. Existence was checked; **runnability was not**. The reviews after it read the prose for
accuracy against the code, and the sentence is accurate about the scored run, which is what the
paragraph is mostly about. Nobody ran the command as a reader with no API key.

The generalisable lesson, and the reason this is written down rather than quietly fixed: *a
command named in an entry has to be executed under a reader's conditions, not the author's.* The
author always has the key.

---

## C2 — `wilson_interval` and `mde_two_proportion` were private to one entry

**Found:** same pass. Not an error in any published number.

Both were private helpers inside `reasoning_ablation.py`. Experiment 005 needs the same two
functions to size its identity coverage, and two copies of an estimator is how two entries come to
quote subtly different numbers from what reads as the same method.

Both are now in `stats.py`, unchanged in behaviour. The move was verified by recomputing every
figure in this artifact that depends on them and requiring exact equality: three Wilson intervals
and the MDE, all identical.
