# Protocol 005 amendment 1 — the alias-affected population was defined too narrowly

**Status: frozen on tagging as `protocol-005-identity-amendment-1`.** Written after the first
coverage cell was computed and before any other cell was accepted. No retrieval number exists at
the time of writing, on either identity, on either corpus.

## What section 6 said

> **C5** the **alias-affected query subset**: queries with at least one entity resolving through
> an alias

Implemented literally: a query counts when one of its own entity strings is a registry key.

## Why that is wrong

It counts one of the two ways typed identity can change a query's result, and not the larger one.

Merging is symmetric in effect but not in form. The registry maps *alias → canonical*, and a
query is helped whenever the document it needs becomes reachable, which happens in two distinct
situations:

- **(a) the query names an alias.** "Cleveland State" resolves to `cleveland state university`
  and now matches a document keyed under the canonical. This is what C5 counted.
- **(b) the query names the canonical, and a document named an alias.** The query entity does not
  resolve — it is already canonical, so it is not a registry key — but the document's node has
  merged into it, and the document is now reachable where it was not.

Case (b) never touches a registry key on the query side, so the literal reading of C5 cannot see
it. On the first cell measured, 2Wiki under GLM, case (a) reached 0.5% of queries. If (b) is the
common case — and 2Wiki's questions are written from article titles, so its queries should skew
canonical — then C5 as written understates the testable population by whatever (b) contributes,
and understates it silently.

**A null computed on that subset would be an artefact of the measurement, not a property of the
data.** It would say "typed identity cannot be tested here" when the truth was "my definition of
affected could not see most of the effect". That is the failure this sequence keeps having, and
the reason to write an amendment rather than a comment.

## The corrected definition

A query is **alias-affected** when at least one of its entities reaches a different set of
documents under typed identity than under string identity.

For query entity `e`, with `s = normalise(e)` and `t = link(e)`:

    affected(e)  ⇔  docs_typed[t] ≠ docs_string[s]

where `docs_string` maps each node key to the documents containing it under 003/004's exact-string
identity, and `docs_typed` the same under the registry. This is the mechanism's own definition —
the walk starts from seeds, and what a seed can reach is exactly what identity changed — and it
subsumes both (a) and (b) rather than privileging one.

## What is reported

Both, and the narrow one is kept rather than replaced, so the correction is visible in the
artifact rather than only in this file:

- `alias_affected_narrow` — the original C5, queries naming an alias
- `alias_affected` — the corrected C5, queries whose reachable set changes
- `mde_at_80_power_on_affected` — now computed on the corrected subset, since that is the
  population Stage 1 decomposes over

## What this does not change

Sections 3, 4, 5 and 7 stand unamended. The identity source, the eight construction rules, the
seam and the permitted outcomes are untouched. The registry built under this amendment is the
same registry, byte for byte; only the population over which coverage is reported changes.

No prediction is registered here. That is still Stage 1's, and still unwritten.
