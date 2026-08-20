# Annotation rule card v1 — Experiment 003 §8.2

**Frozen before passage 1.** Committed before any passage is annotated and before spaCy is
installed. If a rule changes mid-session, the change is recorded with the passage index at which
it took effect, and passages before that index are either re-annotated under the new rule or
reported separately. Rules are never revised silently and never backfilled.

**Version 1.** Any later version supersedes this one by number, and both stay in the repository.

## What is being produced

For each of the 100 passages in `results/003/extraction-sample.jsonl`, the **set of entity
strings** that appear in its `text`.

**A set, not a list.** Duplicate mentions of the same string collapse to one member. This is not
a simplification for the annotator's benefit: the graph deduplicates nodes by string, so a set is
the correct representation of the object being measured. It also removes the title-echo problem
below from the scoring arithmetic entirely.

**Strings, not character offsets.** The downstream consumer matches query entities to graph nodes
by string. Offsets would measure boundary precision on an axis nothing consumes, and hand-produced
offsets for 100 passages without annotation tooling is a 3–6 hour job rather than a 1–2 hour one.

**No entity types.** Types are needed for the whitelist, but assigning them by hand doubles the
per-decision cost. The whitelist is applied to the *extractor's* output at scoring time
(`entity_types.filter_entities`); the annotation records what a whitelisted-type entity would be,
per the inclusion rules below.

## What counts — inclusion rules

**Include** named things that could plausibly be named in a question and shared between documents:
people, organisations, countries, cities, regions, buildings, nationalities and religious or
political groups, products, events, works of art, laws, languages.

**Exclude** measures and quantities: dates, times, percentages, money, quantities, ordinals,
cardinals. *"2010"*, *"28,781"*, *"third-class"* are not annotated.

Grounding, so this is a decision rather than a taste: of the 13,783 gold-document titles in
HotpotQA — the bridge entities this experiment is about — 2 (0.01%) are purely numeric.

## Edge cases, decided now

These are the cases that recur in Wikipedia intros. Each is decided here so passage 97 is judged
as passage 3 was.

1. **The title is prepended to the text and usually repeats.** Measured: the title string occurs a
   median of 2 times per passage, and 69 of 100 passages contain it at least twice. Annotate it
   **once** — the set collapses it. Do not treat the echo as a separate entity.

2. **Year-initial event names.** *"1936 Summer Olympics"* is **one entity**, annotated whole. The
   year is part of the name, not a date beside it. This rule exists because 451 of 13,783 gold
   titles (3.27%) begin with a digit — *1954 FIFA World Cup*, *2002 Commonwealth Games* — and
   because EVENT is one of spaCy's weakest published types (F 0.406), so this is exactly where a
   disagreement will be real rather than clerical.

3. **Nested places.** Annotate the **most specific named span**, plus any other name that stands
   alone. *"the U.S. state of Missouri"* → `Missouri`, and `U.S.` separately if it appears as its
   own token. Do not annotate the container phrase *"U.S. state of Missouri"*.

4. **Comma-chained place names.** *"Marion County, Missouri"* → **two** entities, `Marion County`
   and `Missouri`. Do not merge into one string.

5. **Parenthetical glosses, pronunciations, translations.** *"(; lit. \"The Cat: Eyes that See
   Death\")"* → skip entirely, including the quoted translation inside it.

6. **Quoted work titles.** Include as one entity with the quote marks stripped, using the surface
   form as written.

7. **Foreign-language romanisations** appearing as a gloss beside an English name → skip. If the
   only form given is non-English, annotate it as written.

8. **People with attached roles.** *"directed by Byun Seung-wook"* → `Byun Seung-wook` only. The
   role is not part of the name.

9. **Surface form is preserved exactly as it appears.** Do not canonicalise, do not expand
   abbreviations, do not fix case. *"U.S."* is annotated `U.S.`, never `United States`. If both
   forms appear in one passage, both are set members. Normalisation happens once, at scoring time,
   under a rule fixed in code — not in the annotator's head, and not differently on different days.

10. **Uncertain cases.** If a decision takes more than a few seconds, include it and move on.
    Systematic over-inclusion is visible in precision and is recoverable; a hesitation rule that
    varies with fatigue is not.

## Session procedure

- Annotate in the **seed-drawn order**, not sorted by length. Sorting would align fatigue with a
  passage property and turn drift into a systematic bias rather than noise.
- **Two blocks of 50** with a real break between them. Start and end times are recorded per block
  by the tool.
- **Blind re-annotation of 15 passages in a later sitting**, without consulting the first pass,
  published as intra-annotator agreement.

  What this measures, stated plainly: **self-consistency, not correctness.** It does not answer
  the single-rater objection — nothing available here does — and it is not reported as if it did.

## What is recorded about the session

The tool writes, per record: the annotation, a UTC timestamp, and this rule card's version. It
saves after every record and writes atomically, so an interrupted session resumes rather than
restarts.

## What this annotation is, and is not

It produces a **reported diagnostic**, not a gate. Precision and recall against this set say
whether extraction is obviously broken. They do not say whether the graph will retrieve well, and
no threshold on them was pre-committed — because none could be justified in advance, which is the
reason §8.2 no longer gates anything.
