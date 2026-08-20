"""
The bridge-entity query class — protocols/003-graph-arm.md section 4.

WHAT THIS MEASURES. 002 found dense retrieval losing to full BM25 on HotpotQA in every
query-overlap quartile and named a mechanism it could not isolate: the entity bridging
the two gold documents is absent from the question text, so neither arm has anything to
match on. This module turns "absent from the question text" into a number computed from
data, because the protocol's own assessment is that the entry lives or dies on that
definition being mechanical rather than a judgement made after seeing results.

THE LEVER IS THE CORPUS'S OWN TITLES. For HotpotQA a document title IS an entity — these
are Wikipedia article titles — and a query's qrels name exactly the documents that must
be chained. So the class needs no model, no annotation and no human: count how many of a
query's gold-document titles appear in the query text.

  coverage 0  neither gold title named        1,821 queries  hardest, no anchor
  coverage 1  one named, one must be reached  3,634 queries  the classic bridge
  coverage 2  both named                      1,950 queries  comparison question

Coverage 2 is the experiment's built-in negative control: nothing needs traversing, so a
graph arm that wins there is a bug rather than a triumph.

WHY TWO DEFINITIONS, BOTH FROZEN. Wikipedia titles carry disambiguating parentheticals —
"Kiss and Tell (1945 film)" — in 14.0% of gold titles and never in a natural question.
Stripping one or not RECLASSIFIES 1,138 queries, 15.4% of the corpus, which is larger
than any plausible retrieval effect. A formatting decision nobody would think to write
down would otherwise decide the finding. So both are registered before any run and both
are reported whatever they show:

  PRIMARY      strip one trailing parenthetical, then match
  SENSITIVITY  exact title, unstripped

CORROBORATION, NOT VALIDATION. HotpotQA ships a human-assigned `type` per question. The
primary class agrees with it on 6,938/7,405 = 93.7%. Both labels derive from the same
gold documents, so this is agreement between two overlapping operationalisations and is
reported as such, never as independent evidence. Coverage stays primary and `type` is a
robustness filter only: 465 questions are type `bridge` yet coverage 2, and for those
nothing is hidden for traversal to find. The claim under test is about surface form.
"""

import logging
import re

from rb.experiments.ladder.retrievers.lexical import _tokenize

log = logging.getLogger(__name__)

# One TRAILING parenthetical only. Not every parenthetical anywhere in the title:
# "Sunday Bloody Sunday (song)" is a disambiguated entity, but a title that genuinely
# contains brackets mid-string is part of the name and removing it would silently
# rewrite the entity. Anchored to the end for that reason.
_DISAMBIGUATOR = re.compile(r"\s*\([^)]*\)\s*$")

PRIMARY = "stripped"
SENSITIVITY = "exact"
DEFINITIONS = (PRIMARY, SENSITIVITY)

# Measured before tagging and frozen in protocols/003-graph-arm.md section 4. Checked
# against, never derived from a run: a distribution that recomputes itself from the
# outcome cannot fail, which is the whole reason the class is registered in advance.
FROZEN_DISTRIBUTION = {
    PRIMARY: {0: 1821, 1: 3634, 2: 1950},
    SENSITIVITY: {0: 2529, 1: 3418, 2: 1458},
}
FROZEN_RECLASSIFIED = 1138
FROZEN_QUERIES = 7405


def strip_disambiguator(title: str) -> str:
    """Remove one trailing "(...)" from a title. The PRIMARY definition's only step."""
    return _DISAMBIGUATOR.sub("", title).strip()


def normalise(text: str) -> list[str]:
    """
    Tokens, using the SCORER's own tokenizer rather than a private one.

    002's query-property module fixed this convention — properties are computed with the
    tokenizer the lexical rung actually scores with, so a class describes the same
    vocabulary the arms saw. Checked before adopting it here: the scorer's tokenizer and
    the normalisation the frozen counts were measured with are equivalent on this data
    (both lowercase and split on non-alphanumerics), reproducing 1,821/3,634/1,950
    exactly with zero differing tokenizations across 4,000 queries. So following the
    convention costs nothing and the registered numbers still hold.
    """
    return _tokenize(text)


def title_in_query(title: str, query_tokens: list[str], definition: str = PRIMARY) -> bool:
    """
    Does this title occur in the query as a CONTIGUOUS token run?

    Contiguous, not as a bag of words: "Chris Evans" appearing as the separate words
    "Chris" and "Evans" scattered through a sentence is not the entity being named, and
    a bag-of-words test would count almost any multi-word title against a long question.
    An empty title never counts — it cannot be "named".
    """
    if definition not in DEFINITIONS:
        raise ValueError(f"unknown definition {definition!r}; expected one of {DEFINITIONS}")
    text = strip_disambiguator(title) if definition == PRIMARY else title
    needle = normalise(text)
    if not needle:
        return False
    span = len(needle)
    return any(query_tokens[i:i + span] == needle
               for i in range(len(query_tokens) - span + 1))


def coverage(query: str, gold_titles: list[str], definition: str = PRIMARY) -> int:
    """How many of this query's gold-document titles are named in the query text."""
    tokens = normalise(query)
    return sum(title_in_query(t, tokens, definition) for t in gold_titles)


def coverage_all(queries: dict[str, str], gold_titles: dict[str, list[str]],
                 definition: str = PRIMARY) -> dict[str, int]:
    """query_id -> coverage, over every query that has gold titles."""
    result = {qid: coverage(queries[qid], gold_titles[qid], definition)
              for qid in sorted(gold_titles) if qid in queries}
    log.info("coverage[%s]: %d queries classified", definition, len(result))
    return result


def distribution(classes: dict[str, int]) -> dict[int, int]:
    """Bin counts, with every bin present even when empty, so a missing class is visible
    as a zero rather than as an absent key a caller might not notice."""
    return {c: sum(1 for v in classes.values() if v == c) for c in (0, 1, 2)}


def reclassified(primary: dict[str, int], sensitivity: dict[str, int]) -> int:
    """
    Queries the two definitions disagree about — the 15.4% the protocol warns is larger
    than any effect the experiment could measure.
    """
    shared = set(primary) & set(sensitivity)
    n = sum(1 for q in shared if primary[q] != sensitivity[q])
    log.info("definitions disagree on %d/%d queries", n, len(shared))
    return n


def agreement_with_type(classes: dict[str, int], types: dict[str, str]) -> dict:
    """
    Cross-tabulate the mechanical class against HotpotQA's human-assigned `type`.

    Reported as CORROBORATION between two overlapping operationalisations, never as
    external validation: both derive from the same gold-document set. `agreed` treats
    coverage <= 1 as bridge-like and coverage 2 as comparison-like, which is the mapping
    the protocol's cross-tab uses.
    """
    shared = sorted(set(classes) & set(types))
    table: dict[tuple[str, int], int] = {}
    for q in shared:
        table[(types[q], classes[q])] = table.get((types[q], classes[q]), 0) + 1
    agreed = sum(1 for q in shared
                 if (types[q] == "bridge" and classes[q] <= 1)
                 or (types[q] == "comparison" and classes[q] == 2))
    return {
        "queries": len(shared),
        "agreed": agreed,
        "rate": round(agreed / len(shared), 4) if shared else 0.0,
        "table": {f"{t}|{c}": n for (t, c), n in sorted(table.items())},
    }
