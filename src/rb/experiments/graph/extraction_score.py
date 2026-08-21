"""
Scoring for the extraction diagnostic — protocols/003-graph-arm.md §8.2.

COMMITTED BEFORE THE GOLD SET EXISTS. Every record in the sample still reads
`annotated: false` as this lands. Fixing the procedure before the data is the same
discipline the protocol applies everywhere else: a measurement chosen after seeing the
numbers is not a measurement.

A DIAGNOSTIC, NOT A GATE. No threshold on precision or recall is pre-registered, because
none could be justified in advance — there is no published figure for spaCy NER against
Wikipedia intros under this annotation scheme. A number with no pre-committable decision
rule is a measurement in search of one, so it is reported with its limitations and it
gates nothing. What gates the arm is `graph_connectivity` below, which needs no gold set.
"""

import random
import re
import statistics

from rb.experiments.graph.entity_types import WHITELIST, filter_entities

# Normalisation, fixed here rather than left to the annotator (rule card item 9). Case and
# punctuation only: no stemming, no canonicalisation, no abbreviation expansion. "U.S." and
# "United States" stay DIFFERENT strings, because the graph would treat them as different
# nodes and the diagnostic must measure what the graph will actually do.
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalise(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def score_passage(gold: list[str], predicted: list[str]) -> dict:
    """
    Set-based true/false positives and negatives for one passage.

    SETS, not multisets: the graph deduplicates nodes by string, so a repeated mention is
    one node. This also removes the title echo — measured at a median of 2 occurrences per
    passage — from the arithmetic entirely, rather than leaving it to inflate recall.
    """
    g = {normalise(x) for x in gold if normalise(x)}
    p = {normalise(x) for x in predicted if normalise(x)}
    return {"tp": len(g & p), "fp": len(p - g), "fn": len(g - p), "gold": len(g), "pred": len(p)}


def micro(counts: list[dict]) -> dict:
    """Micro-averaged P/R/F1 — pooled counts, so passages with more entities weigh more,
    which matches how the graph is actually built (per entity, not per document)."""
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def bootstrap_ci(counts: list[dict], metric: str = "precision",
                 b: int = 10_000, seed: int = 20260820) -> dict:
    """
    Percentile CI, resampling PASSAGES rather than entities.

    The independent sampling unit is the passage: the sample was drawn as 100 passages, and
    entities inside one passage share extraction outcomes — a sentence-boundary or
    tokenisation failure takes several of them together. Resampling entities would treat
    ~1,226 correlated observations as independent and report an interval roughly two to
    three times too narrow. With a design effect at ~12.3 entities per passage, the honest
    interval is around +/-0.03 to +/-0.06 rather than the +/-0.022 a per-entity computation
    would suggest.
    """
    rng = random.Random(seed)
    n = len(counts)
    draws = []
    for _ in range(b):
        sample = [counts[rng.randrange(n)] for _ in range(n)]
        draws.append(micro(sample)[metric])
    draws.sort()
    return {"metric": metric, "point": micro(counts)[metric],
            "ci95": [round(draws[int(0.025 * b)], 4), round(draws[int(0.975 * b)], 4)],
            "resamples": b, "seed": seed, "unit": "passage"}


def doc_coverage(entities_by_doc: dict[str, list[str]]) -> tuple[int, int]:
    """(documents, documents with at least one entity).

    Canonical because two callers need it — the §8.3 gate below and run_controls.graph_summary
    — and they had independent copies. Two implementations of one count is how the numbers in
    two published artifacts drift apart with nothing noticing, which is the same reasoning that
    collapsed the two Holm implementations in rb.stats.
    """
    return len(entities_by_doc), sum(1 for e in entities_by_doc.values() if e)


def graph_connectivity(entities_by_doc: dict[str, list[str]]) -> dict:
    """
    The GATE — computed from the extractor's own output, no gold set involved.

    A graph arm can only retrieve by propagating along shared nodes. If almost no entity
    string occurs in two documents, the graph is a pile of disconnected islands and the arm
    cannot work for reasons that have nothing to do with extraction accuracy. That is a
    precondition rather than a quality judgement, which is why it can gate when precision
    cannot.

    Returns the two observed quantities; the threshold comes from `permutation_null` below,
    not from a number picked by hand.
    """
    per_doc = {d: {normalise(e) for e in ents if normalise(e)} for d, ents in entities_by_doc.items()}
    counts: dict[str, int] = {}
    for ents in per_doc.values():
        for e in ents:
            counts[e] = counts.get(e, 0) + 1
    shared = sum(1 for e, n in counts.items() if n >= 2)
    documents, populated = doc_coverage(per_doc)
    return {
        "documents": documents,
        "documents_with_an_entity": populated,
        "distinct_entities": len(counts),
        "entities_in_2plus_documents": shared,
        "shared_fraction": round(shared / len(counts), 4) if counts else 0.0,
    }


def bridge_reachability(entities_by_doc: dict[str, list[str]],
                        gold_pairs: list[tuple[str, str]],
                        b: int = 1000, seed: int = 20260820) -> dict:
    """
    The GATE. No gold annotation involved; it needs only the extractor's output and qrels.

    WHY NOT A PERMUTED NULL OVER ENTITY STRINGS. That was the first design and it was
    tried and discarded, on evidence rather than taste: shuffling entity strings across
    documents preserves how often each string occurs, and "is this entity in two or more
    documents" depends on nothing else. The null came out equal to the observed value on a
    graph with obvious structure — a single hub entity shared by all thirty documents — so
    it had no power to reject anything. A test written for it failed, which is how this was
    caught before it reached the protocol.

    WHAT REPLACES IT. The arm's entire mechanism is reaching a document that the query does
    not name by traversing from one that it does. So the precondition is simply: do the two
    gold documents of a query share an entity at all? If they never do, no amount of
    propagation reaches the second one, and the arm cannot work for structural reasons that
    have nothing to do with extraction accuracy.

    THE NULL THAT DOES HAVE POWER. Compare the gold pairs against RANDOM document pairs.
    Gold pairs are the ones a bridge question chains together; random pairs are not. If
    gold pairs share entities no more often than random pairs do, the extracted graph
    carries no bridging signal, and that is decidable in advance with no threshold invented
    by anybody.
    """
    rng = random.Random(seed)
    per_doc = {d: {normalise(e) for e in ents if normalise(e)}
               for d, ents in entities_by_doc.items()}

    def shares(a: str, c: str) -> bool:
        return bool(per_doc.get(a) and per_doc.get(c) and per_doc[a] & per_doc[c])

    pairs = [(a, c) for a, c in gold_pairs if a in per_doc and c in per_doc]
    observed = sum(shares(a, c) for a, c in pairs) / len(pairs) if pairs else 0.0

    docs = sorted(per_doc)
    draws = []
    if len(docs) >= 2 and pairs:
        for _ in range(b):
            hits = 0
            for _ in pairs:
                a, c = rng.sample(docs, 2)
                hits += shares(a, c)
            draws.append(hits / len(pairs))
        draws.sort()
    null_hi = draws[int(0.95 * b)] if draws else 0.0
    return {
        "gold_pairs_scored": len(pairs),
        "observed_share_rate": round(observed, 4),
        "random_pair_rate_p95": round(null_hi, 4),
        "random_pair_median": round(statistics.median(draws), 4) if draws else 0.0,
        "resamples": b,
        "seed": seed,
        "passed": bool(pairs) and observed > null_hi,
    }


def score(gold_by_doc: dict[str, list[str]], predicted_by_doc: dict[str, list[tuple[str, str]]],
          kept=WHITELIST) -> dict:
    """
    The full diagnostic.

    `predicted_by_doc` carries (text, label) pairs and is filtered to the whitelist HERE,
    on the same terms the gold set was annotated under. Filtering only one side would turn
    every correctly-extracted DATE into a false positive and collapse precision for a
    reason that has nothing to do with extraction quality — the single easiest way to make
    this number wrong, so it lives in one place.
    """
    docs = sorted(gold_by_doc)
    counts = [
        score_passage(gold_by_doc[d], [t for t, _ in filter_entities(predicted_by_doc.get(d, []), kept)])
        for d in docs
    ]
    return {
        "passages": len(docs),
        "micro": micro(counts),
        "precision_ci": bootstrap_ci(counts, "precision"),
        "recall_ci": bootstrap_ci(counts, "recall"),
        "note": (
            "Diagnostic, not a gate. Gold annotated by one rater who is also the author of the "
            "extractor; no inter-annotator agreement exists. No threshold was pre-registered "
            "because none could be justified in advance."
        ),
    }
