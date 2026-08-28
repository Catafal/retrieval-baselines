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
gates nothing. What gates the arm is `bridge_reachability` below, which needs no gold set.
"""

import random
import re
import statistics

from rb.experiments.graph.entity_types import WHITELIST, filter_entities
from rb.stats import percentile_ci, upper_percentile

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
    lo, hi = percentile_ci(draws)
    return {"metric": metric, "point": micro(counts)[metric],
            "ci95": [round(lo, 4), round(hi, 4)],
            "resamples": b, "seed": seed, "unit": "passage"}


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
    # ONE-SIDED, and this one decides rather than reports: `passed` below is
    # `observed > null_hi`. Written by hand as `draws[int(0.95 * b)]` it left only 4.9% of draws
    # above the threshold instead of 5%, so the gate was fractionally easier to pass than it
    # claimed. `upper_percentile` derives the index from the tail being cut. See NB-26 D3.
    null_hi = upper_percentile(draws) if draws else 0.0
    return {
        "gold_pairs_scored": len(pairs),
        "observed_share_rate": round(observed, 4),
        "random_pair_rate_p95": round(null_hi, 4),
        "random_pair_median": round(statistics.median(draws), 4) if draws else 0.0,
        "resamples": b,
        "seed": seed,
        "passed": bool(pairs) and observed > null_hi,
    }


def provenance(rows: list[dict]) -> dict:
    """
    Who produced the reference set, DERIVED from the sample rather than asserted.

    WHY DERIVED. The hardcoded note this replaces said "gold annotated by one rater who is also the
    author of the extractor; no inter-annotator agreement exists" and was published verbatim inside
    `results/003/extraction-diagnostic.json`. By then it was false in every clause: the protocol's
    second revision replaced the single annotator with three independent language-model raters and
    majority adjudication, `extraction-sample.jsonl` records `annotator: llm-panel-3x-majority` with
    a per-passage `rater_jaccard`, and `annotation-agreement.json` publishes mean pairwise Jaccard.
    A reader diffing two committed files in this repository would have found them contradicting each
    other.

    A sentence about the data, hardcoded next to the data, drifts from the data. So it is computed.
    """
    annotators = sorted({r.get("annotator") for r in rows if r.get("annotator")})
    jac = [r["rater_jaccard"] for r in rows if isinstance(r.get("rater_jaccard"), (int, float))]
    return {
        "annotator": annotators[0] if len(annotators) == 1 else annotators,
        "rule_card": sorted({r.get("rule_card") for r in rows if r.get("rule_card")}),
        "passages": len(rows),
        "passages_with_rater_agreement": len(jac),
        "mean_pairwise_jaccard": round(sum(jac) / len(jac), 4) if jac else None,
    }


def score(gold_by_doc: dict[str, list[str]], predicted_by_doc: dict[str, list[tuple[str, str]]],
          kept=WHITELIST, rows: list[dict] | None = None) -> dict:
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
        # PER-PASSAGE COUNTS, EXPOSED. They were computed here and discarded, which made any
        # paired interval on a difference between two extractors unreproducible from the
        # artifact — a reviewer of experiment 004 had to simulate one because of this line.
        "counts": [{"doc_id": d, **c} for d, c in zip(docs, counts)],
        "micro": micro(counts),
        "precision_ci": bootstrap_ci(counts, "precision"),
        "recall_ci": bootstrap_ci(counts, "recall"),
        "note": (
            "Diagnostic, not a gate. No threshold was pre-registered because none could be "
            "justified in advance. The reference set is MODEL-ANNOTATED, not hand-annotated: "
            "three independent language-model raters working alone from a frozen rule card, an "
            "entity kept when at least two of three listed it, adjudicated in deterministic code. "
            "It therefore inherits model biases about entity boundaries and is not independent "
            "human judgement; discount it accordingly. What it does buy is real inter-annotator "
            "agreement, reported below and in results/003/annotation-agreement.json."
        ),
        # Derived from the sample, never asserted. See provenance().
        "reference_set": provenance(rows) if rows else None,
    }
