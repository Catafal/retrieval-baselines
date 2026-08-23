"""
The per-query win/loss analysis, protocols/002-amendment-1-dense.md section 7:
for every scored query, which arm won — dense or full BM25 — and does that
track the four pre-registered query properties (query_properties.py) and no
others.

Bins are reported with an interval on the paired per-query difference, not
just a win count, by reusing rb.stats.paired_bootstrap on the subset of
queries in each bin — the same procedure the lexical factorial's adjacent-rung
comparisons use, applied to a bin instead of a whole corpus. A bin with too
few queries is reported with `"sparse": true` and never dropped, per the
amendment: "Bins with too few queries are reported as sparse, never dropped."
"""

from rb.stats import paired_bootstrap

# Below this many queries, a bin's win rate and interval are noise-dominated
# and must be flagged rather than read as a finding. Not pre-registered by the
# amendment (which fixes the four properties, not a bin-size cutoff) — chosen
# here as a documented, conservative threshold: paired_bootstrap's own interval
# already communicates the uncertainty, "sparse" exists so a reader does not
# have to infer that from n_queries themselves.
SPARSE_THRESHOLD = 5

# Continuous properties are split into this many equal-COUNT bins (quartiles
# by rank, not equal-width ranges by value) so a skewed distribution — IDF and
# Jaccard both are — does not produce three empty bins and one bin holding
# everything.
N_BINS = 4

# gold_count is discrete and typically takes only a few values on BEIR
# (single-hop vs multi-hop), so it is binned by exact value rather than by
# quantile — quantile-splitting an already-discrete variable would produce
# arbitrary boundaries that split what should be one bin (e.g. "2 gold docs")
# across two.
DISCRETE_PROPERTIES = {"gold_count"}

PROPERTY_NAMES = ("query_length", "max_idf", "mean_idf", "gold_jaccard", "gold_count")


def _bin_by_property(values: dict[str, float], property_name: str) -> list[dict]:
    """qid -> property value, split into bins. Returns a list of
    {"label", "qids"} in a stable order (ascending by value)."""
    if property_name in DISCRETE_PROPERTIES:
        distinct = sorted(set(values.values()))
        return [
            {"label": str(v), "qids": sorted(q for q, val in values.items() if val == v)}
            for v in distinct
        ]

    # Equal-count quantile bins: sort queries by (value, qid) — the qid
    # tie-break is what makes this deterministic when many queries share a
    # value, same tie-break convention as every ranking in this repo.
    ordered = sorted(values, key=lambda q: (values[q], q))
    n = len(ordered)
    bins = []
    for i in range(N_BINS):
        lo, hi = i * n // N_BINS, (i + 1) * n // N_BINS
        group = ordered[lo:hi]
        if not group:
            continue  # fewer distinct queries than N_BINS: skip, do not fabricate an empty bin
        group_values = [values[q] for q in group]
        bins.append({"label": f"[{min(group_values):.4g}, {max(group_values):.4g}]", "qids": sorted(group)})
    return bins


def win_loss_by_property(
    dense_ndcg: dict[str, float],
    bm25_ndcg: dict[str, float],
    properties: dict[str, dict],
) -> dict:
    """
    dense vs full BM25, binned by each of the four pre-registered properties.

    `dense_ndcg` / `bm25_ndcg`: qid -> nDCG@10, same query set, query-aligned
    by key rather than by list position (unlike paired_bootstrap's own
    list-position pairing) so a caller cannot accidentally misalign the two
    by passing qids in a different order to each.
    `properties`: qid -> the four-field dict query_properties.compute_query_properties
    returns.

    Returns {property_name: [bin, ...], "per_query_diff": {qid: dense - bm25}}.
    The last field is the raw paired per-query difference the amendment also
    asks for, independent of any binning — a reader who does not trust the
    bin boundaries chosen here can recompute their own from this.
    """
    qids = sorted(dense_ndcg)
    if sorted(bm25_ndcg) != qids:
        raise ValueError("win_loss_by_property requires dense_ndcg and bm25_ndcg over the identical query set")

    result: dict[str, object] = {}
    for prop_name in PROPERTY_NAMES:
        prop_values = {q: properties[q][prop_name] for q in qids}
        bins = _bin_by_property(prop_values, prop_name)
        bin_reports = []
        for b in bins:
            bqids = b["qids"]
            n = len(bqids)
            dense_scores = [dense_ndcg[q] for q in bqids]
            bm25_scores = [bm25_ndcg[q] for q in bqids]
            wins = sum(1 for q in bqids if dense_ndcg[q] > bm25_ndcg[q])
            losses = sum(1 for q in bqids if dense_ndcg[q] < bm25_ndcg[q])
            ties = n - wins - losses
            boot = paired_bootstrap(dense_scores, bm25_scores)
            bin_reports.append({
                "bin": b["label"],
                "n_queries": n,
                "sparse": n < SPARSE_THRESHOLD,
                "dense_wins": wins,
                "bm25_wins": losses,
                "ties": ties,
                "dense_win_rate": round(wins / n, 4),
                "mean_diff": boot["mean_diff"],
                "ci95": boot["ci95"],
                "p_value": boot["p_value"],
            })
        result[prop_name] = bin_reports

    result["per_query_diff"] = {q: dense_ndcg[q] - bm25_ndcg[q] for q in qids}

    # WHY THESE P-VALUES CARRY NO HOLM DECISION, STATED IN THE ARTIFACT ITSELF.
    # protocols/002-ladder.md section 5 registers the Holm family as "the adjacent-rung
    # comparisons made within each corpus". That family is the `comparisons` /
    # `adjacent_rung_comparisons` arrays, and every member of it gets a `holm_significant`
    # key. The bins below are a different thing: amendment-1 section 7 registers them as
    # win rate "within bins of each property, with intervals", and the entry reads only
    # `mean_diff` and `n_queries` out of them — no per-bin accept/reject is ever asserted,
    # so there is no family-wise error rate to control and applying Holm here would be a
    # correction in search of a claim.
    #
    # But a reader opening this file sees 113 p-values sitting beside 24 that ARE gated,
    # with nothing in the data distinguishing them. A review council split precisely on
    # that: one seat derived that no correction is owed, another argued the absence of any
    # in-file signal is itself the defect. This field is the resolution — the numbers stay
    # uncorrected, and say so where they live rather than only in prose the reader may not
    # have.
    result["multiplicity"] = {
        "corrected": False,
        "method": None,
        "n_p_values": sum(len(v) for k, v in result.items()
                          if k in PROPERTY_NAMES and isinstance(v, list)),
        "note": (
            "Exploratory. These per-bin p-values are UNCORRECTED for multiplicity and no "
            "published claim rests on any one of them; the entry reads only mean_diff and "
            "n_queries from these bins. The pre-registered Holm family is the adjacent-rung "
            "/ dense-hybrid comparisons array, whose members each carry holm_significant. "
            "Do not read a single bin here as a significance result."
        ),
    }
    return result
