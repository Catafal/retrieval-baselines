"""
The four query properties pre-registered in protocols/002-amendment-1-dense.md
section 7 — and only these four. Analysing whichever property turns out to
correlate with a win after the fact is how a fishing expedition manufactures a
finding; the amendment fixes the set before any dense number exists.

  1. query length in tokens
  2. max and mean IDF across the query's terms
  3. Jaccard overlap between query and gold-document token sets, averaged
     across that query's gold documents
  4. gold-document count

All four are computed with the SAME tokenizer the lexical scorer scores
with (rb.experiments.ladder.retrievers.lexical._tokenize), per the amendment's
"computed with the scorer's own tokenizer" — using a different tokenizer would
describe query rarity and overlap against a vocabulary neither the lexical nor
the dense rung actually saw.
"""

from rb.experiments.ladder.retrievers.lexical import LexicalIndex, _idf, _tokenize


def query_length(query: str) -> int:
    """Token count, using the scorer's tokenizer. Not deduplicated — length is
    about how much text was written, not how many distinct terms it contains."""
    return len(_tokenize(query))


def query_idf(query: str, index: LexicalIndex) -> dict:
    """
    Max and mean IDF across the query's DISTINCT terms.

    Distinct, not every token, because IDF is a per-term-TYPE value (the same
    term repeated twice has one IDF, not two) and this matches the convention
    the lexical scorer itself uses for its outer sum
    (`sorted(set(_tokenize(query)))` in LexicalRetriever.retrieve).

    A term absent from the corpus vocabulary gets document frequency 0, which
    `_idf` handles as the maximally rare case rather than raising — an
    out-of-vocabulary query term is real information about that query (nothing
    in the corpus uses this word), not a computation to skip.
    """
    terms = set(_tokenize(query))
    if not terms:
        return {"max_idf": 0.0, "mean_idf": 0.0}
    idfs = [
        _idf(int(index.df[index.vocab[t]]) if t in index.vocab else 0, index.n)
        for t in terms
    ]
    return {"max_idf": max(idfs), "mean_idf": sum(idfs) / len(idfs)}


def query_gold_jaccard(query: str, gold_ids: list[str], corpus: dict[str, str]) -> float:
    """
    Jaccard(query tokens, gold-document tokens), averaged across the query's
    gold documents — amendment section 7 item 3. Jaccard operates on sets, so
    both sides are deduplicated token sets, unlike query_length above.

    A gold id missing from `corpus` (should not happen after gold_presence
    passes, but this function does not assume that control has run) is
    treated as an empty document rather than raising, so a property
    computation never becomes the reason a run halts — that is what the
    dedicated controls in rb.controls are for.
    """
    if not gold_ids:
        return 0.0
    q_terms = set(_tokenize(query))
    overlaps = []
    for gid in gold_ids:
        g_terms = set(_tokenize(corpus.get(gid, "")))
        union = q_terms | g_terms
        overlaps.append(len(q_terms & g_terms) / len(union) if union else 0.0)
    return sum(overlaps) / len(overlaps)


def gold_count(gold_ids: list[str]) -> int:
    """Number of gold documents for this query — separates single-hop from
    multi-hop queries, per amendment section 7 item 4."""
    return len(gold_ids)


def compute_query_properties(
    query: str, gold_ids: list[str], corpus: dict[str, str], index: LexicalIndex
) -> dict:
    """All four properties for one query, as a flat dict — the shape
    rb.experiments.ladder.analysis bins on."""
    idf = query_idf(query, index)
    return {
        "query_length": query_length(query),
        "max_idf": idf["max_idf"],
        "mean_idf": idf["mean_idf"],
        "gold_jaccard": query_gold_jaccard(query, gold_ids, corpus),
        "gold_count": gold_count(gold_ids),
    }
