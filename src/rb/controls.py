"""
Controls. These exist to catch a broken harness before it produces a publishable number.

The retracted entry that preceded this repo had no controls at all, which is how
it shipped figures nobody could reproduce. Each control below fails loudly.
"""

from pathlib import Path

from rb.grep_baseline import rank, search


def gold_presence(corpus: dict[str, str], qrels: dict[str, dict[str, int]]) -> dict:
    """
    Every judged document must actually be in the indexed corpus.

    A missing gold document depresses recall for reasons that have nothing to do
    with the retriever — it would look exactly like a finding.
    """
    gold_ids = {d for g in qrels.values() for d in g}
    missing = gold_ids - set(corpus)
    return {
        "gold_documents": len(gold_ids),
        "missing_from_corpus": len(missing),
        "examples": sorted(missing)[:5],
        "passed": not missing,
    }


def empty_query(corpus_path: Path, doc_ids: list[str]) -> dict:
    """
    A query with no surviving terms must retrieve nothing and score zero.

    Catches a scorer that credits gold documents it never actually retrieved.
    """
    hits = search([], corpus_path)
    results = rank(hits, doc_ids)
    return {"retrieved": len(results), "passed": len(results) == 0}
