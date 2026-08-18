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


def bm25_closure(our_ndcg: float, anchor_ndcg: float, tolerance: float = 0.02) -> dict:
    """
    Experiment 002's closure control.

    The all-on corner of the lexical factorial (idf, tf-saturation and length-norm
    all on) is, by construction, full BM25. It must agree with the externally
    anchored BM25 already measured in 001 (results/001/<dataset>/bm25_control.json)
    within tolerance. If it does not, the ladder's own implementation is wrong and
    nothing from experiment 002 ships — this is the single most important control
    in the experiment, because it is what connects a hand-built scorer to an
    external reference.

    Tolerance is tighter than 001's 0.10 anchor-to-BEIR-published tolerance: that
    comparison crosses codebases (this repo's tokenisation vs Anserini's), but this
    one compares two numbers produced inside this repository, so only library and
    rounding differences should separate them.
    """
    # Rounded to 4dp before comparison, not just before display: two numbers a
    # human would call "exactly at tolerance" (e.g. 0.520 vs 0.500 at tolerance
    # 0.02) can differ from the boundary by float noise in the 17th significant
    # digit, and that noise should not flip a control's pass/fail.
    diff = round(abs(our_ndcg - anchor_ndcg), 4)
    return {
        "our_ndcg_cut_10": round(our_ndcg, 4),
        "anchor_ndcg_cut_10": round(anchor_ndcg, 4),
        "absolute_difference": diff,
        "tolerance": tolerance,
        "passed": diff <= tolerance,
    }


def embedding_shuffle(normal_ndcg: float, shuffled_ndcg: float, chance_ceiling: float = 0.15) -> dict:
    """
    Experiment 002's dense-rung control.

    Permuting the document embedding matrix before scoring breaks every
    doc-id -> vector correspondence, so a correctly wired dense retriever must
    collapse toward chance once its embeddings are shuffled. If it does not, the
    index or the id bookkeeping around it is broken — a broken vector index would
    otherwise report a real-looking number instead of failing loudly.
    """
    return {
        "normal_ndcg_cut_10": round(normal_ndcg, 4),
        "shuffled_ndcg_cut_10": round(shuffled_ndcg, 4),
        "chance_ceiling": chance_ceiling,
        "passed": shuffled_ndcg <= chance_ceiling,
    }
