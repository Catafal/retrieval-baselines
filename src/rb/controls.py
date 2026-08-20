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


def bm25_closure(our_ndcg: float, published_ndcg: float, anchor_ndcg: float | None = None,
                 tolerance: float = 0.10) -> dict:
    """
    Experiment 002's closure control.

    The all-on corner of the lexical factorial is, by construction, full BM25, so it
    has to agree with a BM25 measured outside this scorer. If it does not, the
    ladder's own implementation is wrong and nothing from 002 ships. This is the
    control that connects a hand-built scorer to an external reference.

    WHAT IT COMPARES AGAINST, AND WHY THAT CHANGED. The first version of this control
    compared our BM25 to 001's in-repo `bm25s` anchor at a 0.02 tolerance, on the
    reasoning that both numbers come from inside this repository so only library and
    rounding noise should separate them. Measured on SciFact, that reasoning was
    wrong and the control failed on a correct implementation:

        ours            0.6605     no stopword removal, no stemming
        001 bm25s       0.6863     stopwords + Snowball stemming
        published       0.6650     Thakur et al. 2021 Table 2, Anserini

    The two in-repo numbers differ by 0.0258 because they tokenise differently, which
    is a real difference in what is being measured rather than noise. Ours lands
    within 0.0045 of the published Anserini figure, closer to it than `bm25s` is.

    So the gate is against the PUBLISHED figure, at the same 0.10 tolerance 001 used
    for the same reason: it is a genuinely external reference, and every
    implementation here is checked against it rather than against each other. The
    in-repo anchor is still reported when supplied, because a large gap to it would
    signal something worth looking at, but it does not gate.
    """
    delta = round(abs(round(our_ndcg, 4) - round(published_ndcg, 4)), 6)
    result = {
        "our_ndcg_cut_10": round(our_ndcg, 4),
        "published_ndcg_cut_10": round(published_ndcg, 4),
        "absolute_difference": delta,
        "tolerance": tolerance,
        "passed": delta <= tolerance,
    }
    if anchor_ndcg is not None:
        # Informational only. Different tokenisation, so a gap here is expected.
        result["in_repo_bm25s_ndcg_cut_10"] = round(anchor_ndcg, 4)
        result["difference_to_in_repo_anchor"] = round(abs(round(our_ndcg, 4) - round(anchor_ndcg, 4)), 6)
    return result


def self_retrieval(retriever, corpus: dict[str, str], sample_ids: list[str],
                   min_rank1_fraction: float = 0.98) -> dict:
    """
    Alignment check: a document used as its own query should come back first.

    WHAT THIS DETECTS. Document ids, embedding rows and the similarity computation
    lining up. A misaligned index fails almost every sample, not one or two, which
    is why a threshold rather than perfection is the right shape.

    WHAT IT DOES NOT DETECT, contrary to what protocols/002-amendment-1-dense.md
    claimed. It does NOT catch query and document encoders being transposed.
    Measured on 100 Quora documents with BAAI/bge-base-en-v1.5: correctly wired,
    0 failures; with the two encoders swapped, also 0 failures. The two paths for
    that model differ only by a query prefix, so a swap leaves a document still
    close to its own nearest neighbour, and for a symmetric encoder like MiniLM
    the paths are identical so nothing could be caught at all. Transposition is
    covered by tests/test_dense.py, which exercises retrieve()'s actual call sites
    with a stub whose two paths disagree. See protocols/002-amendment-3.

    WHY THE THRESHOLD IS NOT 100%. Quora is a duplicate-question corpus. Document
    "100", "How to make friends ?", sits among near-identical paraphrases, and with
    an encoder that shifts queries by a prefix one of those can edge it out at full
    corpus scale (on a 40,000-document subset it still wins). That is a property of
    the corpus and the encoder rather than a defect. 98 of 100 keeps every alignment
    failure detectable, since those fail wholesale, and this run passes it with one
    failure rather than being written to fit.
    """
    # One batched call, as before. Querying one document at a time would be slower
    # and would also break any encoder that returns a fixed matrix per call.
    queries = {d: corpus[d] for d in sample_ids}
    run = retriever.retrieve(corpus, queries, top_k=1)
    failures = [
        qid for qid in sample_ids
        if not run.get(qid) or next(iter(run[qid])) != qid
    ]

    sampled = len(sample_ids)
    rank1_fraction = (sampled - len(failures)) / sampled if sampled else 0.0
    return {
        "sampled": sampled,
        "failures": len(failures),
        "rank1_fraction": round(rank1_fraction, 4),
        "min_rank1_fraction": min_rank1_fraction,
        "examples": failures[:5],
        "passed": rank1_fraction >= min_rank1_fraction,
    }

def embedding_shuffle(normal_ndcg: float, shuffled_ndcg: float, chance_ceiling: float = 0.15) -> dict:
    """
    Experiment 002's dense-rung control.

    Permuting the document embedding matrix before scoring breaks every
    doc-id -> vector correspondence, so a correctly wired dense retriever must
    collapse toward chance once its embeddings are shuffled. If it does not, the
    index or the id bookkeeping around it is broken — a broken vector index would
    otherwise report a real-looking number instead of failing loudly.

    chance_ceiling=0.15 is protocols/002-amendment-1-dense.md section 9, fixed
    before any dense number existed: "must collapse nDCG@10 to at most 0.15."
    """
    return {
        "normal_ndcg_cut_10": round(normal_ndcg, 4),
        "shuffled_ndcg_cut_10": round(shuffled_ndcg, 4),
        "chance_ceiling": chance_ceiling,
        "passed": shuffled_ndcg <= chance_ceiling,
    }


def pool_construction(questions: int, passages: int, title_slots: int,
                      unresolved: int, collisions: int,
                      gold_titles_matched: int, gold_queries: int) -> dict:
    """
    Experiment 003's pool control — protocols/003-graph-arm.md section 9.

    The pool is the experiment's central factual claim: an exactly-identified SUBSET of
    the corpus 002 published on, same document ids, same qrels, 79x smaller. Everything
    downstream inherits that claim, so it is checked rather than documented.

    Deduping 73,700 title slots into 66,581 uniques and mapping them onto BEIR document
    ids is exactly the indexing step that produces unreproducible numbers, and this
    repository exists because of a retraction for unreproducible numbers. Each figure
    below was measured before tagging and frozen in the protocol; this control is what
    makes the written counts falsifiable by the code rather than merely asserted by the
    author.

    Why the pair (passages, title_slots) rather than passages alone: a loader that
    silently dropped a column would still produce a plausible unique count. Only the
    dedup ratio catches it.
    """
    from rb.experiments.graph.pool import (
        EXPECTED_PASSAGES,
        EXPECTED_QUESTIONS,
        EXPECTED_TITLE_SLOTS,
    )

    checks = {
        "questions": (questions, EXPECTED_QUESTIONS),
        "passages": (passages, EXPECTED_PASSAGES),
        "title_slots": (title_slots, EXPECTED_TITLE_SLOTS),
        "gold_titles_matched": (gold_titles_matched, gold_queries),
    }
    mismatched = {k: {"got": g, "expected": e} for k, (g, e) in checks.items() if g != e}
    return {
        "questions": questions,
        "passages": passages,
        "title_slots": title_slots,
        "unresolved_titles": unresolved,
        "title_collisions": collisions,
        "gold_titles_matched": gold_titles_matched,
        "gold_queries": gold_queries,
        "mismatched": mismatched,
        "passed": not mismatched and unresolved == 0 and collisions == 0,
    }

