"""
BM25 control — an instrument check, not an experimental arm.

Its only job is to be compared against the BM25 nDCG@10 figures published with BEIR.
If this lands far from those, the harness is broken and nothing ships. The retracted
entry that preceded this repo had no external anchor of any kind, which is exactly
how it published figures that could not be reproduced.

It is NOT the vector comparison. It is NOT a headline number. Entry 001 reports it
in methods, as evidence the measuring apparatus works.

Tolerance is deliberately loose: BEIR's published BM25 comes from Anserini/Elasticsearch
with different tokenisation and parameters, so an exact match is not achievable and
claiming one would be dishonest. What is achievable is the same ballpark.
"""

import json
from pathlib import Path

import bm25s
import Stemmer

from rb import datasets, metrics

ROOT = Path(__file__).resolve().parents[2]

# nDCG@10 for BM25 as published in the BEIR paper (Thakur et al., 2021, Table 2).
# These are the anchor. They are not our numbers and are never presented as such.
PUBLISHED_BM25_NDCG10 = {"scifact": 0.665, "quora": 0.789, "hotpotqa": 0.603}
TOLERANCE = 0.10


def run(dataset: str, top_k: int = 100) -> dict:
    corpus, queries, qrels = datasets.load(dataset)
    doc_ids = list(corpus)

    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize([corpus[d] for d in doc_ids], stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(tokens, show_progress=False)

    from rb.run import select_queries  # same subsample as the grep arm, same seed

    qids, sampled = select_queries(queries)
    q_tokens = bm25s.tokenize([queries[q] for q in qids], stopwords="en", stemmer=stemmer, show_progress=False)
    idx, scores = retriever.retrieve(q_tokens, k=top_k, show_progress=False)

    run_dict = {
        qid: {doc_ids[int(idx[i, j])]: float(scores[i, j]) for j in range(idx.shape[1])}
        for i, qid in enumerate(qids)
    }
    per_query = metrics.score_ranked({q: qrels[q] for q in qids}, run_dict)
    ndcg = metrics.mean([per_query[q]["ndcg_cut_10"] for q in qids])

    published = PUBLISHED_BM25_NDCG10[dataset]
    result = {
        "dataset": dataset,
        "queries_scored": len(qids),
        "subsampled": sampled,
        "ndcg_cut_10": round(ndcg, 4),
        "published_bm25_ndcg_cut_10": published,
        "absolute_difference": round(abs(ndcg - published), 4),
        "tolerance": TOLERANCE,
        "passed": abs(ndcg - published) <= TOLERANCE,
        "note": "Anchor is Thakur et al. 2021 Table 2 (Anserini BM25). Different implementation "
                "and tokenisation, so agreement is expected in ballpark only.",
    }
    out = ROOT / "results" / "001" / dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "bm25_control.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[1]), indent=2))
