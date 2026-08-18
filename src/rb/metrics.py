"""
Scoring.

Uses pytrec_eval rather than a hand-rolled nDCG. The whole point of running on
BEIR is that the resulting numbers sit next to published ones, and nDCG has enough
variants that a private implementation would quietly break that comparison.

Two families are reported and they answer different questions:

  ranked  — nDCG@10, Recall@10, Recall@100. Comparable to the literature.
  set     — recall over grep's ENTIRE unranked output, plus the size of that output.
            This is the honest "grep alone" figure. Grep hands back a pile; the
            pile's size is the cost, and it is the number nobody publishes.
"""

import pytrec_eval

MEASURES = {"ndcg_cut_10", "recall_10", "recall_100"}


def score_ranked(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per-query trec_eval measures. Returns query_id -> {measure: value}."""
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, MEASURES)
    return evaluator.evaluate(run)


def set_recall(gold: dict[str, int], retrieved_ids: set[str]) -> float:
    """Fraction of gold documents anywhere in the unranked output."""
    if not gold:
        return 0.0
    return len(set(gold) & retrieved_ids) / len(gold)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
