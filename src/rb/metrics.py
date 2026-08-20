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

MEASURES is the set 001 and 002 published against, and it does not change. An
experiment that needs different cutoffs passes its own set instead (see
`measures` below and rb.experiments.graph.measures). Growing MEASURES globally
would have been the smaller diff and the wrong call: `make reproduce` is a
published promise about 001's output, tests/test_coordination_regression.py
asserts 001's `ranked` dict by exact equality, and both would break on a shape
change even though no measured VALUE moves. Additive means additive for the
experiment that asked, not for every artifact already on disk.
"""

import pytrec_eval

MEASURES = {"ndcg_cut_10", "recall_10", "recall_100"}


def score_ranked(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]],
                 measures: set[str] | None = None) -> dict[str, dict[str, float]]:
    """
    Per-query trec_eval measures. Returns query_id -> {measure: value}.

    `measures` defaults to MEASURES, so every existing caller is unaffected. Passing a
    set is how an experiment adds cutoffs without changing what earlier experiments
    computed — pytrec_eval evaluates each measure independently, so a larger set adds
    keys and never moves the values of the ones already there.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures or MEASURES)
    return evaluator.evaluate(run)


def set_recall(gold: dict[str, int], retrieved_ids: set[str]) -> float:
    """Fraction of gold documents anywhere in the unranked output."""
    if not gold:
        return 0.0
    return len(set(gold) & retrieved_ids) / len(gold)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
