"""
run_lexical_factorial's wiring: the shared index (one build per dataset, not
one per config) and the paired-bootstrap/Holm significance now folded into its
output (rb.stats.paired_bootstrap / holm_correction, previously unit-tested but
never called from src/).

Runs entirely against a tiny in-memory corpus, with RESULTS and ANCHOR_DIR
monkeypatched to a tmp_path — the house rule is "nothing gets written into
results/002/ until the protocol is tagged", and the only way to test the
runner's actual write path without violating that is to point it at a
throwaway directory instead of skipping the write path altogether.
"""

import json

from rb import metrics
from rb.experiments.ladder import run as run_module
from rb.experiments.ladder.retrievers.lexical import LADDER, full_bm25

CORPUS = {
    "insulin_paper": "insulin lowers blood glucose in diabetic patients over time",
    "photosynthesis_paper": "photosynthesis converts light energy into chemical energy in plants",
    "diabetes_review": "insulin resistance is strongly linked to type two diabetes mellitus",
    "cell_biology": "the mitochondria is the powerhouse of the eukaryotic cell",
    "glucose_paper": "blood glucose regulation depends on insulin secretion from the pancreas",
}
QUERIES = {
    "q1": "insulin diabetes",
    "q2": "energy cell",
    "q3": "glucose blood",
}
QRELS = {
    "q1": {"insulin_paper": 1, "diabetes_review": 1},
    "q2": {"photosynthesis_paper": 1, "cell_biology": 1},
    "q3": {"glucose_paper": 1, "insulin_paper": 1},
}


def _fake_load_subsampled(dataset: str):
    return CORPUS, QUERIES, QRELS, False  # sampled=False: every query used, seed reported as None


def _anchor_matching_our_own_full_bm25() -> float:
    """
    The closure control needs an externally-anchored published figure to check
    the all-on corner against. There is no real 001 anchor for a fabricated
    five-document corpus, so this computes our OWN full-BM25 nDCG directly
    (independent of run_lexical_factorial, using the public retrieve() +
    metrics API) and uses it as the anchor — a zero-difference case, which
    exercises the same control code real anchors do without asserting
    anything about a number this test invented.
    """
    run = full_bm25().retrieve(CORPUS, QUERIES, top_k=100)
    per_query = metrics.score_ranked(QRELS, run)
    return metrics.mean([per_query[q]["ndcg_cut_10"] for q in QUERIES])


def test_run_lexical_factorial_shares_one_index_across_all_eight_configs(monkeypatch, tmp_path):
    build_calls = []
    import rb.experiments.ladder.run as run_mod
    from rb.experiments.ladder.retrievers import lexical as lexical_mod

    real_build_index = lexical_mod.build_index

    def counting_build_index(corpus):
        build_calls.append(corpus)
        return real_build_index(corpus)

    monkeypatch.setattr(run_mod, "build_index", counting_build_index)
    monkeypatch.setattr(run_mod, "_load_subsampled", _fake_load_subsampled)
    monkeypatch.setattr(run_mod, "RESULTS", tmp_path / "002")
    monkeypatch.setattr(run_mod, "ANCHOR_DIR", tmp_path / "001")

    anchor_ndcg = _anchor_matching_our_own_full_bm25()
    anchor_dir = tmp_path / "001" / "toy"
    anchor_dir.mkdir(parents=True)
    (anchor_dir / "bm25_control.json").write_text(
        json.dumps({"published_bm25_ndcg_cut_10": anchor_ndcg, "ndcg_cut_10": anchor_ndcg})
    )

    run_mod.run_lexical_factorial("toy", top_k=10)

    # build_index is the expensive, config-independent step this refactor
    # exists to stop repeating — one call per dataset, not one per config.
    assert len(build_calls) == 1


def test_run_lexical_factorial_writes_adjacent_rung_comparisons(monkeypatch, tmp_path):
    import rb.experiments.ladder.run as run_mod

    monkeypatch.setattr(run_mod, "_load_subsampled", _fake_load_subsampled)
    monkeypatch.setattr(run_mod, "RESULTS", tmp_path / "002")
    monkeypatch.setattr(run_mod, "ANCHOR_DIR", tmp_path / "001")

    anchor_ndcg = _anchor_matching_our_own_full_bm25()
    anchor_dir = tmp_path / "001" / "toy"
    anchor_dir.mkdir(parents=True)
    (anchor_dir / "bm25_control.json").write_text(
        json.dumps({"published_bm25_ndcg_cut_10": anchor_ndcg, "ndcg_cut_10": anchor_ndcg})
    )

    summary = run_mod.run_lexical_factorial("toy", top_k=10)

    comparisons = summary["adjacent_rung_comparisons"]
    assert len(comparisons) == len(LADDER) - 1 == 3
    expected_pairs = list(zip((c.name for c in LADDER), (c.name for c in LADDER[1:])))
    assert [(c["from"], c["to"]) for c in comparisons] == expected_pairs
    for c in comparisons:
        assert set(c) == {"from", "to", "mean_diff", "ci95", "p_value", "holm_significant"}
        assert len(c["ci95"]) == 2
        assert c["ci95"][0] <= c["ci95"][1]
        assert 0.0 <= c["p_value"] <= 1.0
        assert isinstance(c["holm_significant"], bool)

    # Written to the monkeypatched tmp_path, never to the real results/002/.
    written = json.loads((tmp_path / "002" / "toy" / "lexical_factorial.json").read_text())
    assert written == summary
