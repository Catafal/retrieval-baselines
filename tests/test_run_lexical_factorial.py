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

import pytest

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


# A second, separate fixture for the Shapley-bootstrap/regeneration tests
# below. CORPUS's two same-length, same-term docs per query (e.g. glucose_paper
# vs insulin_paper on q1) produce raw scores that are IDENTICAL before the
# 1e-9-per-rank tie-break epsilon LexicalRetriever applies — a real, pre-existing
# edge case where pytrec_eval's own float precision can resolve that near-tie
# differently than the strict rank order _per_query_ndcg reconstructs from
# per_query.jsonl, for configs where length_norm is off. That is a property of
# the harness's tie handling, not of the Shapley bootstrap wiring under test
# here, so this fixture instead uses term counts that keep every document's
# raw score cleanly separated across all eight configs (verified directly:
# every config's own run_rung ndcg matches the per_query.jsonl reconstruction
# exactly), so these tests aren't coupled to that pre-existing subtlety.
CORPUS2 = {
    "d1": "alpha alpha alpha beta",
    "d2": "alpha beta beta beta beta",
    "d3": "gamma gamma delta",
    "d4": "delta epsilon epsilon epsilon epsilon epsilon",
    "d5": "zeta zeta zeta zeta",
}
QUERIES2 = {"q1": "alpha beta", "q2": "gamma delta"}
QRELS2 = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1, "d4": 1}}


def _fake_load_subsampled2(dataset: str):
    return CORPUS2, QUERIES2, QRELS2, False


def _anchor2_matching_our_own_full_bm25() -> float:
    run = full_bm25().retrieve(CORPUS2, QUERIES2, top_k=100)
    per_query = metrics.score_ranked(QRELS2, run)
    return metrics.mean([per_query[q]["ndcg_cut_10"] for q in QUERIES2])


def _write_anchor2(tmp_path) -> None:
    anchor_ndcg = _anchor2_matching_our_own_full_bm25()
    anchor_dir = tmp_path / "001" / "toy"
    anchor_dir.mkdir(parents=True)
    (anchor_dir / "bm25_control.json").write_text(
        json.dumps({"published_bm25_ndcg_cut_10": anchor_ndcg, "ndcg_cut_10": anchor_ndcg})
    )


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


def test_run_lexical_factorial_writes_shapley_intervals_and_pairwise_ordering(monkeypatch, tmp_path):
    """The Shapley bootstrap job's output must ride along with a fresh
    factorial run, not just the regeneration path — same fields, same shape,
    on every mechanism."""
    import rb.experiments.ladder.run as run_mod
    from rb.experiments.ladder.retrievers.lexical import PLAYERS

    monkeypatch.setattr(run_mod, "_load_subsampled", _fake_load_subsampled2)
    monkeypatch.setattr(run_mod, "RESULTS", tmp_path / "002")
    monkeypatch.setattr(run_mod, "ANCHOR_DIR", tmp_path / "001")
    _write_anchor2(tmp_path)

    summary = run_mod.run_lexical_factorial("toy", top_k=10)

    assert set(summary["shapley_ci95"]) == set(PLAYERS)
    for player in PLAYERS:
        lo, hi = summary["shapley_ci95"][player]
        assert lo <= hi

    assert set(summary["shapley_pairwise_ordering"]) == {
        "idf>tf_saturation", "idf>length_norm", "tf_saturation>length_norm",
    }
    for frac in summary["shapley_pairwise_ordering"].values():
        assert 0.0 <= frac <= 1.0


def test_add_shapley_intervals_regenerates_from_committed_artifacts_without_rerunning_retrieval(monkeypatch, tmp_path):
    """
    add_shapley_intervals must reproduce the committed exact Shapley point
    values from the per_query.jsonl artifacts alone (asserted implicitly: it
    would raise on a mismatch), add the new interval/ordering fields, and
    leave everything else in lexical_factorial.json untouched — all without
    calling build_index or retrieve() again, which is checked by monkeypatching
    both to explode if called.
    """
    import rb.experiments.ladder.run as run_mod
    from rb.experiments.ladder.retrievers import lexical as lexical_mod

    monkeypatch.setattr(run_mod, "_load_subsampled", _fake_load_subsampled2)
    monkeypatch.setattr(run_mod, "RESULTS", tmp_path / "002")
    monkeypatch.setattr(run_mod, "ANCHOR_DIR", tmp_path / "001")
    _write_anchor2(tmp_path)

    original = run_mod.run_lexical_factorial("toy", top_k=10)

    # Regeneration reads queries/qrels via _load_queries_and_qrels, not
    # _load_subsampled (which loads a corpus this step must not need) — patch
    # datasets.load_qrels/load_queries directly to the fixed toy fixtures.
    monkeypatch.setattr(run_mod.datasets, "load_qrels", lambda dataset: QRELS2)
    monkeypatch.setattr(run_mod.datasets, "load_queries", lambda dataset: QUERIES2)

    def _explode(*args, **kwargs):
        raise AssertionError("regeneration must not rebuild the index or re-run retrieval")

    monkeypatch.setattr(lexical_mod, "build_index", _explode)
    monkeypatch.setattr(run_mod, "build_index", _explode)

    regenerated = run_mod.add_shapley_intervals("toy")

    assert regenerated["configs"] == original["configs"]
    assert regenerated["shapley_ndcg_cut_10"] == original["shapley_ndcg_cut_10"]
    assert regenerated["adjacent_rung_comparisons"] == original["adjacent_rung_comparisons"]
    assert regenerated["controls"] == original["controls"]
    assert regenerated["cost"] == original["cost"]
    # The new fields are present and match what the fresh run already computed
    # (both were reconstructed from the same per_query.jsonl artifacts).
    assert regenerated["shapley_ci95"] == original["shapley_ci95"]
    assert regenerated["shapley_pairwise_ordering"] == original["shapley_pairwise_ordering"]

    on_disk = json.loads((tmp_path / "002" / "toy" / "lexical_factorial.json").read_text())
    assert on_disk == regenerated


def test_add_shapley_intervals_raises_on_a_point_value_mismatch(monkeypatch, tmp_path):
    """If the committed lexical_factorial.json disagrees with what its own
    per_query.jsonl artifacts recompute, add_shapley_intervals must refuse to
    regenerate rather than silently overwrite a subtly different number —
    this is the "must come out IDENTICAL, or stop and report" requirement."""
    import rb.experiments.ladder.run as run_mod

    monkeypatch.setattr(run_mod, "_load_subsampled", _fake_load_subsampled2)
    monkeypatch.setattr(run_mod, "RESULTS", tmp_path / "002")
    monkeypatch.setattr(run_mod, "ANCHOR_DIR", tmp_path / "001")
    _write_anchor2(tmp_path)
    run_mod.run_lexical_factorial("toy", top_k=10)

    monkeypatch.setattr(run_mod.datasets, "load_qrels", lambda dataset: QRELS2)
    monkeypatch.setattr(run_mod.datasets, "load_queries", lambda dataset: QUERIES2)

    factorial_path = tmp_path / "002" / "toy" / "lexical_factorial.json"
    summary = json.loads(factorial_path.read_text())
    summary["configs"][next(iter(summary["configs"]))] += 1.0  # corrupt one committed value
    factorial_path.write_text(json.dumps(summary))

    with pytest.raises(RuntimeError):
        run_mod.add_shapley_intervals("toy")
