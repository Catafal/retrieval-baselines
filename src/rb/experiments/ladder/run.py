"""
Experiment 002 runner.

    python -m rb.experiments.ladder.run --dataset scifact --rung coordination
    python -m rb.experiments.ladder.run --dataset scifact --rung lexical-factorial

Mirrors 001's rb.run in shape (one command, results/002/<dataset>/<rung>/ artifacts)
but is written once against the Retriever interface via rb.retriever.run_rung(), so
it does not grow a new branch per rung the way a from-scratch runner would.

Query subsample is reused from rb.run.select_queries with the SAME seed, not
reimplemented, so the subsample is identical to 001's and per-query pairing
across the two entries stays valid (spec requirement: same corpora, same query
subsample, same metric code).

The dense and hybrid rungs (`--rung dense`, `--rung hybrid`, `--rung
dense-hybrid-analysis`) ARE wired in now, gated on protocol-002-amendment-1
being tagged and `sentence-transformers==6.0.0` being pinned in
requirements.txt per that amendment's section 3 — both true as of this
change. They are still not part of `make reproduce-002-lexical`: that target
is the cheap lexical rungs only, and dense/hybrid need a real encoder pass,
which is a separate, explicitly invoked step.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from rb import controls, datasets, metrics
from rb.retriever import run_rung
from rb.run import select_queries
from rb.stats import holm_correction, paired_bootstrap, shapley_bootstrap
from rb.experiments.ladder import query_properties
from rb.experiments.ladder.analysis import win_loss_by_property
from rb.experiments.ladder.retrievers.coordination import CoordinationRetriever
from rb.experiments.ladder.retrievers.dense import DenseRetriever, SentenceTransformerEncoder
from rb.experiments.ladder.retrievers.hybrid import HybridRetriever
from rb.experiments.ladder.retrievers.lexical import (
    ALL_CONFIGS,
    LADDER,
    PLAYERS,
    active_players,
    build_index,
    full_bm25,
    shapley_from_ndcg,
)

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results" / "002"
# 001's BM25 numbers are the external anchor the closure control checks against.
ANCHOR_DIR = ROOT / "results" / "001"

# protocols/002-amendment-1-dense.md section 3, pinned before any embedding is
# computed. Not a default a caller can override from the CLI — a different
# model or revision is a different experiment, not a flag on this one.
ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Document embeddings cached here, keyed by revision (see DenseRetriever's
# _cached_doc_vectors), so a rerun on the same corpus does not re-embed.
EMBEDDING_CACHE_DIR = ROOT / "data" / "embeddings_cache"

# Self-retrieval control (amendment section 9) samples this many documents
# rather than the whole corpus — re-encoding every document as a query is the
# same cost as the retrieval run itself. Deterministic (sorted doc ids, first
# N), not random, so the control's own pass/fail is itself reproducible.
SELF_RETRIEVAL_SAMPLE_SIZE = 100

# Same seed as every other bootstrap/shuffle in this repo (BOOTSTRAP_SEED in
# rb.stats, the SciFact query subsample seed) — one seed, reused everywhere,
# rather than a fresh one invented per new piece of code.
CONTROL_SEED = 20260818


def _load_subsampled(dataset: str):
    corpus, queries, qrels = datasets.load(dataset)
    qids, sampled = select_queries(queries)
    subsampled_queries = {q: queries[q] for q in qids}
    return corpus, subsampled_queries, qrels, sampled


def _per_query_ndcg(out_dir: Path, qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """
    Recompute per-query nDCG@10 from the per_query.jsonl artifact run_rung()
    already wrote, instead of re-running retrieval or having run_rung() write
    out a redundant second artifact. per_query.jsonl only records rank ORDER
    (the retrieved doc-id list), not the retriever's raw score magnitude — but
    pytrec_eval's ranked measures depend only on order, so assigning any
    strictly-decreasing synthetic score to that order (len(retrieved) - i)
    reproduces exactly the per-query nDCG run_rung computed the first time.
    """
    run: dict[str, dict[str, float]] = {}
    with open(out_dir / "per_query.jsonl", encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            retrieved = row["retrieved"]
            run[row["query_id"]] = {d: len(retrieved) - i for i, d in enumerate(retrieved)}
    per_query = metrics.score_ranked(qrels, run)
    return {q: per_query[q]["ndcg_cut_10"] for q in per_query}


def run_coordination(dataset: str, top_k: int = 100) -> dict:
    """Rung 0, carried forward unchanged (see retrievers/coordination.py)."""
    corpus, queries, qrels, sampled = _load_subsampled(dataset)
    out_dir = RESULTS / dataset / "coordination"
    return run_rung(
        CoordinationRetriever(), dataset, corpus, queries, qrels, out_dir,
        top_k=top_k, subsampled=sampled, seed=20260818 if sampled else None,
    )


def run_lexical_factorial(dataset: str, top_k: int = 100) -> dict:
    """
    All eight lexical configurations on one dataset, plus the Shapley attribution
    computed from that factorial. Cheap: minutes per dataset (spec estimate),
    which is why this is the target wired into the Makefile's new
    reproduce-002-lexical rule.
    """
    corpus, queries, qrels, sampled = _load_subsampled(dataset)
    seed = 20260818 if sampled else None

    # The significance block below recomputes nDCG@10 from each rung's
    # per_query.jsonl, which stores only the top_k the run wrote. Below ten,
    # that reconstruction scores against fewer than ten documents and returns
    # something that looks like nDCG@10 and is not. No caller does this today;
    # the guard is here so none can start without noticing.
    if top_k < 10:
        raise ValueError(f"top_k={top_k} cannot support ndcg_cut_10; need at least 10")

    # Built ONCE per dataset, shared read-only across all eight configs below.
    # The index (postings, doc lengths, document frequencies, average length) is
    # config-independent — only the scoring formula changes between configs —
    # so rebuilding it inside each of the eight run_rung() calls would repeat
    # the one genuinely expensive step (a full tokenisation pass over the
    # corpus) eight times for no reason. See LexicalIndex's docstring.
    t_index = time.perf_counter()
    shared_index = build_index(corpus)
    index_seconds = time.perf_counter() - t_index
    t_scoring = time.perf_counter()

    ndcg_by_config = {}
    summaries = {}
    out_dirs = {}
    for cfg in ALL_CONFIGS:
        cfg_with_index = dataclasses.replace(cfg, index=shared_index)
        out_dir = RESULTS / dataset / cfg.name
        summary = run_rung(
            cfg_with_index, dataset, corpus, queries, qrels, out_dir, top_k=top_k, subsampled=sampled, seed=seed
        )
        # `index` is excluded from LexicalRetriever equality/hash (compare=False),
        # so cfg and cfg_with_index are the same dict key — using the plain cfg
        # here keeps ndcg_by_config[full_bm25()] below working without callers
        # needing to know an index was ever attached.
        ndcg_by_config[cfg] = summary["ranked"]["ndcg_cut_10"]
        summaries[cfg.name] = summary
        out_dirs[cfg.name] = out_dir

    # The closure control, run BEFORE anything is written. The all-on corner is full
    # BM25 by construction, so it has to agree with the externally anchored BM25 that
    # 001 already checked against the published BEIR figures. Without this gate the
    # factorial happily produces eight plausible numbers and a Shapley attribution
    # even when the scorer is wrong, which is the exact failure that retracted the
    # previous entry: a number that looks entirely reasonable and is not right.
    anchor = json.loads((ANCHOR_DIR / dataset / "bm25_control.json").read_text())
    closure = controls.bm25_closure(
        ndcg_by_config[full_bm25()],
        anchor["published_bm25_ndcg_cut_10"],
        anchor_ndcg=anchor["ndcg_cut_10"],
    )
    if not closure["passed"]:
        raise RuntimeError(
            f"BM25 closure control failed on {dataset}: {closure}. The all-on corner of "
            "the factorial is meant to BE full BM25, so disagreeing with the anchor means "
            "the lexical scorer is wrong. Nothing is written."
        )

    scoring_seconds = time.perf_counter() - t_scoring

    shapley = shapley_from_ndcg(ndcg_by_config)

    # Per-query nDCG@10 for every one of the eight configs, reconstructed from
    # the per_query.jsonl artifact each config's run_rung() call already wrote
    # above — no rescoring, no second retrieval. Feeds two consumers below:
    # the LADDER adjacent-rung significance (as before), and the Shapley
    # bootstrap, which needs every cell of the factorial, not just LADDER's
    # four-rung slice through it.
    qids = sorted(queries)
    aligned_qrels = {q: qrels[q] for q in qids}
    per_config_ndcg = {cfg: _per_query_ndcg(out_dirs[cfg.name], aligned_qrels) for cfg in ALL_CONFIGS}
    ladder_ndcg = [per_config_ndcg[cfg] for cfg in LADDER]

    # Adjacent-rung significance, run BEFORE anything is written — same shape as
    # the closure control above. The point estimates in `ndcg_by_config` cannot
    # say whether adding the next mechanism up LADDER bought a real gain or is
    # noise on this query subsample; only the paired per-query difference can,
    # because it compares the two rungs on the SAME queries rather than two
    # marginal means. Holm correction is then applied once across the three
    # comparisons made on this dataset, per protocols/002-ladder.md section 5.
    comparisons = []
    for i in range(len(LADDER) - 1):
        lo_cfg, hi_cfg = LADDER[i], LADDER[i + 1]
        lo_scores = [ladder_ndcg[i][q] for q in qids]
        hi_scores = [ladder_ndcg[i + 1][q] for q in qids]
        boot = paired_bootstrap(hi_scores, lo_scores)
        comparisons.append({"from": lo_cfg.name, "to": hi_cfg.name, **boot})
    significant = holm_correction([c["p_value"] for c in comparisons])
    for comparison, is_significant in zip(comparisons, significant):
        comparison["holm_significant"] = is_significant

    # Shapley bootstrap: 95% interval per mechanism plus pairwise ordering
    # fractions, from the SAME per-query reconstruction above. See
    # rb.stats.shapley_bootstrap's docstring for why queries (not cells) are
    # the resampling unit and why one query draw must cover all eight cells.
    per_cell_scores = {active_players(cfg): [per_config_ndcg[cfg][q] for q in qids] for cfg in ALL_CONFIGS}
    bootstrap = shapley_bootstrap(per_cell_scores, list(PLAYERS))

    factorial_summary = {
        "dataset": dataset,
        "configs": {cfg.name: ndcg for cfg, ndcg in ndcg_by_config.items()},
        "shapley_ndcg_cut_10": shapley,
        "shapley_ci95": bootstrap["phi_ci95"],
        "shapley_pairwise_ordering": bootstrap["pairwise_ordering"],
        # Ties travel with the ordering or the ordering is misreadable: 0.0 in the
        # one direction reported means "never won", which is not the same as
        # "never differed", and only the tie fraction distinguishes them.
        "shapley_pairwise_ties": bootstrap["pairwise_ties"],
        "adjacent_rung_comparisons": comparisons,
        "controls": {"bm25_closure": closure},
        # The index is built once and shared, so its cost lands in no config's own
        # `cost.total_seconds` and would otherwise appear nowhere at all. Every
        # per-config figure would then understate what reproducing that config from
        # cold actually costs, and the total wall clock for the factorial would exist
        # only as prose. Both numbers are recorded here so a reader can check the
        # cost claims against an artifact rather than against a sentence.
        "cost": {
            "index_build_seconds": round(index_seconds, 1),
            "scoring_seconds": round(scoring_seconds, 1),
            "total_seconds": round(index_seconds + scoring_seconds, 1),
        },
    }
    out_dir = RESULTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lexical_factorial.json").write_text(json.dumps(factorial_summary, indent=2) + "\n")
    return factorial_summary


def _make_encoder() -> SentenceTransformerEncoder:
    """One pinned encoder, per amendment section 3. A single function so
    run_dense and run_hybrid cannot drift into loading two different encoders
    for what is supposed to be the same dense arm."""
    return SentenceTransformerEncoder(ENCODER_MODEL_NAME, ENCODER_REVISION)


def _write_summary(out_dir: Path, summary: dict) -> None:
    """run_rung already wrote summary.json once; the two dense-specific
    controls below are computed AFTER that call (they need the retriever and
    the scored nDCG run_rung just produced), so the artifact is rewritten in
    place rather than left to describe a summary that predates its own
    controls section."""
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_dense(dataset: str, top_k: int = 100) -> dict:
    """
    Rung 9 — dense (protocols/002-amendment-1-dense.md section 4).

    Runs the scored retrieval, then the two dense-specific controls that halt
    the run on failure: embedding shuffle (must collapse toward chance) and
    self-retrieval (a document must retrieve itself at rank 1 when used as its
    own query). Both need the SAME embeddings the scored run used, which the
    on-disk cache (DenseRetriever(cache_dir=...)) makes cheap to recompute
    from rather than a second full encoder pass.
    """
    corpus, queries, qrels, sampled = _load_subsampled(dataset)
    seed = CONTROL_SEED if sampled else None

    encoder = _make_encoder()
    retriever = DenseRetriever(encoder, cache_dir=EMBEDDING_CACHE_DIR)
    out_dir = RESULTS / dataset / "dense"
    summary = run_rung(
        retriever, dataset, corpus, queries, qrels, out_dir,
        top_k=top_k, subsampled=sampled, seed=seed, extra_manifest=retriever.manifest(),
    )
    normal_ndcg = summary["ranked"]["ndcg_cut_10"]

    # Embedding shuffle: same corpus, same encoder (cache hit), embeddings
    # permuted before scoring. Must collapse to at most the chance ceiling.
    shuffled = DenseRetriever(encoder, shuffle_seed=CONTROL_SEED, cache_dir=EMBEDDING_CACHE_DIR)
    shuffled_run = shuffled.retrieve(corpus, queries, top_k=top_k)
    shuffled_per_query = metrics.score_ranked(qrels, shuffled_run)
    shuffled_ndcg = metrics.mean([shuffled_per_query[q]["ndcg_cut_10"] for q in queries])
    shuffle_result = controls.embedding_shuffle(normal_ndcg, shuffled_ndcg)

    # Self-retrieval: a deterministic sample of documents, embedded and used
    # as their own query text, must come back at rank 1.
    sample_ids = sorted(corpus)[:SELF_RETRIEVAL_SAMPLE_SIZE]
    self_retrieval_result = controls.self_retrieval(retriever, corpus, sample_ids)

    summary["controls"]["embedding_shuffle"] = shuffle_result
    summary["controls"]["self_retrieval"] = self_retrieval_result
    _write_summary(out_dir, summary)

    if not shuffle_result["passed"]:
        raise RuntimeError(f"embedding shuffle control failed on {dataset}: {shuffle_result}. Nothing else runs.")
    if not self_retrieval_result["passed"]:
        raise RuntimeError(f"self-retrieval control failed on {dataset}: {self_retrieval_result}. Nothing else runs.")
    return summary


def run_hybrid(dataset: str, top_k: int = 100) -> dict:
    """
    Rung 10 — hybrid (RRF of full BM25 and dense, k=60, amendment section 5).

    Builds its own full-BM25 lexical index and its own dense retriever (the
    same pinned encoder run_dense uses; the embedding cache means this does
    not re-embed if run_dense already ran on this dataset) rather than reusing
    retriever objects a caller happens to have lying around — HybridRetriever
    owns exactly two Retriever instances, matching hybrid.py's own contract.
    """
    corpus, queries, qrels, sampled = _load_subsampled(dataset)
    seed = CONTROL_SEED if sampled else None

    lexical = dataclasses.replace(full_bm25(), index=build_index(corpus))
    encoder = _make_encoder()
    dense = DenseRetriever(encoder, cache_dir=EMBEDDING_CACHE_DIR)
    hybrid = HybridRetriever(lexical, dense)

    out_dir = RESULTS / dataset / "hybrid"
    summary = run_rung(
        hybrid, dataset, corpus, queries, qrels, out_dir,
        top_k=top_k, subsampled=sampled, seed=seed,
        extra_manifest={"lexical_component": lexical.name, "dense_component": dense.manifest(), "rrf_k": hybrid.k},
    )
    return summary


def run_dense_hybrid_analysis(dataset: str, top_k: int = 100) -> dict:
    """
    The statistics and query-property analysis amendment sections 7 and 8 ask
    for, run AFTER dense, hybrid, coordination and the lexical factorial all
    have committed per_query.jsonl artifacts on this dataset — this function
    only reconstructs from those artifacts (via _per_query_ndcg, the same
    helper run_lexical_factorial's own significance block uses), it does not
    retrieve anything itself.
    """
    corpus, queries, qrels, _ = _load_subsampled(dataset)
    qids = sorted(queries)
    aligned_qrels = {q: qrels[q] for q in qids}

    dense_ndcg = _per_query_ndcg(RESULTS / dataset / "dense", aligned_qrels)
    hybrid_ndcg = _per_query_ndcg(RESULTS / dataset / "hybrid", aligned_qrels)
    bm25_ndcg = _per_query_ndcg(RESULTS / dataset / full_bm25().name, aligned_qrels)
    coordination_ndcg = _per_query_ndcg(RESULTS / dataset / "coordination", aligned_qrels)

    # Section 7: the four pre-registered query properties, dense vs full BM25.
    index = build_index(corpus)
    properties = {
        q: query_properties.compute_query_properties(queries[q], sorted(qrels[q]), corpus, index)
        for q in qids
    }
    win_loss = win_loss_by_property(dense_ndcg, bm25_ndcg, properties)

    # Section 8: paired bootstrap + Holm correction across the three
    # pre-registered comparisons for this corpus. "Hybrid vs its better
    # component" is decided by each component's OWN mean nDCG@10 on this
    # dataset, not assumed — a hybrid is not guaranteed to beat dense just
    # because dense usually beats lexical.
    bm25_mean = metrics.mean([bm25_ndcg[q] for q in qids])
    dense_mean = metrics.mean([dense_ndcg[q] for q in qids])
    better_component_name = "full_bm25" if bm25_mean >= dense_mean else "dense"
    better_component_ndcg = bm25_ndcg if better_component_name == "full_bm25" else dense_ndcg

    comparisons = []
    for label, a_scores, b_scores in (
        ("dense_vs_full_bm25", dense_ndcg, bm25_ndcg),
        ("dense_vs_coordination", dense_ndcg, coordination_ndcg),
        (f"hybrid_vs_better_component({better_component_name})", hybrid_ndcg, better_component_ndcg),
    ):
        boot = paired_bootstrap([a_scores[q] for q in qids], [b_scores[q] for q in qids])
        comparisons.append({"comparison": label, **boot})
    significant = holm_correction([c["p_value"] for c in comparisons])
    for comparison, is_significant in zip(comparisons, significant):
        comparison["holm_significant"] = is_significant

    analysis = {
        "dataset": dataset,
        "mean_ndcg_cut_10": {
            "dense": round(dense_mean, 4),
            "hybrid": round(metrics.mean([hybrid_ndcg[q] for q in qids]), 4),
            "full_bm25": round(bm25_mean, 4),
            "coordination": round(metrics.mean([coordination_ndcg[q] for q in qids]), 4),
        },
        "comparisons": comparisons,
        "query_property_win_loss": win_loss,
    }
    out_dir = RESULTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dense_hybrid_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return analysis


def _load_queries_and_qrels(dataset: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """
    The query subsample and gold judgements only — NOT the corpus text.

    add_shapley_intervals() only needs query ids and qrels to recompute
    per-query nDCG@10 from the already-committed per_query.jsonl artifacts.
    Routing through `datasets.load()` (used by `_load_subsampled` above) would
    load the corpus too, which for HotpotQA is 5.2M documents — turning a
    regeneration meant to cost seconds back into a multi-minute one for data
    this function never reads. Filtering queries to those with qrels mirrors
    `datasets.load()`'s own rule (BEIR ships more queries than qrels for some
    datasets) without loading the corpus to get there.
    """
    qrels = datasets.load_qrels(dataset)
    queries = {qid: q for qid, q in datasets.load_queries(dataset).items() if qid in qrels}
    qids, _ = select_queries(queries)
    return {q: queries[q] for q in qids}, qrels


def add_shapley_intervals(dataset: str) -> dict:
    """
    Add shapley_ci95 and shapley_pairwise_ordering to an already-committed
    lexical_factorial.json, reconstructed entirely from the per_query.jsonl
    artifacts run_lexical_factorial already wrote for this dataset — no
    retrieval, no index build, no closure control re-run. This is what makes
    a stats-layer addition cost seconds rather than HotpotQA's 45-minute
    scoring pass.

    Recomputes the eight cells' mean nDCG@10 from those same artifacts as a
    consistency check against what is already stored in `configs` and
    `shapley_ndcg_cut_10`: if either disagrees beyond floating-point noise,
    something about the committed artifacts or this reconstruction is wrong,
    and this raises rather than silently writing a subtly different point
    estimate next to the (unaffected) bootstrap interval.

    The per-config means are rounded to 4 decimals before comparison and
    before feeding shapley_from_ndcg — `run_rung`'s own `summary["ranked"]`
    is rounded to 4 decimals (see retriever.py), so `ndcg_by_config` in
    run_lexical_factorial, and therefore the committed `shapley_ndcg_cut_10`
    it produced, were themselves computed from rounded inputs, not full
    per-query precision. Comparing at full precision would flag every dataset
    as a "mismatch" over rounding noise the original computation already had.
    """
    out_dir = RESULTS / dataset
    factorial_path = out_dir / "lexical_factorial.json"
    summary = json.loads(factorial_path.read_text())

    queries, qrels = _load_queries_and_qrels(dataset)
    qids = sorted(queries)
    aligned_qrels = {q: qrels[q] for q in qids}
    per_config_ndcg = {cfg: _per_query_ndcg(out_dir / cfg.name, aligned_qrels) for cfg in ALL_CONFIGS}

    recomputed_mean = {
        cfg: round(sum(per_config_ndcg[cfg][q] for q in qids) / len(qids), 4) for cfg in ALL_CONFIGS
    }
    for cfg, value in recomputed_mean.items():
        stored = summary["configs"][cfg.name]
        if abs(value - stored) > 1e-9:
            raise RuntimeError(
                f"{dataset}/{cfg.name}: recomputed mean nDCG@10 {value!r} disagrees with the "
                f"committed value {stored!r}. Refusing to regenerate on a point-value mismatch — "
                "see run_lexical_factorial's per_query.jsonl artifacts for this dataset."
            )
    recomputed_shapley = shapley_from_ndcg(recomputed_mean)
    for player, value in recomputed_shapley.items():
        stored = summary["shapley_ndcg_cut_10"][player]
        if abs(value - stored) > 1e-9:
            raise RuntimeError(
                f"{dataset}: recomputed Shapley value for {player!r} ({value!r}) disagrees with the "
                f"committed value ({stored!r}). Refusing to regenerate on a point-value mismatch."
            )

    per_cell_scores = {active_players(cfg): [per_config_ndcg[cfg][q] for q in qids] for cfg in ALL_CONFIGS}
    bootstrap = shapley_bootstrap(per_cell_scores, list(PLAYERS))

    summary["shapley_ci95"] = bootstrap["phi_ci95"]
    summary["shapley_pairwise_ordering"] = bootstrap["pairwise_ordering"]
    summary["shapley_pairwise_ties"] = bootstrap["pairwise_ties"]
    factorial_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 002 — the retrieval ladder")
    p.add_argument("--dataset", required=True, choices=datasets.DATASETS)
    p.add_argument(
        "--rung",
        required=True,
        choices=(
            "coordination", "lexical-factorial", "shapley-intervals",
            "dense", "hybrid", "dense-hybrid-analysis",
        ),
    )
    args = p.parse_args()

    t0 = time.time()
    if args.rung == "coordination":
        summary = run_coordination(args.dataset)
    elif args.rung == "lexical-factorial":
        summary = run_lexical_factorial(args.dataset)
    elif args.rung == "shapley-intervals":
        summary = add_shapley_intervals(args.dataset)
    elif args.rung == "dense":
        summary = run_dense(args.dataset)
    elif args.rung == "hybrid":
        summary = run_hybrid(args.dataset)
    else:
        summary = run_dense_hybrid_analysis(args.dataset)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
