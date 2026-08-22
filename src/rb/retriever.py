"""
The one seam every experiment-002 rung implements.

    class Retriever(Protocol):
        name: str
        def retrieve(corpus, queries, top_k) -> dict[query_id, dict[doc_id, score]]

Everything downstream of retrieval — scoring, the artifact writer, cost accounting,
the environment manifest — is written once against this interface (run_rung() below)
and is never specialised per rung. Adding a new rung means adding a new file that
implements this protocol; run_rung() does not change. This is deliberately the
highest seam available: the point where an experiment's only genuine variable, the
retrieval function, enters the system.

Shared across experiments, same as datasets.py / metrics.py / controls.py — this is
not experiment-002-specific, so a future experiment 003 built the same way reuses it
rather than forking it.
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol

from rb import controls, metrics

ROOT = Path(__file__).resolve().parents[2]


class Retriever(Protocol):
    """Structural interface — any object with a `name` attribute and a matching
    `retrieve` method satisfies this without inheriting from it."""

    name: str

    def retrieve(
        self, corpus: dict[str, str], queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        ...


ALLOW_DIRTY = "RB_ALLOW_DIRTY_SCORED_RUN"


def working_tree_state() -> dict:
    """
    Which tracked source files differ from HEAD.

    `git_commit` in every artifact is only meaningful if the tree was clean when the run
    happened. It was not, once: 003's 2Wiki arms were scored while `pool2wiki.py` — the module
    that builds their corpus — was still uncommitted, so the recorded commit did not contain the
    code that produced the numbers. It was caught by re-running an arm and comparing hashes,
    which is luck rather than a control, and nothing would have caught it on a run nobody thought
    to repeat.

    Reports `src/` separately because that is the code under test. A dirty `results/` or a stale
    note is not a reason to refuse to score.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    # NOT `.stdout.strip()`. Porcelain lines are `XY<space>path`, and an unstaged modification
    # begins with a SPACE, so stripping the whole output eats the first line's leading space and
    # slices that one path one character short -- silently, and only for the first entry. Caught
    # by this module's own test, which is the only reason it is not in the code.
    paths = [line[3:].strip() for line in out.split("\n") if line.strip()]
    return {
        "clean": not paths,
        "dirty_paths": sorted(paths),
        "dirty_source_paths": sorted(p for p in paths if p.startswith("src/")),
    }


def assert_scorable(state: dict | None = None) -> dict:
    """
    Refuse to score from a source tree that differs from HEAD.

    WHY THIS HALTS RATHER THAN WARNS. A scored run writes `git_commit` into an artifact that a
    reader is invited to check out and reproduce. If uncommitted source produced the numbers,
    that invitation is false, and it is false in the direction that looks fine: the artifact
    names a real commit, the reader gets different numbers, and nothing says why.

    Escape hatch, deliberately awkward and deliberately recorded. Exploratory runs are
    legitimate, so `RB_ALLOW_DIRTY_SCORED_RUN=1` proceeds — but the override and the exact dirty
    paths are written into the artifact, so a run made this way cannot later be mistaken for a
    clean one. An escape hatch that leaves no trace is just a disabled check.
    """
    state = state if state is not None else working_tree_state()
    if state["dirty_source_paths"] and not os.environ.get(ALLOW_DIRTY):
        listed = "\n  ".join(state["dirty_source_paths"])
        raise RuntimeError(
            "refusing to score: uncommitted changes under src/, so the git_commit recorded in "
            "the artifact would not contain the code that produced it.\n  " + listed +
            f"\n\nCommit them, or set {ALLOW_DIRTY}=1 for an exploratory run — the override and "
            "these paths are then recorded in the artifact."
        )
    return {
        **state,
        "override_used": bool(state["dirty_source_paths"] and os.environ.get(ALLOW_DIRTY)),
    }


def environment() -> dict:
    """
    Everything a stranger needs to tell an environment difference from a finding.

    Unlike 001's rb.run.environment(), this has no ripgrep field: not every rung
    shells out to it. Rung-specific detail (e.g. the dense encoder's pinned model
    name and revision) belongs in that rung's own manifest, folded into the summary
    by the caller — this function only covers what every rung shares.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    manifest_path = ROOT / "manifests" / "datasets.json"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": commit,
        # Whether that commit actually describes the code that ran. See assert_scorable.
        "working_tree": working_tree_state(),
        "dataset_checksums": json.loads(manifest_path.read_text()) if manifest_path.exists() else {},
    }


def run_rung(
    retriever: Retriever,
    dataset: str,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    out_dir: Path,
    top_k: int = 100,
    subsampled: bool = False,
    seed: int | None = None,
    extra_manifest: dict | None = None,
    # Controls that need the scored run. Called with (ranked_run, per_query) and
    # expected to return {name: {"passed": bool, ...}}. Any failure raises before
    # a single artifact is written.
    post_scoring_controls=None,
    # The trec_eval measure set. None means rb.metrics.MEASURES — what 001 and 002
    # published against, and what every existing caller keeps getting. Experiment 003
    # passes a larger set because its queries have exactly two gold documents each,
    # which makes Recall@2/@5 the informative cutoffs and nDCG@10 mostly empty rank
    # positions (protocols/003-graph-arm.md section 5). Scoped rather than global so
    # 001's and 002's committed artifacts keep their exact shape.
    measures: set[str] | None = None,
) -> dict:
    """
    Score one retriever on one dataset's (already query-subsampled) queries.

    Writes per_query.jsonl (retrieved doc ids in rank order, so any aggregate is
    recomputable without a rerun — same principle as 001) and summary.json into
    out_dir. Decoupled from rb.datasets.load() so it is testable against tiny
    in-memory fixtures without downloading a corpus.
    """
    # Before anything expensive, and before any artifact is written: the recorded commit must
    # describe the code about to run. See assert_scorable for why this halts rather than warns.
    assert_scorable()

    check = controls.gold_presence(corpus, qrels)
    if not check["passed"]:
        raise RuntimeError(f"control gold_presence failed: {check}. Nothing is scored on a broken harness.")

    out_dir.mkdir(parents=True, exist_ok=True)
    qids = sorted(queries)

    t0 = time.perf_counter()
    run_dict = retriever.retrieve(corpus, queries, top_k)
    elapsed = time.perf_counter() - t0

    # The contract says scores are strictly decreasing. The re-encoding below would
    # quietly repair a retriever that broke that, turning a bug into a plausible
    # ranking via the doc-id tie-break, so check first and fail loudly instead.
    for qid, docs in run_dict.items():
        scores = [sc for _, sc in sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))]
        if any(a <= b for a, b in zip(scores, scores[1:])):
            raise RuntimeError(
                f"{retriever.name} returned non-strictly-decreasing scores for query {qid}. "
                "The Retriever contract requires strict ordering; re-encoding would hide this."
            )

    # Score the retriever's ORDER, not its score magnitudes.
    #
    # pytrec_eval compares scores at reduced precision, so two documents separated
    # only by the lexical rung's 1e-9 tie-break epsilon are tied to it and get
    # resolved by its own internal rule rather than by the document-id tie-break
    # protocols/002-ladder.md pre-registers. Demonstrated directly: for scores
    # {a: 4.0, b: 4.0 - 1e-9} pytrec_eval returns the same nDCG@10 whichever of the
    # two is meant to be first, and only differs once the gap is a real rank apart.
    #
    # This is the same defect 001 had and fixed, arriving by a different route:
    # there the two-level sort was encoded as one number that tied, here the
    # separation is real but below the scorer's resolution. Re-encoding to strictly
    # decreasing integers here fixes it once for every rung rather than asking each
    # retriever to guess what magnitude survives the scorer.
    ranked_run = {
        qid: {
            d: float(len(docs) - i)
            for i, (d, _) in enumerate(sorted(docs.items(), key=lambda kv: (-kv[1], kv[0])))
        }
        for qid, docs in run_dict.items()
    }
    measures = measures or metrics.MEASURES
    per_query = metrics.score_ranked({q: qrels[q] for q in qids}, ranked_run, measures)

    # Controls that need the SCORED run, supplied by the caller, evaluated BEFORE
    # anything reaches disk.
    #
    # A reviewer demonstrated the hole this closes: run_rung used to write
    # per_query.jsonl unconditionally, and the dense arm's own controls (embedding
    # shuffle, self-retrieval) ran afterwards in run_dense. A transposed encoder
    # therefore failed its control AND left a complete, real-looking per_query.jsonl
    # on disk, which the hybrid and analysis rungs read later without ever consulting
    # whether the controls had passed. Those rungs are separate CLI invocations run on
    # different days, so "the run halted" was true of one process and false of the
    # pipeline. The amendment says every control halts the run; this is what makes
    # that true of the artifacts rather than of a process.
    post_controls: dict = {}
    if post_scoring_controls is not None:
        post_controls = post_scoring_controls(ranked_run, per_query)
        failed = [n for n, c in post_controls.items() if not c.get("passed")]
        if failed:
            raise RuntimeError(
                f"{retriever.name} failed post-scoring controls {failed} on {dataset}: "
                f"{ {n: post_controls[n] for n in failed} }. Nothing written."
            )

    with open(out_dir / "per_query.jsonl", "w", encoding="utf8") as f:
        for qid in qids:
            # Re-sort defensively rather than trust dict insertion order: a Retriever
            # implementation only has to satisfy the contract (strictly decreasing
            # scores), not build its dict in rank order.
            retrieved = sorted(run_dict.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            f.write(json.dumps({
                "query_id": qid,
                "gold": sorted(qrels[qid]),
                "retrieved": [d for d, _ in retrieved],
            }) + "\n")

    summary = {
        "dataset": dataset,
        "retriever": retriever.name,
        "queries_scored": len(qids),
        "queries_available": len(queries),
        "subsampled": subsampled,
        "seed": seed if subsampled else None,
        "corpus_documents": len(corpus),
        "controls": {"gold_presence": check, **post_controls},
        "ranked": {
            m: round(metrics.mean([per_query[q][m] for q in qids]), 4) for m in sorted(measures)
        },
        "cost": {"total_seconds": round(elapsed, 3), "usd": 0.0},
        "environment": environment(),
    }
    # Callable, not a dict, when the manifest can only be complete after retrieval.
    # The dense arm records a hash of the embedding matrix, which does not exist until
    # the encoder has run, so evaluating the manifest at call time silently dropped it.
    # Caught by checking the written artifact rather than trusting the commit.
    if callable(extra_manifest):
        extra_manifest = extra_manifest()
    if extra_manifest:
        summary["retriever_manifest"] = extra_manifest
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
