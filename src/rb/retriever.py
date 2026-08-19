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
) -> dict:
    """
    Score one retriever on one dataset's (already query-subsampled) queries.

    Writes per_query.jsonl (retrieved doc ids in rank order, so any aggregate is
    recomputable without a rerun — same principle as 001) and summary.json into
    out_dir. Decoupled from rb.datasets.load() so it is testable against tiny
    in-memory fixtures without downloading a corpus.
    """
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
    per_query = metrics.score_ranked({q: qrels[q] for q in qids}, ranked_run)

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
            m: round(metrics.mean([per_query[q][m] for q in qids]), 4) for m in sorted(metrics.MEASURES)
        },
        "cost": {"total_seconds": round(elapsed, 3), "usd": 0.0},
        "environment": environment(),
    }
    if extra_manifest:
        summary["retriever_manifest"] = extra_manifest
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
