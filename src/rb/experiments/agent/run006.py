"""
Experiment 006's scored run — protocol 006.

Everything that can be computed without a model is computed BEFORE the scored loop opens:
the question sample, the corpus directory, the BM25 and dense rankings, and the graph's
recalled facts. Nothing inside the loop decides what an arm receives. That is what makes the
run resumable, and it is also what stops a retrieval failure mid-run from silently changing
the arm definition partway through.

The run refuses to start on a dirty source tree, for the reason retriever.assert_scorable
records: 003's 2Wiki arms were once scored while the module that built their corpus was still
uncommitted, so the recorded commit did not contain the code that produced the numbers.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from rb import retriever
from rb.experiments.agent import driver, graphmem, runner
from rb.experiments.agent.corpus import manifest, sample, write_corpus_dir

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "006"
CORPUS_DIR = ROOT / "data" / "006-corpus"
N, SAMPLE_SEED, SHUFFLE_SEED = 100, 20260820, 20260828


def prepare() -> tuple[list, dict, dict, dict]:
    qs, pool = sample(N, SAMPLE_SEED)
    write_corpus_dir(pool, CORPUS_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "questions.json").write_text(
        json.dumps([{"id": q.id, "question": q.question, "answer": q.answer,
                     "gold": list(q.gold)} for q in qs], indent=2) + "\n")
    (OUT / "sample-manifest.json").write_text(
        json.dumps(manifest(qs, pool, SAMPLE_SEED), indent=2) + "\n")

    print("precomputing retrieval...", flush=True)
    retrieved = driver.retrieve_all(qs, pool)
    print("recalling graph facts...", flush=True)
    facts = driver.recall_all(qs, OUT / "graph.db")
    seeded = sum(1 for v in facts.values() if not v.startswith("memory: no facts"))
    print(f"graph seeded on {seeded}/{len(qs)} questions", flush=True)
    (OUT / "graph-facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    (OUT / "retrieved.json").write_text(json.dumps(retrieved, indent=2) + "\n")
    return qs, pool, retrieved, facts


def main(dry: bool = False) -> None:
    state = retriever.assert_scorable()
    qs, pool, retrieved, facts = prepare()
    jobs = driver.shuffle_jobs(
        driver.build_jobs(qs, pool, CORPUS_DIR, retrieved, facts), SHUFFLE_SEED)
    print(f"{len(jobs)} jobs = {len(qs)} questions x 6 arms x 3 tiers", flush=True)
    if dry:
        for j in jobs[:3]:
            print(json.dumps({k: (v[:160] if isinstance(v, str) else v)
                              for k, v in j.items()}, indent=2))
        return

    (OUT / "run-manifest.json").write_text(json.dumps({
        "cli_version": runner.cli_version(), "flags": runner.BASE_FLAGS,
        "n_questions": len(qs), "n_jobs": len(jobs), "sample_seed": SAMPLE_SEED,
        "shuffle_seed": SHUFFLE_SEED, "grep_max_turns": driver.GREP_MAX_TURNS,
        "budget_tokens": 400, "git": state, "environment": retriever.environment(),
    }, indent=2, default=str) + "\n")

    t0 = time.time()
    def prog(c, i, n):
        if i % 25 == 0 or i == n:
            print(f"  {i}/{n}  {time.time()-t0:5.0f}s  last={c.arm}/{c.model}/{c.outcome}",
                  flush=True)

    calls = runner.run_all(jobs, OUT / "calls.jsonl", workers=24, on_done=prog)
    print(f"ran {len(calls)} calls in {time.time()-t0:.0f}s", flush=True)

    gold = {q.id: q.answer for q in qs}
    scored = driver.load_scored(OUT / "calls.jsonl", gold)
    (OUT / "scored.json").write_text(json.dumps(scored, indent=2) + "\n")
    print(f"scored {len(scored)} calls -> results/006/scored.json", flush=True)


if __name__ == "__main__":
    import sys
    main(dry="--dry" in sys.argv)
