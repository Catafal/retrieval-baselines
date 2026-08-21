"""
The §9 pool control, as a standalone artifact — protocols/003-amendment-4-pool-control.md.

    make reproduce-003-pool-control

WHY A NEW FILE RATHER THAN A REWRITE OF THE FIVE PUBLISHED SUMMARIES. Each arm's
`results/003/pool/<arm>/summary.json` carries a `pool_control` block written by the pre-amendment
call, in which `unresolved`, `collisions` and `gold_titles_matched` were asserted rather than
measured. The obvious repair — a command that recomputes the block and rewrites it in place — was
considered and REJECTED. `results/003/pool/graph-defective/` is one of those five, and its own
README says it is "kept, not deleted... a reader must be able to see what was published". A loop
over the five would launder a deliberately-preserved defective run's control block, and mutating
published artifacts is the wrong instrument regardless of how careful the loop is.

So the corrected control is published HERE, beside the old ones rather than over them. The
amendment note records both blocks side by side, and a reader can diff them without having to
reconstruct what changed.

Nothing is scored by this module. It only measures the pool.
"""

import json
import time
from pathlib import Path

from rb import controls, datasets
from rb.experiments.graph import pool

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"


def measure() -> dict:
    """Every field of the §9 control, measured. See pool.construction_counts for what changed."""
    ctx = pool.load_distractor_context()
    corpus = datasets.load_corpus("hotpotqa")
    titles = datasets.load_titles("hotpotqa")
    qrels = datasets.load_qrels("hotpotqa")

    counts = pool.construction_counts(titles, ctx, qrels)
    pool_corpus, _ = pool.build(corpus, titles, ctx)
    return controls.pool_construction(
        questions=counts["questions"], passages=len(pool_corpus),
        title_slots=counts["title_slots"],
        unresolved=counts["unresolved"], collisions=counts["collisions"],
        gold_titles_matched=counts["gold_titles_matched"],
        gold_queries=counts["gold_queries"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = measure()
    payload = {
        "status": (
            "POOL CONSTRUCTION CONTROL, protocols/003-graph-arm.md section 9, as amended by "
            "003-amendment-4. Supersedes the `pool_control` block inside each arm's summary.json, "
            "which was written before unresolved/collisions/gold_titles_matched were measured. "
            "Those summaries are left untouched on purpose; see this module's docstring."
        ),
        "control": result,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT / "pool-control.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
