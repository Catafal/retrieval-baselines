"""
The cross-arm comparison table — both corpora.

WHY THIS MODULE EXISTS. `results/003/arms-summary.json` was published with **no producer anywhere
in the repository**, and it carried `graph.recall_2 = 0.2132` — the RETRACTED pre-PPR-fix number
from the defective walk that `003-amendment-3` corrected. The live figure is 0.2148. So a
published artifact stated a retracted number, with no command that could regenerate or refute it.

Three audits closed this exact defect class (NB-25 found `run_controls.py` orphaned and two blocks
with no producer; NB-26 found the pool control asserting rather than measuring) and all three
missed this file, because each searched the graph arm's source tree and this artifact is produced
by nothing that lives there. It is defect fifteen, and it is recorded in results/003/corrections.md.

Reads each arm's committed summary.json rather than re-running retrieval, the same way the
analyses re-score from committed per-query output: the table must be derivable from what was
published, or a reader cannot check it.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "003"

CORPORA = {
    "hotpotqa": {
        "dir": OUT / "pool",
        "description": "HotpotQA distractor pool, 66,581 passages, 7,405 queries",
    },
    "2wiki": {
        "dir": OUT / "2wiki",
        "description": ("2WikiMultiHopQA distractor pool, 43,487 passages, 9,825 two-gold "
                        "questions (amendment 6)"),
    },
}
# graph-glm is experiment 004's arm: the same graph and the same walk with the extractor
# swapped. Listed here so the cross-arm table carries it beside the arm it replaces rather than
# living in a second file that nothing regenerates — which is the defect this module exists for.
ARMS = ("bm25", "graph", "graph-glm", "dense-minilm", "dense-bge")
MEASURES = ("recall_2", "recall_5", "ndcg_cut_10", "recall_10", "recall_100")


def arm_row(path: Path) -> dict | None:
    """One arm's published figures, or None when that arm was not run on that corpus."""
    f = path / "summary.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    row = {m: d["ranked"][m] for m in MEASURES if m in d["ranked"]}
    row["queries_scored"] = d.get("queries_scored")

    # "Retrieves nothing" is a headline figure for the graph arm and it was NOT derivable from
    # any committed JSON — only from per_query.jsonl. The notebook's oracle table therefore
    # carried the graph arm's rate as a hard-coded 17.7% inside a block marked as generated,
    # which is worse than a number in prose: a reader sees it in a generated table and assumes it
    # traces to an artifact. It happened to be correct, which is the whole problem — right by
    # luck rather than by construction, and nothing would have caught it drifting. Counted here so
    # the table can read it. See results/003/corrections.md.
    pq = path / "per_query.jsonl"
    if pq.exists():
        empty = total = 0
        with pq.open() as fh:
            for line in fh:
                total += 1
                if not json.loads(line).get("retrieved"):
                    empty += 1
        row["empty_results"] = empty
        row["empty_rate"] = round(empty / total, 4) if total else None
    return row


def build() -> dict:
    corpora = {}
    for name, spec in CORPORA.items():
        arms = {a: arm_row(spec["dir"] / a) for a in ARMS}
        corpora[name] = {"description": spec["description"],
                         "arms": {a: r for a, r in arms.items() if r is not None}}
    return {
        "status": (
            "Cross-arm comparison, both corpora. Derived from each arm's committed summary.json; "
            "no retrieval is re-run. Supersedes the previous arms-summary.json, which had no "
            "producer and carried the retracted pre-PPR-fix graph figure (0.2132). See "
            "results/003/corrections.md."
        ),
        "corpora": corpora,
        # The graph arm's DEFECTIVE run is deliberately excluded from the table above and named
        # here instead, so it cannot be read as a fifth arm.
        "superseded_runs": {
            "graph-defective": "results/003/pool/graph-defective/ — pre-amendment-3 walk, kept "
                               "deliberately; see that directory's README.",
        },
    }


def main() -> None:
    (OUT / "arms-summary.json").write_text(json.dumps(build(), indent=2) + "\n")
    d = build()
    for corpus, block in d["corpora"].items():
        print(f"\n{corpus}: {block['description']}")
        for arm, row in sorted(block["arms"].items(), key=lambda kv: -kv[1]["recall_2"]):
            print(f"   {arm:14s} R@2 {row['recall_2']:.4f}  R@5 {row['recall_5']:.4f}  "
                  f"nDCG@10 {row['ndcg_cut_10']:.4f}")


if __name__ == "__main__":
    main()
