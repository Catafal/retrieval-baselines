"""
Experiment 003's second candidate set: the 2WikiMultiHopQA distractor pool.

Registered in protocols/003-amendment-6-second-corpus.md and tagged before any arm was scored on
it. §2 of the original protocol already named 2Wiki as the corpus where the same HippoRAG graph
behaves oppositely to HotpotQA (59.2 -> 70.7 there, 64.7 -> 60.5 here), so this is the completion
of the comparison rather than a second attempt at it.

ONE VARIABLE MOVES. Same extractor, whitelist, linking, damping, seed, B, margin and scoring code
as the HotpotQA run. Experiment 004 moves the extractor; this moves the corpus; neither moves both.

TWO CONSTRUCTION DECISIONS, both frozen in the amendment before scoring, because either could be
used to move a result:

  1. Only the 9,825 questions with EXACTLY TWO gold documents are scored. 2Wiki's 2,751
     `bridge_comparison` questions have four, and R@2 cannot exceed 0.50 on those, so pooling the
     groups would make the primary measure mean something different here than on HotpotQA and the
     crossover would compare two different quantities.
  2. 1,242 titles carry more than one text -- whitespace and tokenisation artifacts, median
     pairwise similarity 0.994. The most frequent variant wins, ties broken lexicographically.

WHY IDS ARE MINTED. There is no BEIR release of 2Wiki, so unlike HotpotQA there are no published
document ids to inherit. Ids are assigned from sorted title order, which is deterministic and
independent of how the dataset happened to be iterated. The title map is kept separately because
`coverage` needs real titles: on this corpus, as on HotpotQA, a document title IS an entity.
"""

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "manifests" / "datasets.json"

URL = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet"
FILE = "2wiki_dev.parquet"

# Frozen in protocols/003-amendment-6-second-corpus.md before any scored run. Checked against,
# never derived from a run: a count that recomputes itself from the outcome cannot fail.
EXPECTED_QUESTIONS = 9825
EXPECTED_PASSAGES = 43487
EXPECTED_TITLE_SLOTS = 98250
EXPECTED_VARIANT_TITLES = 1242
EXPECTED_GOLD_TITLES = 16468

GOLD_PER_QUESTION = 2  # the restriction decision 1 records


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    """Fetch the dev split, recording its hash. Same discipline as rb.datasets and pool.py:
    a hash that changes under us invalidates every number published against it."""
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / FILE
    if not path.exists():
        with requests.get(URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    digest = _sha256(path)
    key = "2wiki_dev"
    if key in manifest and manifest[key]["sha256"] != digest:
        raise RuntimeError(
            f"{FILE} hash changed: manifest has {manifest[key]['sha256']}, file is {digest}. "
            "Published results reference the manifest hash."
        )
    manifest[key] = {"sha256": digest, "bytes": path.stat().st_size}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_rows() -> list[dict]:
    """The scored question set: exactly-two-gold questions only (decision 1).

    Columns arrive as JSON strings rather than nested arrays, so they are parsed here in one
    place rather than at each call site.
    """
    import pyarrow.parquet as pq

    rows = []
    for r in pq.read_table(download()).to_pylist():
        facts = json.loads(r["supporting_facts"])
        gold = {f[0] for f in facts}
        if len(gold) != GOLD_PER_QUESTION:
            continue
        rows.append({"id": r["_id"], "question": r["question"], "type": r["type"],
                     "context": json.loads(r["context"]), "gold": gold})
    return rows


def build(rows: list[dict] | None = None):
    """
    Returns (corpus, titles, queries, qrels).

    `corpus` maps minted doc_id -> passage text, `titles` maps doc_id -> title, matching the shape
    `rb.datasets.load_corpus` / `load_titles` hand the rest of the pipeline, so every downstream
    module runs unchanged.
    """
    rows = rows if rows is not None else load_rows()

    # Majority variant per title (decision 2), computed before any id is minted so the choice
    # cannot depend on id assignment.
    variants: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        for title, sents in r["context"]:
            variants[title][" ".join(sents)] += 1
    text_for = {t: sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                for t, v in variants.items()}

    # Ids from SORTED title order: deterministic, and independent of how the dataset was iterated.
    doc_id = {t: f"2wiki-{i:06d}" for i, t in enumerate(sorted(text_for))}
    corpus = {doc_id[t]: text_for[t] for t in text_for}
    titles = {doc_id[t]: t for t in text_for}

    queries = {r["id"]: r["question"] for r in rows}
    qrels = {r["id"]: {doc_id[t]: 1 for t in r["gold"]} for r in rows}
    return corpus, titles, queries, qrels


def construction_counts(rows: list[dict] | None = None) -> dict:
    """
    The §9-style control's inputs, MEASURED — every one, without raising.

    Same shape and the same reasoning as `pool.construction_counts`: a control whose fields are
    asserted rather than measured is the defect amendment 4 was written to close, and it is not
    going to be reintroduced on a second corpus.
    """
    rows = rows if rows is not None else load_rows()
    variants: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    slots = 0
    for r in rows:
        for title, sents in r["context"]:
            slots += 1
            variants[title][" ".join(sents)] += 1
    gold_titles = set()
    for r in rows:
        gold_titles |= r["gold"]
    pooled = set(variants)
    matched = sum(1 for r in rows if r["gold"] <= pooled)
    return {
        "questions": len(rows),
        "passages": len(pooled),
        "title_slots": slots,
        "variant_titles": sum(1 for v in variants.values() if len(v) > 1),
        "gold_titles": len(gold_titles),
        "gold_titles_matched": matched,
        "gold_queries": len(rows),
    }


def control(counts: dict | None = None) -> dict:
    """Halts the run on any mismatch against the frozen counts."""
    c = counts if counts is not None else construction_counts()
    checks = {
        "questions": (c["questions"], EXPECTED_QUESTIONS),
        "passages": (c["passages"], EXPECTED_PASSAGES),
        "title_slots": (c["title_slots"], EXPECTED_TITLE_SLOTS),
        "variant_titles": (c["variant_titles"], EXPECTED_VARIANT_TITLES),
        "gold_titles": (c["gold_titles"], EXPECTED_GOLD_TITLES),
        "gold_titles_matched": (c["gold_titles_matched"], c["gold_queries"]),
    }
    mismatched = {k: {"got": g, "expected": e} for k, (g, e) in checks.items() if g != e}
    return {**c, "mismatched": mismatched, "passed": not mismatched}
