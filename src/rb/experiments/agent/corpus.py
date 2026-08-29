"""
Experiment 006's corpus and question sample.

006 must run on a SMALL corpus, which rules out 003's 66,581-passage pool. The construction
here takes the sampled questions' own HotpotQA distractor contexts and unions them into one
shared pool: every question's 2 gold passages plus its 8 distractors, deduplicated by title.

Why a union rather than per-question contexts. If each question were answered against only
its own 10 passages, retrieval would be nearly trivial and the arms would not separate --
the experiment would measure reading comprehension, not retrieval. Unioning across the
sample makes every other question's distractors into this question's haystack, so the pool
grows with n while staying small enough to satisfy the corpus constraint.

Everything here is deterministic given (n, seed, filters), so the sample can be frozen by
tagging the protocol and regenerated exactly by any reader.
"""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
PARQUET = ROOT / "data" / "hotpotqa_distractor_validation.parquet"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    gold: tuple[str, ...]      # the supporting-fact titles, always 2 for bridge questions
    level: str
    type: str


def _load_rows() -> list[dict]:
    f = pq.ParquetFile(PARQUET)
    rows: list[dict] = []
    for b in f.iter_batches(batch_size=2000):
        rows.extend(b.to_pylist())
    return rows


def sample(n: int, seed: int, qtype: str = "bridge", level: str = "hard"
           ) -> tuple[list[Question], dict[str, str]]:
    """Draw n questions and build the shared pool their contexts imply.

    Returns (questions, pool) where pool maps title -> passage text. Titles are HotpotQA's
    own document identifiers and are what supporting_facts references, so they are the
    document ids throughout 006 exactly as they were in 003.
    """
    rows = [r for r in _load_rows() if r["type"] == qtype and r["level"] == level]
    rows.sort(key=lambda r: r["id"])          # parquet order is not guaranteed stable
    rng = random.Random(seed)
    picked = rng.sample(rows, n)
    picked.sort(key=lambda r: r["id"])

    questions, pool = [], {}
    for r in picked:
        gold = tuple(dict.fromkeys(r["supporting_facts"]["title"]))
        questions.append(Question(id=r["id"], question=r["question"], answer=r["answer"],
                                  gold=gold, level=r["level"], type=r["type"]))
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            # First writer wins. Titles are unique documents in HotpotQA, so a title seen
            # under two questions is the same document and the texts are identical.
            pool.setdefault(title, " ".join(sents).strip())
    return questions, pool


def slug(title: str) -> str:
    """A filename for a pool document. Reversible enough to audit, safe on a case-insensitive
    filesystem: a hash suffix keeps 'Mercury (planet)' and 'Mercury (element)' distinct."""
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "doc"
    return f"{s}-{hashlib.sha1(title.encode()).hexdigest()[:8]}"


def write_corpus_dir(pool: dict[str, str], path: Path) -> dict[str, str]:
    """Materialise the pool as files for the grep arm, returning title -> relative filename.

    The document's title is written as the first line of each file as well as into its
    filename. This is a registered design decision, not an accident: real corpora have named
    files, the prior art being replicated has named files, and hiding titles would make the
    grep arm artificially hard. Because it cuts in the grep arm's favour, it cannot inflate
    the graph arm's result.
    """
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.md"):
        old.unlink()
    names = {}
    for title, text in sorted(pool.items()):
        name = slug(title) + ".md"
        (path / name).write_text(f"# {title}\n\n{text}\n")
        names[title] = name
    return names


def manifest(questions: list[Question], pool: dict[str, str], seed: int) -> dict:
    """A content hash of the frozen sample, so a reader can prove they rebuilt the same one."""
    qh = hashlib.sha256("\n".join(q.id for q in questions).encode()).hexdigest()
    ph = hashlib.sha256("\n".join(f"{t}\t{len(v)}" for t, v in sorted(pool.items()))
                        .encode()).hexdigest()
    return {"n_questions": len(questions), "n_pool_docs": len(pool), "seed": seed,
            "questions_sha256": qh, "pool_sha256": ph,
            "pool_chars": sum(len(v) for v in pool.values()),
            "gold_always_2": all(len(q.gold) == 2 for q in questions)}


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    qs, pool = sample(n, 20260820)
    print(json.dumps(manifest(qs, pool, 20260820), indent=2))
