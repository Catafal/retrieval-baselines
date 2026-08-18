"""
grep-baseline-v1 — the system under test for Experiment 001.

Frozen by protocol.md before any scored run. Every choice below is a place this
experiment could have been rigged, so each one is stated rather than buried:

  1. Query -> terms: lowercase, strip non-alphanumerics, drop the frozen stopword
     list, deduplicate. No stemming, no expansion, no rewriting.
  2. Matching: `rg -i -w -F`, i.e. case-insensitive, WORD-BOUNDED, literal.
     Word boundaries rather than raw substring: unbounded substring matching makes
     "insulin" match "insulinoma" and inflates the candidate set without adding
     signal. The unbounded variant is reported as a sensitivity check, not headline.
  3. Ranking: grep returns a SET. Metrics need an order. The order used is
     coordination-level matching — count of distinct query terms present, tie-broken
     by total match count, then by document id so the result is deterministic.
     This is deliberately the dumbest set-to-order rule that exists. It is not BM25
     and no number produced here may be presented as if it were.

The corpus is materialised as one document per line so ripgrep line numbers map
straight onto document ids. That invariant is asserted, not assumed.
"""

import re
import subprocess
from collections import Counter
import time
from pathlib import Path

from rb.stopwords import STOPWORDS

TOKEN_RE = re.compile(r"[^a-z0-9]+")


def tokenize(query: str) -> list[str]:
    """Query -> deduplicated non-stopword terms, order preserved for reproducibility."""
    seen, terms = set(), []
    for tok in TOKEN_RE.split(query.lower()):
        if tok and tok not in STOPWORDS and tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return terms


def materialise(corpus: dict[str, str], path: Path) -> list[str]:
    """
    Write one document per line and return doc ids in line order.

    Newlines and tabs inside document text are collapsed to spaces. Without that,
    one document would occupy several lines and every line-number-to-doc-id lookup
    after it would be off by a silent, growing offset.
    """
    doc_ids = list(corpus.keys())
    if not path.exists():
        with open(path, "w", encoding="utf8") as f:
            for did in doc_ids:
                f.write(re.sub(r"\s+", " ", corpus[did]).strip() + "\n")
    n_lines = sum(1 for _ in open(path, encoding="utf8"))
    if n_lines != len(doc_ids):
        raise RuntimeError(
            f"corpus file has {n_lines} lines for {len(doc_ids)} documents — "
            "the one-doc-per-line invariant is broken, every rank would be wrong"
        )
    return doc_ids


def search(terms: list[str], corpus_path: Path, word_bounded: bool = True) -> dict[int, tuple[int, int]]:
    """
    Line number -> (distinct terms matched, total match count).

    One ripgrep pass PER TERM rather than one pass for the whole query. The
    single-pass version needs to know which term produced each match, which forces a
    per-line Python set and allocates one object per match — on HotpotQA that is tens
    of millions of allocations per query and the run never finishes. Per term, the
    output is just line numbers, so both counts fall out of `Counter` at C speed.

    Identical results either way. Only the cost differs, and cost is a reported metric,
    so it is worth stating that the number published is this implementation's, not the
    slowest one that produces the same answer.
    """
    if not terms:
        return {}

    distinct: Counter[int] = Counter()
    total: Counter[int] = Counter()

    for term in terms:
        cmd = ["rg", "-i", "-F", "-o", "-n", "--no-heading", "--no-filename"]
        if word_bounded:
            cmd.append("-w")
        cmd += ["-e", term, str(corpus_path)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        # rg exits 1 when nothing matched, which is not an error here.
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"ripgrep failed with code {proc.returncode} on term {term!r}")
        if not proc.stdout:
            continue

        lines = [int(row[: row.index(":")]) for row in proc.stdout.splitlines()]
        total.update(lines)
        distinct.update(set(lines))  # a document counts once per term, however often it matched

    return {i: (distinct[i], total[i]) for i in distinct}


def rank(hits: dict[int, tuple[int, int]], doc_ids: list[str], top_k: int = 100) -> list[tuple[str, float]]:
    """
    Set -> ranked list. Ties broken by document id so two runs agree exactly.

    The score returned is `distinct + total/(total+1)`, a monotone encoding of the
    two-level sort into one number, because trec_eval takes a scalar. Ordering is
    identical to sorting on the pair; the value itself carries no meaning.
    """
    ordered = sorted(
        hits.items(),
        key=lambda kv: (-kv[1][0], -kv[1][1], doc_ids[kv[0] - 1]),
    )[:top_k]
    # Score STRICTLY decreasing by final rank position, so the order trec_eval scores is
    # exactly the pre-registered order. An earlier version emitted
    # `distinct + total/(total+1)`, which ties heavily: in one SciFact query's top 100
    # there were 16 distinct scores across 100 documents, so trec_eval was breaking 84
    # ties by its own internal rule rather than by document id as the protocol states.
    # The value carries no meaning; only the ordering does.
    return [(doc_ids[i - 1], float(len(ordered) - pos)) for pos, (i, _) in enumerate(ordered)]


def run_query(query: str, corpus_path: Path, doc_ids: list[str], top_k: int = 100, word_bounded: bool = True):
    """
    Returns (ranked top-k, full hit map, wall-clock seconds, terms).

    The full hit map is returned rather than just its size because set recall is
    defined over grep's ENTIRE output. Computing it from the truncated top-k instead
    would make "recall over the full set" silently identical to Recall@k — which is
    exactly the bug this signature exists to prevent.
    """
    terms = tokenize(query)
    t0 = time.perf_counter()
    hits = search(terms, corpus_path, word_bounded=word_bounded)
    elapsed = time.perf_counter() - t0
    return rank(hits, doc_ids, top_k), hits, elapsed, terms
