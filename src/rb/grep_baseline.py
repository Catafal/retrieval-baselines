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
    One ripgrep pass for the whole query.

    Returns line_number -> (distinct terms matched, total match count). `-o` prints
    each match on its own line as `lineno:matched-text`, which is exactly the two
    facts the ranker needs, at the cost of one pass rather than one pass per term.
    """
    if not terms:
        return {}
    cmd = ["rg", "-i", "-F", "-o", "-n", "--no-heading", "--no-filename"]
    if word_bounded:
        cmd.append("-w")
    for t in terms:
        cmd += ["-e", t]
    cmd.append(str(corpus_path))

    hits: dict[int, tuple[set[str], int]] = {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:
        lineno, _, matched = line.partition(":")
        if not _:
            continue
        idx = int(lineno)
        found, total = hits.setdefault(idx, (set(), 0))
        found.add(matched.strip().lower())
        hits[idx] = (found, total + 1)
    proc.wait()
    # rg exits 1 when nothing matched, which is not an error here.
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ripgrep failed with code {proc.returncode}")
    return {i: (len(f), t) for i, (f, t) in hits.items()}


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
    return [(doc_ids[i - 1], d + t / (t + 1)) for i, (d, t) in ordered]


def run_query(query: str, corpus_path: Path, doc_ids: list[str], top_k: int = 100, word_bounded: bool = True):
    """Returns (ranked results, unranked set size, wall-clock seconds)."""
    terms = tokenize(query)
    t0 = time.perf_counter()
    hits = search(terms, corpus_path, word_bounded=word_bounded)
    elapsed = time.perf_counter() - t0
    return rank(hits, doc_ids, top_k), len(hits), elapsed, terms
