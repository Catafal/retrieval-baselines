"""
The lexical rung — one BM25-shaped scorer with three independent boolean switches.

All eight configurations come from this one function, not from a third-party
library or three separate implementations. Rung-to-rung differences must be
attributable to the mechanism toggled; mixing implementations across rungs would
confound every comparison with tokenisation and parameter differences instead.

The three switches:
  idf            — inverse document frequency weighting, on or off
  tf_saturation  — the k1 term (diminishing returns on repeated matches), on or off
  length_norm    — the b term (penalise long documents), on or off

k1 and b are fixed at the standard 1.2 / 0.75 (Robertson & Zaragoza) and are never
tuned: tuning them against the evaluation set would be fitting on the test data,
and tuning them on a dev split is a different experiment about parameter
sensitivity, not this one.

All-off is raw term-frequency sum. All-on is full BM25 — see full_bm25() below,
which is the canonical constructor the closure control checks against 001's
externally anchored BM25.

PERFORMANCE. LexicalRetriever.retrieve() used to scan every document for every
query term in pure Python (O(|corpus| * |query terms|) per query) and rebuild its
term-frequency/df bookkeeping from scratch on every call. Fine for SciFact
(~5.2k docs), unusable for HotpotQA (~5.2M docs) run across all eight configs of
the factorial. It now builds a sparse inverted index once (build_index()) and
scoring only touches documents that contain at least one query term — the
postings list for a term IS the set of candidate documents, so a query with rare
terms only visits a handful of rows out of millions. The index is
config-independent (postings, document lengths, document frequencies, average
length never depend on idf/tf_saturation/length_norm), so run_lexical_factorial
builds it once per dataset and every one of the eight configs reuses it — see
LexicalRetriever.index below and rb.experiments.ladder.run.run_lexical_factorial.

_ReferenceLexicalRetriever below is the original naive implementation, kept
verbatim and never called from the fast path. It exists solely so
tests/test_lexical_equivalence.py can assert the fast path returns byte-for-byte
identical run dicts (within float tolerance) — this refactor is not allowed to
change a single score, and the only convincing proof of that is running both
implementations and diffing their output, not reasoning about the algebra.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

TOKEN_RE = re.compile(r"[^a-z0-9]+")

K1 = 1.2
B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics. No stopword removal and no dedup —
    unlike 001's grep tokenizer, BM25 needs raw term frequency, not just presence."""
    return [t for t in TOKEN_RE.split(text.lower()) if t]


def _idf(df_t: int, n: int) -> float:
    """Robertson-Walker positive IDF, the same variant bm25s's "lucene" method uses
    (see bm25_control.py) — kept consistent so the closure control compares like
    with like rather than two different IDF conventions."""
    return math.log((n - df_t + 0.5) / (df_t + 0.5) + 1)


@dataclass(frozen=True)
class LexicalIndex:
    """
    The config-independent half of BM25-shaped scoring: postings, document
    frequencies, document lengths, average length. None of this depends on
    idf/tf_saturation/length_norm, so it is built once per corpus and shared
    read-only across every LexicalRetriever config scored on that corpus.

    `matrix` is CSC (compressed sparse column): scoring a query needs, for each
    query term, "every document containing it and its term frequency there" —
    exactly a matrix column's nonzero entries. CSC gives O(nnz in that column)
    access to that via matrix.indptr slicing, with no per-query conversion. CSR
    (row-sliced) would be the wrong layout here since queries look things up by
    term (column), not by document (row).
    """

    doc_ids: tuple[str, ...]
    doc_len: np.ndarray  # int64, aligned to doc_ids: doc_len[i] is len(doc_ids[i])
    avgdl: float
    n: int
    vocab: dict[str, int]  # term -> column index into `matrix`
    df: np.ndarray  # int64, aligned to vocab's column indices: document frequency per term
    matrix: sp.csc_matrix  # (n_docs, n_terms) raw term counts


def build_index(corpus: dict[str, str]) -> LexicalIndex:
    """
    One tokenisation pass over the corpus, building the sparse term-count matrix
    and the bookkeeping (doc lengths, document frequencies, average length) every
    config of the factorial needs. Config-independent — call this once per
    dataset, not once per config.
    """
    doc_ids = tuple(sorted(corpus))  # sorted, not insertion order: deterministic regardless of dict construction
    n = len(doc_ids)
    doc_len = np.zeros(n, dtype=np.int64)
    vocab: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for i, d in enumerate(doc_ids):
        counts = Counter(_tokenize(corpus[d]))
        doc_len[i] = sum(counts.values())
        for term, tf in counts.items():
            col = vocab.setdefault(term, len(vocab))
            rows.append(i)
            cols.append(col)
            data.append(tf)

    n_terms = len(vocab)
    # dtype=float64: term counts feed straight into float arithmetic downstream
    # (tf/norm etc.), so building the matrix as floats avoids an int->float cast
    # of every nonzero on every query.
    matrix = sp.csc_matrix((data, (rows, cols)), shape=(n, n_terms), dtype=np.float64)
    # nnz per column == number of distinct documents containing that term, i.e.
    # exactly document frequency (each doc contributes at most one entry per
    # term, since `counts` is a Counter keyed by term).
    df = np.diff(matrix.indptr)
    avgdl = float(doc_len.mean()) if n else 0.0
    return LexicalIndex(doc_ids=doc_ids, doc_len=doc_len, avgdl=avgdl, n=n, vocab=vocab, df=df, matrix=matrix)


@dataclass(frozen=True)
class LexicalRetriever:
    idf: bool = True
    tf_saturation: bool = True
    length_norm: bool = True
    # Shared LexicalIndex, injected by a caller (run_lexical_factorial) that
    # already built one for this corpus. compare=False/repr=False: two configs
    # with the same three switches must stay equal and interchangeable as dict
    # keys (ndcg_by_config[full_bm25()] etc.) regardless of which index object,
    # if any, happens to be attached — the index is a performance detail, not
    # part of a config's identity.
    index: LexicalIndex | None = field(default=None, compare=False, repr=False)

    @property
    def name(self) -> str:
        # Lists only the active mechanisms, so all eight configs get distinct,
        # readable names in results rather than being told apart only by flags.
        on = [n for n, flag in
              (("idf", self.idf), ("tf_sat", self.tf_saturation), ("len_norm", self.length_norm))
              if flag]
        return f"lexical({','.join(on) or 'none'})"

    def retrieve(
        self, corpus: dict[str, str], queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        index = self.index if self.index is not None else build_index(corpus)
        run: dict[str, dict[str, float]] = {}
        for qid in sorted(queries):
            terms = set(_tokenize(queries[qid]))  # BM25's outer sum is over distinct query terms
            run[qid] = self._score_query(terms, index, top_k)
        return run

    def _score_query(self, terms: set[str], index: LexicalIndex, top_k: int) -> dict[str, float]:
        """
        Score one query against the shared index, touching only documents that
        contain at least one query term (a term absent from the corpus
        contributes nothing and is skipped before ever reaching the matrix).
        """
        cols = [index.vocab[t] for t in terms if t in index.vocab]
        if not cols:
            return {}

        touched_rows = []
        touched_scores = []
        for col in cols:
            start, end = index.matrix.indptr[col], index.matrix.indptr[col + 1]
            rows = index.matrix.indices[start:end]  # doc row indices containing this term
            tfs = index.matrix.data[start:end]  # their term frequencies
            df_t = index.df[col]
            touched_rows.append(rows)
            touched_scores.append(self._term_scores(tfs, index.doc_len[rows], index.avgdl, df_t, index.n))

        all_rows = np.concatenate(touched_rows)
        all_scores = np.concatenate(touched_scores)
        # A document can be touched by more than one query term (once per
        # postings list it appears in); collapse duplicates by summing, which is
        # exactly BM25's outer sum over query terms. np.unique + np.add.at is the
        # vectorised equivalent of a dict accumulator, without a Python-level
        # loop over every touched (doc, term) pair.
        unique_rows, inverse = np.unique(all_rows, return_inverse=True)
        totals = np.zeros(len(unique_rows), dtype=np.float64)
        np.add.at(totals, inverse, all_scores)

        ranked_idx = np.argsort(-totals, kind="stable")
        result: dict[str, float] = {}
        rank = 0
        for idx in ranked_idx:
            if rank >= top_k:
                break
            score = totals[idx]
            if score <= 0:
                continue
            doc_id = index.doc_ids[unique_rows[idx]]
            result[doc_id] = score
            rank += 1
        # Sort by (-score, doc_id) for a deterministic tie-break by document id
        # (np.argsort's tie order depends on row index, not doc id), then apply
        # the same 1e-9-per-rank epsilon the reference implementation uses so
        # ties satisfy the contract's "strictly decreasing" requirement without
        # perturbing any real score comparison.
        ordered = sorted(result.items(), key=lambda kv: (-kv[1], kv[0]))
        return {d: raw - i * 1e-9 for i, (d, raw) in enumerate(ordered)}

    def _term_scores(self, tf: np.ndarray, doc_len: np.ndarray, avgdl: float, df_t: int, n: int) -> np.ndarray:
        """Vectorised form of _term_score() below, applied to every document a
        single query term touches at once. Kept algebraically identical to
        _term_score — same weight/norm/tf_component formulas — so the two can
        only diverge on floating-point summation order, not on the formula."""
        weight = _idf(df_t, n) if self.idf else 1.0
        if self.length_norm and avgdl > 0:
            norm = 1 - B + B * doc_len / avgdl
        else:
            norm = np.ones_like(doc_len, dtype=np.float64)
        if self.tf_saturation:
            tf_component = tf * (K1 + 1) / (tf + K1 * norm)
        else:
            tf_component = tf / norm
        return weight * tf_component

    def _term_score(self, tf: int, doc_len: int, avgdl: float, df_t: int, n: int) -> float:
        """Scalar form, used by _ReferenceLexicalRetriever and by
        tests/test_lexical.py's direct exercise of the all-off corner's
        formula. Kept as the single source of truth for "what is BM25's
        formula here" — _term_scores() above must agree with it elementwise."""
        weight = _idf(df_t, n) if self.idf else 1.0
        norm = (1 - B + B * doc_len / avgdl) if (self.length_norm and avgdl > 0) else 1.0
        if self.tf_saturation:
            tf_component = tf * (K1 + 1) / (tf + K1 * norm)
        else:
            # Unsaturated: no diminishing returns, but still divided by the same
            # length penalty saturation would use, so length_norm has an effect
            # independent of whether saturation is on (see implementation-notes.html
            # for why this is the reading chosen where the spec doesn't pin it down).
            tf_component = tf / norm
        return weight * tf_component


@dataclass(frozen=True)
class _ReferenceLexicalRetriever:
    """
    REFERENCE IMPLEMENTATION — not used by run_lexical_factorial or any
    production path. This is the original naive LexicalRetriever.retrieve(),
    preserved verbatim (scans every document for every query, rebuilds
    doc_tf/doc_len/df from scratch on every call): O(|corpus| * |query terms|)
    per query, correct by inspection but far too slow for HotpotQA at scale.

    Its only job is to exist as ground truth for
    tests/test_lexical_equivalence.py, which asserts the fast inverted-index
    path in LexicalRetriever returns identical scores. Do not optimise this
    class — its value is being obviously, boringly correct.
    """

    idf: bool = True
    tf_saturation: bool = True
    length_norm: bool = True

    def _term_score(self, tf: int, doc_len: int, avgdl: float, df_t: int, n: int) -> float:
        weight = _idf(df_t, n) if self.idf else 1.0
        norm = (1 - B + B * doc_len / avgdl) if (self.length_norm and avgdl > 0) else 1.0
        if self.tf_saturation:
            tf_component = tf * (K1 + 1) / (tf + K1 * norm)
        else:
            tf_component = tf / norm
        return weight * tf_component

    def retrieve(
        self, corpus: dict[str, str], queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        doc_ids = list(corpus)
        doc_tf: dict[str, Counter] = {}
        doc_len: dict[str, int] = {}
        df: Counter = Counter()

        for d in doc_ids:
            tokens = _tokenize(corpus[d])
            counts = Counter(tokens)
            doc_tf[d] = counts
            doc_len[d] = len(tokens)
            df.update(counts.keys())  # each distinct term increments df once per doc

        n = len(doc_ids)
        avgdl = sum(doc_len.values()) / n if n else 0.0

        run: dict[str, dict[str, float]] = {}
        for qid in sorted(queries):
            terms = set(_tokenize(queries[qid]))
            scores: dict[str, float] = {}
            for d in doc_ids:
                s = 0.0
                for t in terms:
                    tf = doc_tf[d].get(t, 0)
                    if tf == 0:
                        continue
                    s += self._term_score(tf, doc_len[d], avgdl, df.get(t, 0), n)
                if s > 0:
                    scores[d] = s
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            run[qid] = {d: raw - i * 1e-9 for i, (d, raw) in enumerate(ranked)}
        return run


def full_bm25() -> LexicalRetriever:
    """The all-on corner: this repository's own full BM25. Canonical constructor
    used by the closure control, which checks this against 001's externally
    anchored BM25 run."""
    return LexicalRetriever(idf=True, tf_saturation=True, length_norm=True)


# The full eight-cell factorial, fixed order: idf outermost, then tf_saturation,
# then length_norm. Order here is only for iteration — Shapley attribution (see
# rb.stats.shapley_values) is what makes the analysis order-independent.
ALL_CONFIGS: tuple[LexicalRetriever, ...] = tuple(
    LexicalRetriever(idf=i, tf_saturation=t, length_norm=l)
    for i in (True, False)
    for t in (True, False)
    for l in (True, False)
)

# One cumulative ladder through the factorial, for the adjacent-rung narrative
# comparisons protocols/002-ladder.md section 5 asks for (Shapley, not this
# ordering, is the order-independent attribution — this exists only to answer
# "does adding the next mechanism, in this order, buy a real gain"). Mechanisms
# are added in the same idf -> tf_saturation -> length_norm order ALL_CONFIGS
# already iterates in, so there is exactly one canonical ladder, not one per
# caller's taste. LADDER[-1] is full_bm25() by construction.
LADDER: tuple[LexicalRetriever, ...] = (
    LexicalRetriever(idf=False, tf_saturation=False, length_norm=False),
    LexicalRetriever(idf=True, tf_saturation=False, length_norm=False),
    LexicalRetriever(idf=True, tf_saturation=True, length_norm=False),
    LexicalRetriever(idf=True, tf_saturation=True, length_norm=True),
)


def shapley_from_ndcg(ndcg_by_config: dict[LexicalRetriever, float]) -> dict[str, float]:
    """
    Convert the eight measured nDCG@10 values (one per ALL_CONFIGS entry) into the
    frozenset-keyed value function rb.stats.shapley_values() expects, and return
    the Shapley value per mechanism.

    Kept separate from stats.shapley_values() so the generic Shapley machinery
    (tested against a hand-computed example) stays independent of what "a player"
    means for this particular factorial.
    """
    from rb.stats import shapley_values

    players = ["idf", "tf_saturation", "length_norm"]
    values = {}
    for cfg, ndcg in ndcg_by_config.items():
        active = frozenset(
            p for p, on in (("idf", cfg.idf), ("tf_saturation", cfg.tf_saturation), ("length_norm", cfg.length_norm))
            if on
        )
        values[active] = ndcg
    return shapley_values(values, players)
