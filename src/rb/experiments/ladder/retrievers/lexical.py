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
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

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
class LexicalRetriever:
    idf: bool = True
    tf_saturation: bool = True
    length_norm: bool = True

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
            terms = set(_tokenize(queries[qid]))  # BM25's outer sum is over distinct query terms
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
            # Unlike grep_baseline.rank() (which collapses to bare rank position,
            # because coordination-level matching only ever produces a handful of
            # distinct values and ties heavily), BM25 scores are continuous and
            # carry real magnitude — a mechanism's effect on that magnitude is
            # exactly what tests/test_lexical.py checks. So the raw score is kept,
            # with a tie-break epsilon (1e-9 * rank position) subtracted only to
            # satisfy the contract's "strictly decreasing" requirement for the
            # rare case of an exact tie; it is far too small to alter any
            # comparison of actual score magnitude.
            run[qid] = {d: raw - i * 1e-9 for i, (d, raw) in enumerate(ranked)}
        return run

    def _term_score(self, tf: int, doc_len: int, avgdl: float, df_t: int, n: int) -> float:
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
