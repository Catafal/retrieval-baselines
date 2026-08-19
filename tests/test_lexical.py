"""
Each lexical switch tested in isolation, on hand-built corpora where the
mechanism's effect has a known direction — same shape as test_grep_baseline.py's
word-boundary test: small, fast, data-free, behavioural.
"""

import math

import pytest

from rb.experiments.ladder.retrievers.lexical import ALL_CONFIGS, LADDER, LexicalRetriever, corpus_length_stats, full_bm25


def test_idf_on_prefers_rare_term_match_over_common_term_match():
    """Two documents of equal length, each matching the query on one term with
    equal raw frequency, but one matched term is rare across the corpus and the
    other is common. With IDF on, the rare-term match must outrank the common one."""
    corpus = {
        "common_doc": "widget widget widget filler filler filler",
        "rare_doc": "gizmo gizmo gizmo filler filler filler",
    }
    # "widget" appears in three documents, "gizmo" in only one — pad the corpus
    # with two more documents containing "widget" so document frequency differs.
    corpus["padding1"] = "widget appears here too"
    corpus["padding2"] = "widget appears here as well"
    queries = {"q": "widget gizmo"}

    retriever = LexicalRetriever(idf=True, tf_saturation=False, length_norm=False)
    run = retriever.retrieve(corpus, queries, top_k=10)["q"]
    assert run["rare_doc"] > run["common_doc"], "rare-term match should outrank common-term match with IDF on"


def test_idf_off_scores_rare_and_common_matches_equally():
    corpus = {
        "common_doc": "widget filler filler filler",
        "rare_doc": "gizmo filler filler filler",
        "padding1": "widget appears here too",
        "padding2": "widget appears here as well",
    }
    queries = {"q": "widget gizmo"}
    retriever = LexicalRetriever(idf=False, tf_saturation=False, length_norm=False)
    run = retriever.retrieve(corpus, queries, top_k=10)["q"]
    assert abs(run["rare_doc"] - run["common_doc"]) < 1e-6, "without IDF, a single match should score identically regardless of rarity"


def test_saturation_on_diminishes_marginal_gain_of_repeated_matches():
    """The tenth repetition of a term must add less score than the second, when
    tf_saturation is on. All documents are equal length so length_norm cannot
    confound the comparison; idf is off so a single term's weight is constant."""
    def doc(term: str, n: int) -> str:
        return " ".join([term] * n + ["pad"] * (20 - n))

    corpus = {"tf1": doc("x", 1), "tf2": doc("x", 2), "tf9": doc("x", 9), "tf10": doc("x", 10)}
    queries = {"q": "x"}
    retriever = LexicalRetriever(idf=False, tf_saturation=True, length_norm=False)
    run = retriever.retrieve(corpus, queries, top_k=10)["q"]

    gain_low = run["tf2"] - run["tf1"]     # 1 -> 2 repetitions
    gain_high = run["tf10"] - run["tf9"]   # 9 -> 10 repetitions
    assert gain_high < gain_low, "the tenth repetition must add less than the second under saturation"


def test_saturation_off_gives_constant_marginal_gain():
    def doc(term: str, n: int) -> str:
        return " ".join([term] * n + ["pad"] * (20 - n))

    corpus = {"tf1": doc("x", 1), "tf2": doc("x", 2), "tf9": doc("x", 9), "tf10": doc("x", 10)}
    queries = {"q": "x"}
    retriever = LexicalRetriever(idf=False, tf_saturation=False, length_norm=False)
    run = retriever.retrieve(corpus, queries, top_k=10)["q"]

    gain_low = run["tf2"] - run["tf1"]
    gain_high = run["tf10"] - run["tf9"]
    assert abs(gain_high - gain_low) < 1e-9, "without saturation, marginal gain per repetition should be constant"


def test_length_norm_on_prefers_short_document_over_long_document():
    """A short document matching a term once must outrank a long document
    matching the same term once, when length_norm is on."""
    corpus = {
        "short": "target",
        "long": "target " + " ".join(["filler"] * 200),
    }
    queries = {"q": "target"}
    retriever = LexicalRetriever(idf=False, tf_saturation=False, length_norm=True)
    run = retriever.retrieve(corpus, queries, top_k=10)["q"]
    assert run["short"] > run["long"], "short document should outrank long document with length_norm on"


def test_length_norm_off_scores_equal_length_independent():
    corpus = {
        "short": "target",
        "long": "target " + " ".join(["filler"] * 200),
    }
    queries = {"q": "target"}
    retriever = LexicalRetriever(idf=False, tf_saturation=False, length_norm=False)
    run = retriever.retrieve(corpus, queries, top_k=10)["q"]
    assert abs(run["short"] - run["long"]) < 1e-6, "without length_norm, document length should not affect a single match's score"


def test_factorial_has_eight_distinct_configurations():
    assert len(ALL_CONFIGS) == 8
    assert len({(c.idf, c.tf_saturation, c.length_norm) for c in ALL_CONFIGS}) == 8


def test_all_on_corner_is_full_bm25():
    on_corner = LexicalRetriever(idf=True, tf_saturation=True, length_norm=True)
    assert on_corner in ALL_CONFIGS
    assert on_corner == full_bm25()


def test_all_off_corner_is_raw_term_frequency_sum():
    """
    Through retrieve(), not through a private method.

    With every mechanism off the score must be the plain count of query-term
    occurrences: "x x x y" holds x three times and y once, so 4; "x y y" holds
    one x and two y, so 3. Asserting on the public output means this test also
    covers the vectorised path, which is the one that actually runs.
    """
    corpus = {"a": "x x x y", "b": "x y y"}
    queries = {"q": "x y"}
    off_corner = LexicalRetriever(idf=False, tf_saturation=False, length_norm=False)
    run = off_corner.retrieve(corpus, queries, top_k=10)["q"]

    # The runner subtracts a 1e-9-per-rank epsilon to keep scores strictly
    # decreasing for the Retriever contract, so compare with a tolerance well
    # below the gap between the two documents.
    assert run["a"] == pytest.approx(4.0, abs=1e-6)
    assert run["b"] == pytest.approx(3.0, abs=1e-6)


def test_ladder_runs_from_all_off_to_full_bm25_one_mechanism_at_a_time():
    assert LADDER[0] == LexicalRetriever(idf=False, tf_saturation=False, length_norm=False)
    assert LADDER[-1] == full_bm25()
    for prev_cfg, next_cfg in zip(LADDER, LADDER[1:]):
        flips = sum(
            getattr(prev_cfg, f) != getattr(next_cfg, f) for f in ("idf", "tf_saturation", "length_norm")
        )
        assert flips == 1, "each ladder step must turn on exactly one additional mechanism"


def test_corpus_length_stats_against_hand_built_corpus_with_known_token_counts():
    """
    Token counts by construction: 1, 3, 5, 7 (using plain ascii words so
    _tokenize's lowercasing/splitting does not change the count).
    mean = 16/4 = 4; median of an even-sized sorted set is the mean of the two
    middle values, (3+5)/2 = 4; population variance ((1-4)^2+(3-4)^2+(5-4)^2+
    (7-4)^2)/4 = 20/4 = 5, so std = sqrt(5); cv = std/mean.
    """
    corpus = {
        "one_token": "a",
        "three_tokens": "a b c",
        "five_tokens": "a b c d e",
        "seven_tokens": "a b c d e f g",
    }
    stats = corpus_length_stats(corpus)
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["median"] == pytest.approx(4.0)
    assert stats["std"] == pytest.approx(math.sqrt(5))
    assert stats["coefficient_of_variation"] == pytest.approx(math.sqrt(5) / 4.0)


def test_corpus_length_stats_uses_the_same_tokenizer_the_scorer_uses():
    """Punctuation and casing must be handled exactly as _tokenize handles
    them (lowercase, split on non-alphanumerics, no dedup) — a naive
    whitespace split would count "World!!" as one token instead of the one
    real token "world" it tokenizes to, and "Hello," as one instead of
    "hello"."""
    corpus = {"doc": "Hello, World!! Hello again."}  # tokenizes to: hello world hello again (4 tokens)
    stats = corpus_length_stats(corpus)
    assert stats["mean"] == pytest.approx(4.0)


def test_corpus_length_stats_rejects_empty_corpus():
    with pytest.raises(ValueError):
        corpus_length_stats({})


def test_retrieval_is_identical_across_processes(tmp_path):
    """
    Reproducibility across processes, not just within one.

    BM25's outer sum runs over the distinct query terms. When those came from a
    bare set, iteration order depended on PYTHONHASHSEED, which is randomised per
    process. Float addition is not associative, so the same contributions summed
    in a different order produced totals differing in the last bits, which was
    enough to flip documents that tie. Two processes disagreed on the ranking for
    63 of 300 SciFact queries, and nothing in the suite noticed, because every
    determinism test ran inside a single process.

    A stranger rerunning this repo has to get the ranking we published.
    """
    import subprocess
    import sys

    script = tmp_path / "once.py"
    script.write_text(
        "import json\n"
        "from rb.experiments.ladder.retrievers.lexical import LexicalRetriever, build_index\n"
        "import dataclasses\n"
        # Documents engineered to tie: same terms, different order, so any
        # order-dependent summation shows up as a flipped ranking.
        "corpus = {f'd{i}': 'alpha beta gamma delta' for i in range(40)}\n"
        "corpus['d7'] = 'delta gamma beta alpha'\n"
        "q = {'q': 'alpha beta gamma delta'}\n"
        "r = dataclasses.replace(LexicalRetriever(idf=True, tf_saturation=True, length_norm=True),"
        " index=build_index(corpus))\n"
        "print(json.dumps(list(r.retrieve(corpus, q, 40)['q'])))\n"
    )
    env_runs = [
        subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, check=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        ).stdout
        for _ in range(3)
    ]
    assert env_runs[0] == env_runs[1] == env_runs[2]
