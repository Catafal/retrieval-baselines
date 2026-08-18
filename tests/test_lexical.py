"""
Each lexical switch tested in isolation, on hand-built corpora where the
mechanism's effect has a known direction — same shape as test_grep_baseline.py's
word-boundary test: small, fast, data-free, behavioural.
"""

from rb.experiments.ladder.retrievers.lexical import ALL_CONFIGS, LADDER, LexicalRetriever, full_bm25


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
    corpus = {"a": "x x x y", "b": "x y y"}
    queries = {"q": "x y"}
    off_corner = LexicalRetriever(idf=False, tf_saturation=False, length_norm=False)
    scores = off_corner._term_score  # exercise directly: tf/norm with norm=1, weight=1
    assert scores(tf=3, doc_len=4, avgdl=4, df_t=2, n=2) == 3.0
    assert scores(tf=1, doc_len=3, avgdl=4, df_t=2, n=2) == 1.0


def test_ladder_runs_from_all_off_to_full_bm25_one_mechanism_at_a_time():
    assert LADDER[0] == LexicalRetriever(idf=False, tf_saturation=False, length_norm=False)
    assert LADDER[-1] == full_bm25()
    for prev_cfg, next_cfg in zip(LADDER, LADDER[1:]):
        flips = sum(
            getattr(prev_cfg, f) != getattr(next_cfg, f) for f in ("idf", "tf_saturation", "length_norm")
        )
        assert flips == 1, "each ladder step must turn on exactly one additional mechanism"
