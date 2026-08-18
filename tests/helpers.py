"""Shared assertion for the Retriever contract, used across every rung's tests
so the four contract properties are checked identically for each implementation
rather than being re-derived (and possibly drifting) per test file."""


def assert_retriever_contract(retriever, corpus: dict[str, str], queries: dict[str, str], top_k: int) -> None:
    """
    Every Retriever implementation must, on a tiny in-memory corpus:
      - return at most top_k documents per query
      - return only document ids present in the corpus
      - return strictly decreasing scores per query
      - be deterministic across two invocations
    """
    run1 = retriever.retrieve(corpus, queries, top_k)
    run2 = retriever.retrieve(corpus, queries, top_k)
    assert run1 == run2, "retrieve() must be deterministic across invocations"

    for qid, docs in run1.items():
        assert len(docs) <= top_k, f"{qid}: returned more than top_k={top_k} documents"
        assert set(docs) <= set(corpus), f"{qid}: returned a document id not in the corpus"
        scores = list(docs.values())
        assert scores == sorted(scores, reverse=True), f"{qid}: scores are not decreasing"
        assert len(set(scores)) == len(scores), f"{qid}: scores are not strictly decreasing (a tie survived)"
