"""
The hybrid rung — reciprocal rank fusion of full BM25 and dense, k=60, fixed.

No weight search. The fusion rule is fixed in the pre-registration before any
hybrid number exists, so its weights cannot be tuned against the test set. If a
weighted fusion turns out to be interesting later, that is a separate experiment
about fusion, not a rung in this one.
"""

# protocols/002-amendment-1-dense.md section 5: "k = 60, fixed here before any
# hybrid number exists." Not a tunable default — changing it after seeing a
# hybrid result would be the weight search the amendment explicitly rules out.
RRF_K = 60


class HybridRetriever:
    def __init__(self, lexical, dense, k: int = RRF_K, candidate_k: int = 100):
        self.lexical = lexical
        self.dense = dense
        self.k = k
        # Each component retriever is asked for more candidates than the caller's
        # top_k, so RRF has enough of each ranking to combine before the final cut;
        # the cut to top_k happens after fusion, not before.
        self.candidate_k = candidate_k
        self.name = f"hybrid({lexical.name}+{dense.name},k={k})"

    def retrieve(
        self, corpus: dict[str, str], queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        pool_k = max(top_k, self.candidate_k)
        lex_run = self.lexical.retrieve(corpus, queries, pool_k)
        dense_run = self.dense.retrieve(corpus, queries, pool_k)

        run: dict[str, dict[str, float]] = {}
        for qid in sorted(queries):
            rrf = self.fuse(lex_run.get(qid, {}), dense_run.get(qid, {}))
            ordered = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            run[qid] = {d: float(len(ordered) - i) for i, (d, _) in enumerate(ordered)}
        return run

    def fuse(self, *components: dict[str, float]) -> dict[str, float]:
        """
        Raw RRF weights for one query, before the rank-position re-encoding.

        Split out of `retrieve` so it can be tested directly. It has to be: the
        re-encoding in `retrieve` throws magnitude away and keeps only order, and
        RRF's order is invariant to a constant shift in the starting rank — so
        `1/(k+pos)` with pos starting at 0 instead of 1 produces different weights
        and an identical ranking. A mutation test confirmed that flipping
        `start=1` to `start=0` left the whole suite passing, because nothing could
        observe the weights. This method is the seam that makes that observable.
        """
        rrf: dict[str, float] = {}
        for component in components:
            # Component scores are already the strictly-decreasing rank-position
            # encoding every rung in this repo uses, so rank is recovered by
            # re-sorting rather than by trusting the raw score's magnitude —
            # RRF only ever needs rank position, never the score itself.
            ranked = sorted(component.items(), key=lambda kv: (-kv[1], kv[0]))
            # 1-based: the top document of a component contributes 1/(k+1), not 1/k.
            for pos, (d, _) in enumerate(ranked, start=1):
                rrf[d] = rrf.get(d, 0.0) + 1.0 / (self.k + pos)
        return rrf
