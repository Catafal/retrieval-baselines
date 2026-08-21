# Superseded — produced by a defective walk

These artifacts were produced by an implementation that was **not** a personalized PageRank:
it applied `B^T (B rank)` on an unnormalised co-occurrence matrix and rescaled globally.
See `results/003/graph-arm-defect-report.json` and
`protocols/003-amendment-3-ppr-correction.md`.

**Kept, not deleted.** The corrected run lives in `../graph/`. Retracting the original numbers
silently would be a worse fault than the bug — a reader must be able to see what was published,
what replaced it, and by how much it moved.

Headline of this defective run: Recall@2 0.2132, which is below a baseline that did no
propagation at all (0.2307).
