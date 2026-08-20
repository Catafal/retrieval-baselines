"""
Experiment 003's metric set — protocols/003-graph-arm.md section 5.

WHY THIS DIFFERS FROM 001 AND 002. Every judged HotpotQA query has EXACTLY TWO gold
documents. nDCG@10 therefore spends eight of its ten rank positions scoring against
relevant documents that do not exist: those positions can only ever contribute zero, so
they add variance without adding signal. Recall@2 and Recall@5 are the cutoffs that
actually discriminate on a two-gold-document task.

There is a second, harder reason. The closure control compares this arm against
HippoRAG's published table, and that table reports R@2 and R@5. A metric family the
prior art does not publish cannot be checked against it at all, so nDCG@10 alone would
leave the graph arm with no external reference — which is the situation the protocol
exists to prevent.

nDCG@10, Recall@10 and Recall@100 are RETAINED rather than replaced, so 003's numbers
sit beside 002's on the same axes and a reader can follow the sequence across entries.

SCOPED, NOT GLOBAL. rb.metrics.MEASURES is what 001 and 002 published against and it is
left alone. Growing it would change the shape of `make reproduce`'s output — a published
promise — and break tests/test_coordination_regression.py, which asserts 001's `ranked`
dict by exact equality, all without moving a single measured value. Additive means
additive for the experiment that asked for it.
"""

from rb.metrics import MEASURES

# The two cutoffs section 5 makes primary, plus everything 001 and 002 already report.
GRAPH_MEASURES = MEASURES | {"recall_2", "recall_5"}

# Section 7: R@2 carries the registered prediction; R@5 and nDCG@10 are reported
# alongside it under the Holm family. Named here so the analysis cannot quietly promote
# whichever metric happens to reach significance.
PRIMARY_MEASURE = "recall_2"
SECONDARY_MEASURES = ("recall_5", "ndcg_cut_10")
