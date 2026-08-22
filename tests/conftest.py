"""
Shared test configuration.

THE CLEAN-TREE GATE IS EXEMPTED FOR TESTS, deliberately and in one place.

`rb.retriever.run_rung` refuses to score when `src/` differs from HEAD, because the `git_commit`
it stamps into a published artifact would otherwise name a commit that does not contain the code
that produced the numbers. That guarantee is about ARTIFACTS. The suite drives `run_rung` against
tiny in-memory fixtures and writes to `tmp_path`; nothing it produces is published, and the whole
point of a test run is that the tree is mid-edit.

Leaving the gate armed here would make the suite fail whenever there is uncommitted work — which
is always, while writing the very change the suite is meant to check — and the fix someone reaches
for at that point is to weaken the gate. So the exemption is granted explicitly, scoped to the
test session, and named.

`tests/test_scored_run_requires_clean_tree.py` still exercises the gate directly: it drives
`assert_scorable` with constructed state and asserts `run_rung` calls it before retrieving, so the
behaviour stays covered without the suite being subject to it.
"""

import pytest

from rb import retriever as rr


@pytest.fixture(autouse=True)
def _allow_dirty_tree_during_tests(monkeypatch):
    monkeypatch.setenv(rr.ALLOW_DIRTY, "1")
