"""
`run_rung` must refuse to score from a source tree that differs from HEAD.

The defect this closes: 003's 2Wiki arms were scored while `pool2wiki.py` -- the module building
their corpus -- was still uncommitted, so the `git_commit` written into every one of those
artifacts did not contain the code that produced them. It was caught by re-running an arm and
comparing hashes, which is luck, not a control.
"""

import pytest

from rb import retriever as rr


def _state(porcelain: str, monkeypatch):
    class R:
        stdout = porcelain
    monkeypatch.setattr(rr.subprocess, "run", lambda *a, **k: R())
    return rr.working_tree_state()


def test_an_unstaged_modification_is_parsed_with_its_full_path(monkeypatch):
    """KILLS: `.stdout.strip()` before splitting lines.

    Porcelain lines are `XY<space>path`. An UNSTAGED modification begins with a space, so
    stripping the whole output eats the first line's leading space and slices that path one
    character short -- `src/rb/x.py` becomes `rc/rb/x.py`, which then fails the `src/` prefix
    test and the run proceeds when it should halt. It corrupts only the FIRST entry, so a test
    with one dirty file is the one that catches it, and a test with two might not.
    """
    st = _state(" M src/rb/retriever.py\n", monkeypatch)
    assert st["dirty_paths"] == ["src/rb/retriever.py"], "leading space must not be stripped"
    assert st["dirty_source_paths"] == ["src/rb/retriever.py"]
    assert st["clean"] is False


def test_staged_untracked_and_unstaged_entries_all_parse(monkeypatch):
    st = _state("M  src/rb/a.py\n?? src/rb/b.py\n D results/003/old.json\n", monkeypatch)
    assert st["dirty_paths"] == ["results/003/old.json", "src/rb/a.py", "src/rb/b.py"]
    assert st["dirty_source_paths"] == ["src/rb/a.py", "src/rb/b.py"]


def test_a_clean_tree_is_scorable(monkeypatch):
    st = _state("", monkeypatch)
    assert st["clean"] is True and st["dirty_source_paths"] == []
    assert rr.assert_scorable(st)["override_used"] is False


def test_dirty_source_halts_the_run(monkeypatch):
    """KILLS: downgrading the refusal to a warning, or dropping the call from run_rung."""
    monkeypatch.delenv(rr.ALLOW_DIRTY, raising=False)
    with pytest.raises(RuntimeError, match="refusing to score"):
        rr.assert_scorable({"clean": False, "dirty_paths": ["src/rb/x.py"],
                            "dirty_source_paths": ["src/rb/x.py"]})


def test_a_dirty_results_dir_does_not_halt_the_run(monkeypatch):
    """Only `src/` gates. A stale artifact or an unrelated note is not a reason to refuse to
    score, and a check that fires on those would be turned off within a week."""
    monkeypatch.delenv(rr.ALLOW_DIRTY, raising=False)
    out = rr.assert_scorable({"clean": False, "dirty_paths": ["results/003/x.json"],
                              "dirty_source_paths": []})
    assert out["override_used"] is False


def test_the_override_proceeds_and_is_recorded(monkeypatch):
    """KILLS: an escape hatch that leaves no trace.

    Exploratory runs are legitimate, so the override must work -- but a run made with it must not
    be mistakable for a clean one afterwards, which means `override_used` has to reach the
    artifact.
    """
    monkeypatch.setenv(rr.ALLOW_DIRTY, "1")
    out = rr.assert_scorable({"clean": False, "dirty_paths": ["src/rb/x.py"],
                              "dirty_source_paths": ["src/rb/x.py"]})
    assert out["override_used"] is True
    assert out["dirty_source_paths"] == ["src/rb/x.py"]


def test_run_rung_calls_the_gate_before_retrieving(monkeypatch):
    """The check must precede the expensive half, or it costs an hour to learn the tree was
    dirty."""
    import inspect

    body = inspect.getsource(rr.run_rung)
    assert "assert_scorable()" in body
    assert body.index("assert_scorable()") < body.index("retriever.retrieve(")
