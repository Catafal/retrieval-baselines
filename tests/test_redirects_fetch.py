"""
The redirect fetch — protocols/005-identity.md section 3.

Offline. These test the two behaviours that fail silently rather than loudly: dropping the
second page of a continued response, and treating a rate-limit as a permanent failure. A fetch
that quietly returns 90% of the aliases produces a coverage number that is simply wrong, and
nothing downstream can tell.
"""

import pytest

from rb.experiments.graph import redirects as R


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._body, self.status_code, self.headers = body, status, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status at {self.status_code}")


class FakeSession:
    """Returns queued responses in order and records the params it was called with."""

    def __init__(self, responses):
        self._responses, self.calls = list(responses), []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        return self._responses.pop(0)


def _page(title, aliases):
    return {"title": title, "redirects": [{"title": a} for a in aliases]}


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """The retry paths sleep; the tests must not."""
    monkeypatch.setattr(R.time, "sleep", lambda _s: None)


def test_continuation_is_followed_not_truncated():
    """A heavily-redirected article exceeds one page. Keeping only the first drops exactly the
    aliases that a popular entity has most of."""
    session = FakeSession([
        FakeResponse({"query": {"pages": [_page("Aristotle", ["Aristotle of Stagira"])]},
                      "continue": {"rdcontinue": "X", "continue": "||"}}),
        FakeResponse({"query": {"pages": [_page("Aristotle", ["Aristoteles"])]}}),
    ])
    found, calls = R._batch(session, ["Aristotle"])

    assert calls == 2
    assert sorted(found["Aristotle"]) == ["Aristoteles", "Aristotle of Stagira"]
    assert session.calls[1]["rdcontinue"] == "X", "continuation token was not sent back"


def test_result_is_keyed_by_the_canonical_title_the_api_returns():
    """
    The API resolves the title it was asked for. What comes back is Wikipedia's canonical form,
    which is what the aliases actually point at, so that is what the registry must be keyed on.
    """
    session = FakeSession([FakeResponse(
        {"query": {"normalized": [{"from": "cleveland_state_university",
                                   "to": "Cleveland State University"}],
                   "pages": [_page("Cleveland State University", ["Cleveland State"])]}})])
    found, _ = R._batch(session, ["cleveland_state_university"])
    assert found == {"Cleveland State University": ["Cleveland State"]}


def test_titles_with_no_redirects_are_absent():
    """Absence must mean 'no aliases', not 'never asked'. The manifest counts both."""
    session = FakeSession([FakeResponse({"query": {"pages": [{"title": "A"}]}})])
    found, _ = R._batch(session, ["A"])
    assert found == {}


def test_namespace_and_limit_are_pinned_in_the_request():
    """R2 is a protocol rule, so it has to be in the wire request, not just the docstring."""
    session = FakeSession([FakeResponse({"query": {"pages": []}})])
    R._batch(session, ["A"])
    assert session.calls[0]["rdnamespace"] == "0"
    assert session.calls[0]["rdlimit"] == "max"


def test_rate_limit_is_retried_then_succeeds():
    session = FakeSession([
        FakeResponse({}, status=429, headers={"Retry-After": "0"}),
        FakeResponse({"query": {"pages": [_page("A", ["a"])]}}),
    ])
    found, _ = R._batch(session, ["A"])
    assert found == {"A": ["a"]}


def test_maxlag_arrives_as_a_200_and_is_still_retried():
    """WMF reports replication lag as a normal 200 with an error member. Treating it as success
    would record an empty result for every title in the batch."""
    session = FakeSession([
        FakeResponse({"error": {"code": "maxlag", "info": "waiting for a replica"}}),
        FakeResponse({"query": {"pages": [_page("A", ["a"])]}}),
    ])
    found, _ = R._batch(session, ["A"])
    assert found == {"A": ["a"]}


def test_persistent_failure_raises_rather_than_returning_empty():
    session = FakeSession([FakeResponse({}, status=503) for _ in range(R.MAX_RETRIES)])
    with pytest.raises(R.FetchFailed):
        R._batch(session, ["A"])
