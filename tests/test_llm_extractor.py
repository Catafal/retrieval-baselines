"""
The LLM extractor's contract, its cache, and the two failure modes that would be invisible.

Every test here runs with no API key and no network: the transport is injected, so the logic can
be exercised without spending money or depending on a provider being up.
"""
import json

import pytest

from rb.experiments.graph import llm_extractor as m


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """
    No test may write to the real extraction cache. Autouse, not opt-in.

    Two tests here originally relied on the default CACHE path and appended fake entities
    ("Palmyra", "Rome", "Alice") into results/004/extraction-cache.jsonl — the artifact of record
    for a paid, pre-registered run. Nothing would have flagged it: the file is append-only by
    design, the fake keys are well-formed, and a later real run would have read them as cached
    extractions and skipped calling for those passages. Opt-in isolation is one forgotten
    decorator away from that every time, so it is enforced here for every test in the file.
    """
    monkeypatch.setattr(m, "CACHE", tmp_path / "extraction-cache.jsonl")


def _reply(passages):
    """A fake OpenRouter response carrying `passages` as the structured payload."""
    return {"choices": [{"message": {"content": json.dumps({"passages": passages})}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_contract_matches_the_spacy_extractor():
    """Same shape as extractor.extract: a list of (surface_form, label) per doc, unfiltered."""
    calls = []

    def client(endpoint, body):
        calls.append(body)
        return _reply([{"index": 0, "entities": [{"text": "Palmyra", "label": "GPE"}]}])

    out = m.extract_many({"d1": "Its county seat is Palmyra."}, client=client)

    assert out == {"d1": [("Palmyra", "GPE")]}
    assert isinstance(out["d1"][0], tuple) and len(out["d1"][0]) == 2


def test_the_pin_is_sent_and_fallbacks_are_disabled():
    """
    KILLS: a pin that degrades gracefully.

    A silent reroute to another backend, or another quantisation, must fail the run rather than
    return numbers from a model that is not the registered one.
    """
    seen = {}

    def client(endpoint, body):
        seen.update(body)
        return _reply([{"index": 0, "entities": []}])

    m.extract_many({"d1": "text"}, client=client)

    assert seen["model"] == "z-ai/glm-4.7-flash"
    assert seen["provider"]["only"] == ["DeepInfra"]
    assert seen["provider"]["quantizations"] == ["bf16"]
    assert seen["provider"]["allow_fallbacks"] is False
    assert seen["temperature"] == 0


def test_a_dropped_passage_does_not_shift_its_neighbours_entities():
    """
    KILLS the misalignment that would still look plausible in the artifact.

    If the model returns two objects for a three-passage batch and the code zipped them
    positionally, passage 2's entities would land on passage 1 and passage 2 would inherit
    passage 2's. Every doc would have entities, none would be wrong-looking, and the graph would
    be built on a silent permutation. Indices are honoured instead, and a missing index yields an
    empty list.
    """
    def client(endpoint, body):
        # index 1 is missing entirely
        return _reply([{"index": 0, "entities": [{"text": "Alice", "label": "PERSON"}]},
                       {"index": 2, "entities": [{"text": "Berlin", "label": "GPE"}]}])

    out = m.extract_many({"a": "p0", "b": "p1", "c": "p2"}, client=client)

    assert out["a"] == [("Alice", "PERSON")]
    assert out["b"] == [], "a dropped passage must yield nothing, not its neighbour's entities"
    assert out["c"] == [("Berlin", "GPE")]


def test_offline_mode_refuses_to_call_on_a_cache_miss():
    """
    KILLS: a scored number produced by an uncached API call.

    Protocol 004 §6 makes the cache the artifact of record. If a miss could quietly reach the
    network, `make reproduce` would silently re-buy non-deterministic extractions and the
    experiment would not be reproducible at all.
    """
    with pytest.raises(RuntimeError, match="not in the extraction cache"):
        m.extract_many({"d1": "uncached"}, offline=True)


def test_the_cache_key_changes_when_the_prompt_changes(monkeypatch):
    """
    KILLS: a re-tuned prompt silently inheriting the registered prompt's extractions.

    The prompt is frozen in the protocol. If it is edited, every key must miss, so the change is
    visible as a re-extraction rather than as nothing at all.
    """
    before = m.cache_key("some passage")
    monkeypatch.setattr(m, "PROMPT", m.PROMPT + " Also extract nicknames.")
    assert m.cache_key("some passage") != before


def test_the_cache_key_changes_when_the_pin_changes(monkeypatch):
    """Same argument for the model pin: a different backend is a different experiment."""
    before = m.cache_key("some passage")
    monkeypatch.setattr(m, "QUANTIZATION", "fp8")
    assert m.cache_key("some passage") != before


def test_a_second_run_reads_the_cache_and_does_not_call():
    """The cache has to actually save the money it exists to save."""
    calls = []

    def client(endpoint, body):
        calls.append(1)
        return _reply([{"index": 0, "entities": [{"text": "Rome", "label": "GPE"}]}])

    texts = {"d1": "Rome is old."}
    first = m.extract_many(texts, client=client)
    second = m.extract_many(texts, client=client)

    assert first == second == {"d1": [("Rome", "GPE")]}
    assert len(calls) == 1, "the second run must be served entirely from the cache"


def test_concurrent_extraction_keeps_every_passage_with_its_own_entities():
    """
    KILLS: a threaded fan-out that returns the right entities attached to the wrong documents.

    With WORKERS batches in flight and a shared cache dict, the plausible failure is not a crash
    but a permutation: every document has entities, none looks wrong, and the graph is built on a
    silent shuffle. Each passage here carries a unique marker and must come back with its own.
    """
    n = 250
    texts = {f"d{i}": f"passage about MARKER{i} in a town" for i in range(n)}

    def client(endpoint, body):
        # Echo back the marker found in each numbered passage, so a misattribution is detectable.
        blocks = body["messages"][1]["content"].split("\n\n")
        out = []
        for j, block in enumerate(blocks):
            marker = block.split("MARKER")[1].split(" ")[0]
            out.append({"index": j, "entities": [{"text": f"MARKER{marker}", "label": "GPE"}]})
        return _reply(out)

    result = m.extract_many(texts, client=client)

    assert len(result) == n
    for i in range(n):
        assert result[f"d{i}"] == [(f"MARKER{i}", "GPE")], f"d{i} got another document's entities"


def test_a_resumed_run_extracts_only_what_is_missing():
    """
    The cache is what makes a multi-hour run survivable. If a crash at 80% meant re-buying
    everything, the run would be unaffordable to retry and the temptation would be to press on
    with partial results.
    """
    calls = []

    def client(endpoint, body):
        blocks = body["messages"][1]["content"].split("\n\n")
        calls.append(len(blocks))
        return _reply([{"index": j, "entities": []} for j in range(len(blocks))])

    first = {f"d{i}": f"text {i}" for i in range(20)}
    m.extract_many(first, client=client)
    extracted_first = sum(calls)

    calls.clear()
    second = dict(first)
    second.update({f"new{i}": f"fresh {i}" for i in range(5)})
    m.extract_many(second, client=client)

    assert extracted_first == 20
    assert sum(calls) == 5, "a resumed run must extract only the passages it does not already have"


def test_a_truncated_response_is_retried_not_fatal():
    """
    KILLS the defect that killed the first full run.

    One batch returned truncated JSON eight minutes in, `json.loads` raised through
    ThreadPoolExecutor.map, and 110,068 passages of work stopped at 1,300. A malformed response
    from one call must cost that call, not the run.
    """
    attempts = []

    def client(endpoint, body):
        attempts.append(1)
        if len(attempts) == 1:
            return {"choices": [{"finish_reason": "length",
                                 "message": {"content": '{"passages": [{"index": 0, "enti'}}]}
        return _reply([{"index": 0, "entities": [{"text": "Kyoto", "label": "GPE"}]}])

    m.MAX_RETRIES  # documents the dependency
    out = m.extract_many({"d1": "Kyoto is old."}, client=client)

    assert out == {"d1": [("Kyoto", "GPE")]}
    assert len(attempts) == 2, "the truncated call should have been retried once, not raised"


def test_one_bad_passage_does_not_cost_its_batch_its_work(monkeypatch):
    """
    KILLS: a single pathological input discarding nine neighbours' successful extractions.

    The batch is retried, then split to single passages so the failure is isolated to the one
    passage that causes it.
    """
    monkeypatch.setattr(m, "MAX_RETRIES", 1)

    def client(endpoint, body):
        content = body["messages"][1]["content"]
        if "POISON" in content and content.count("[") > 1:
            raise m.MalformedResponse("batch containing the bad passage fails")
        if "POISON" in content:
            raise m.MalformedResponse("and the bad passage fails alone too")
        blocks = content.split("\n\n")
        return _reply([{"index": j, "entities": [{"text": "Ok", "label": "GPE"}]}
                       for j in range(len(blocks))])

    texts = {f"d{i}": f"clean passage {i}" for i in range(9)}
    texts["d9"] = "a POISON passage"

    out = m.extract_many(texts, client=client)

    # The nine good passages were banked. The bad one is ABSENT from the result rather than
    # present-and-empty, so a caller can tell "no entities" from "never extracted".
    cached = m.load_cache()
    good = [d for d in texts if d != "d9" and m.cache_key(texts[d]) in cached]
    assert len(good) == 9, "the clean passages in a failed batch must survive"
    assert m.cache_key(texts["d9"]) not in cached, (
        "a failed extraction must not be cached as empty — it would be indistinguishable "
        "from a passage that genuinely has no entities"
    )
    assert "d9" not in out, "an excluded passage must be absent from the result, not empty"
    assert len(out) == 9


def test_a_truncated_loop_is_salvaged_to_its_distinct_entities():
    """
    protocol-004 amendment 2: the model emits its full list, then loops on an ambiguous member.

    Measured on all eight real failures: the first repeat falls at exactly the distinct count, so
    the truncated prefix carries everything a terminating response would have. The graph dedupes
    nodes by string regardless, so the salvaged set is what it would have received.
    """
    loop = ('{"passages": [{"index": 0, "entities": ['
            '{"text": "Fenn College", "label": "ORG"}, '
            '{"text": "Cleveland State", "label": "ORG"}, '
            + '{"text": "Cleveland State", "label": "GPE"}, {"text": "Cleveland State", "label": "ORG"}, ' * 40
            + '{"text": "Cleveland Sta')

    def client(endpoint, body):
        return {"choices": [{"finish_reason": "length", "message": {"content": loop}}]}

    out = m.extract_many({"d1": "Woodling coached at Fenn College."}, client=client)

    assert out["d1"] == [("Fenn College", "ORG"), ("Cleveland State", "ORG"),
                         ("Cleveland State", "GPE")], "distinct entities, order preserved"
    assert m.load_cache()[m.cache_key("Woodling coached at Fenn College.")] == out["d1"]


def test_a_half_written_entity_is_discarded_rather_than_guessed():
    """The final object in a truncated response is incomplete; only whole objects are recovered."""
    partial = '{"passages": [{"index": 0, "entities": [{"text": "Kyoto", "label": "GPE"}, {"text": "Osa'

    def client(endpoint, body):
        return {"choices": [{"finish_reason": "length", "message": {"content": partial}}]}

    assert m.extract_many({"d1": "Kyoto and Osaka."}, client=client)["d1"] == [("Kyoto", "GPE")]
