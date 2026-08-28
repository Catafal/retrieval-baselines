"""
The identity source — protocols/005-identity.md sections 3 and 4.

WHY REDIRECTS. 003 and 004 key entities by exact normalised string, so "Cleveland State" and
"Cleveland State University" are two nodes with no edge between them and no walk can cross what
the node set never joined. Typed identity needs an alias registry, and NB-23 forbids authoring
one: a file written by the same person who reads the outcome is the input tuned to the result.

Wikipedia redirects are the alternative because they already exist, were authored for reasons
unrelated to this experiment, and cannot be adjusted by it. Both corpora are Wikipedia-derived and
their document titles are exact article titles, so redirects apply directly.

WHY A SCOPED FETCH AND NOT THE DUMP. Only redirects whose TARGET is a pool title can ever merge
two nodes in this graph. That is at most ~66,581 HotpotQA and ~43,487 2Wiki titles. Batched at the
API's 50-title limit it is on the order of 2,200 requests. The `redirect` and `page` SQL dumps
would need a multi-gigabyte download and a join to answer the same question, and no reader would
run that to check a number.

WHY THE SNAPSHOT IS THE ARTIFACT. The live API is mutable: a redirect created tomorrow changes
what this returns. So the fetch happens once, the result is committed with a manifest recording
endpoint, date, request count and sha256, and every downstream command replays from the committed
file with no network access. This is the precedent 004 set for its extraction cache, adopted for
the same reason — when the source is not deterministic, the artifact is what a reader re-runs.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
BATCH = 50  # the API's titles-per-request limit for unauthenticated clients
MAXLAG = 5  # seconds of replication lag above which WMF asks clients to back off
TIMEOUT = 60
MAX_RETRIES = 5
PAUSE = 0.1  # between requests, well inside WMF's guidance for a serial client

# Identifies the client to WMF as their policy asks. A generic agent is what gets rate-limited.
USER_AGENT = (
    "retrieval-baselines/005 (https://github.com/Catafal/retrieval-baselines; "
    "experiment 005 identity coverage) python-requests"
)

OUT = Path("results/005")


class FetchFailed(RuntimeError):
    """A batch that did not survive MAX_RETRIES. Collected, never swallowed."""


def _get(session: requests.Session, params: dict) -> dict:
    """One API call, retried on the failures that are worth retrying and no others.

    Retries 429 and 5xx with exponential backoff, honouring Retry-After when the server sends
    one. A 4xx other than 429 is a bug in the request and retrying it just repeats the bug.
    """
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(API, params=params, timeout=TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", delay))
                time.sleep(wait)
                delay *= 2
                continue
            r.raise_for_status()
            body = r.json()
            # maxlag is reported as a normal 200 with an error member, not an HTTP status.
            if "error" in body and body["error"].get("code") == "maxlag":
                time.sleep(delay)
                delay *= 2
                continue
            return body
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise FetchFailed(f"{params.get('titles', '')[:80]}: {exc}") from exc
            time.sleep(delay)
            delay *= 2
    raise FetchFailed(f"{params.get('titles', '')[:80]}: exhausted {MAX_RETRIES} attempts")


def _batch(session: requests.Session, titles: list[str]) -> tuple[dict[str, list[str]], int]:
    """Redirects pointing AT each of `titles`. Returns (canonical -> aliases, requests made).

    Continuation is followed rather than truncated. A heavily-redirected article can exceed the
    per-request limit, and silently keeping the first page would drop exactly the aliases a
    popular entity has most of.
    """
    found: dict[str, list[str]] = {}
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "redirects",
        "rdprop": "title",
        "rdnamespace": "0",  # R2: main namespace only
        "rdlimit": "max",
        "maxlag": str(MAXLAG),
        "titles": "|".join(titles),
    }
    calls = 0
    while True:
        body = _get(session, params)
        calls += 1
        for page in body.get("query", {}).get("pages", []):
            if page.get("missing"):
                continue
            aliases = [r["title"] for r in page.get("redirects", [])]
            if aliases:
                found.setdefault(page["title"], []).extend(aliases)
        if "continue" not in body:
            return found, calls
        params.update(body["continue"])
        time.sleep(PAUSE)


def fetch(titles: set[str], progress_every: int = 200) -> tuple[list[dict], dict]:
    """Every redirect targeting one of `titles`. Returns (rows, stats).

    Failures are collected and reported at the end rather than aborting the run: 2,200 requests
    is long enough that losing all of them to one bad batch is a real cost, and a partial fetch
    with its gap counted is more useful than nothing. The gap is recorded in the manifest, so a
    coverage figure can never be quoted as if the fetch had been complete.
    """
    ordered = sorted(titles)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    rows: list[dict] = []
    failures: list[str] = []
    calls = 0
    started = time.time()

    for i in range(0, len(ordered), BATCH):
        chunk = ordered[i:i + BATCH]
        try:
            found, made = _batch(session, chunk)
            calls += made
            for canonical, aliases in found.items():
                rows.append({"title": canonical, "redirects": sorted(set(aliases))})
        except FetchFailed as exc:
            failures.append(str(exc))
        if i and (i // BATCH) % progress_every == 0:
            done = i + len(chunk)
            rate = done / max(1e-9, time.time() - started)
            print(f"  {done}/{len(ordered)} titles  {calls} requests  "
                  f"{rate:.0f} titles/s  {len(failures)} failed batches", flush=True)
        time.sleep(PAUSE)

    stats = {
        "titles_requested": len(ordered),
        "titles_with_redirects": len(rows),
        "aliases_total": sum(len(r["redirects"]) for r in rows),
        "requests": calls,
        "failed_batches": len(failures),
        "failures": failures[:20],
        "seconds": round(time.time() - started, 1),
    }
    return rows, stats


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(corpus: str, titles: set[str]) -> dict:
    """Fetch and commit the snapshot plus its manifest. The only function here that writes."""
    print(f"005: fetching redirects for {len(titles):,} {corpus} pool titles", flush=True)
    rows, stats = fetch(titles)
    rows.sort(key=lambda r: r["title"])  # deterministic file for a deterministic sha256

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"redirects-{corpus}.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    manifest = {
        "corpus": corpus,
        "endpoint": API,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "namespace": 0,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        **stats,
    }
    (OUT / f"redirects-{corpus}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "failures"}, indent=2))
    return manifest


def load(corpus: str) -> dict[str, list[str]]:
    """Replay the committed snapshot. No network, and it must exist."""
    path = OUT / f"redirects-{corpus}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is the artifact of record for identity and is committed; "
            "a coverage number computed without it would not be reproducible."
        )
    out: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["title"]] = row["redirects"]
    return out


def pool_titles(corpus: str) -> set[str]:
    """The pool's own titles — R1's scope. Loaded through the same builders every scored run
    uses, so the identity registry is scoped to exactly the documents that get retrieved."""
    from rb import datasets
    from rb.experiments.graph import pool
    from rb.experiments.graph import run as graph_run

    if corpus == "hotpotqa":
        # Same route oracle.py takes: pool.build returns title -> doc_id for the pooled
        # documents only, which is the scope R1 asks for. datasets.load_titles alone would
        # hand back all 5.2M BEIR titles, almost none of which this graph can ever retrieve.
        ctx = pool.load_distractor_context()
        _, resolved = pool.build(datasets.load_corpus("hotpotqa"),
                                 datasets.load_titles("hotpotqa"), ctx)
        return set(resolved)
    if corpus == "2wiki":
        _, titles, _, _, _ = graph_run.load_pool_2wiki()
        return set(titles.values())
    raise ValueError(f"unknown corpus {corpus!r}")


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:] or ["hotpotqa", "2wiki"]:
        snapshot(name, pool_titles(name))
