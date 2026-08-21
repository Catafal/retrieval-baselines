"""
Experiment 003's candidate set: the HotpotQA distractor pool.

WHY THIS EXISTS. 002 measured dense losing to full BM25 on HotpotQA by 0.130 nDCG@10
and named a mechanism it could not isolate: the entity bridging the two gold documents
is absent from the question. Testing that needs a graph, a graph needs entity
extraction over the corpus, and BEIR's hotpotqa corpus is 5,233,329 documents — too
large to extract from at any extractor quality.

The first design answered that by re-ranking over the union of the other arms' top-k.
That pool is CIRCULAR in the worst possible direction: where BM25 and dense fail to
surface the bridge document, the graph cannot recover it however good its mechanism is,
and those are exactly the coverage-1 queries the experiment predicts a win on. The pool
would have suppressed the effect under test.

The distractor setting replaces it. HotpotQA ships, per question, the two supporting
passages plus eight distractors; pooling them over all 7,405 judged questions gives
66,581 unique passages. The candidate set is therefore defined by the DATASET rather
than by our own retrievers, which is what makes the restriction principled instead of
circular. It is also the recipe HippoRAG follows (its 9,221-passage HotpotQA corpus is
this same construction over a 1,000-question subset), so their published numbers become
a like-for-like reference.

WHAT MAKES THIS A SUBSET RATHER THAN A NEW CORPUS. Every pooled title resolves to a
document already in BEIR's hotpotqa corpus, uniquely, and the pooled documents keep
their BEIR document ids and BEIR text. So the pool is an exactly-identified subset of
the corpus 002 published on, scored by the same code against the same qrels. What
changes is the size of the haystack, and that change is declared: 003 does NOT reproduce
002's -0.130 and does not extend it (protocols/003-graph-arm.md section 3).

Counts are asserted, never trusted. Deduping 73,700 title slots into 66,581 uniques and
mapping them onto BEIR ids is precisely the indexing step that produces unreproducible
numbers, and this repository exists because of a retraction for unreproducible numbers.
rb.controls.pool_construction turns every count below into a control that halts the run.
"""

import hashlib
import json
from pathlib import Path

# Measured before tagging and frozen in protocols/003-graph-arm.md section 3. These are
# expectations the run is CHECKED against, not values derived from whatever the data
# happens to contain — a count that recomputes itself from the outcome cannot fail.
EXPECTED_QUESTIONS = 7405
EXPECTED_PASSAGES = 66581
EXPECTED_TITLE_SLOTS = 73700

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "manifests" / "datasets.json"

# HotpotQA's distractor-setting validation split, which is the same 7,405 questions as
# BEIR hotpotqa's test qrels (verified: 7,405/7,405 overlap in both directions).
DISTRACTOR_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
)
DISTRACTOR_FILE = "hotpotqa_distractor_validation.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_distractor() -> Path:
    """Fetch the distractor split, recording its hash. Same discipline as rb.datasets:
    a hash that changes under us invalidates every number published against it."""
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / DISTRACTOR_FILE
    if not path.exists():
        with requests.get(DISTRACTOR_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    digest = _sha256(path)
    key = "hotpotqa_distractor"
    if key in manifest and manifest[key]["sha256"] != digest:
        raise RuntimeError(
            f"{DISTRACTOR_FILE} hash changed: manifest has {manifest[key]['sha256']}, "
            f"file is {digest}. Published results reference the manifest hash."
        )
    manifest[key] = {"sha256": digest, "bytes": path.stat().st_size}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_distractor_context() -> dict[str, list[str]]:
    """
    question_id -> the titles of its ten candidate passages, in file order.

    Imports pyarrow lazily so the rest of the suite never needs it installed, the same
    convention rb.experiments.ladder's encoder follows.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(download_distractor(), columns=["id", "context"])
    return {row["id"]: list(row["context"]["title"]) for row in table.to_pylist()}


def pool_titles(context: dict[str, list[str]]) -> tuple[set[str], int]:
    """Unique pooled titles, and the total number of title slots they were deduped from.

    Both are returned because only their PAIR is diagnostic: the unique count alone
    cannot distinguish a correct dedup from a loader that silently dropped rows."""
    slots = 0
    titles: set[str] = set()
    for ts in context.values():
        slots += len(ts)
        titles.update(ts)
    return titles, slots


def _index_and_collisions(corpus_titles: dict[str, str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    The non-raising core of `title_index`: (title -> doc_id, duplicate titles).

    Split out so the same traversal can serve two callers that need OPPOSITE behaviour — the
    build path, which must halt on a duplicate, and the control, which must be able to REPORT
    a duplicate count. Before NB-26 the control could not: every path that computed collisions
    raised before returning one, so `run.py` passed a hardcoded 0 and a reader saw an unmeasured
    number inside a `"passed": true` block.
    """
    index: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for doc_id, title in corpus_titles.items():
        if not title:
            continue
        if title in index:
            collisions.setdefault(title, [index[title]]).append(doc_id)
        else:
            index[title] = doc_id
    return index, collisions


def title_index(corpus_titles: dict[str, str]) -> dict[str, str]:
    """
    title -> doc_id over a corpus, refusing to build if a title is ambiguous.

    A duplicate title would make the pool's identity depend on iteration order, which
    is how a run stops being reproducible without anyone noticing. Measured on BEIR
    hotpotqa restricted to the pooled titles: zero collisions. That is a property of
    the data, so it is checked rather than assumed.
    """
    index, collisions = _index_and_collisions(corpus_titles)
    if collisions:
        example = sorted(collisions)[:3]
        raise RuntimeError(
            f"{len(collisions)} duplicate titles in the corpus; the pool would not be "
            f"reproducible. Examples: { {t: collisions[t] for t in example} }"
        )
    return index


def construction_counts(corpus_titles: dict[str, str], context: dict[str, list[str]],
                        qrels: dict[str, dict[str, int]]) -> dict:
    """
    The §9 control's inputs, MEASURED — every one of them, without raising.

    WHY THIS EXISTS. `run.py` used to pass `unresolved=0, collisions=0` as literals and
    `gold_titles_matched=len(qrels)` alongside `gold_queries=len(qrels)` — the same expression
    twice, so that check was `len(qrels) == len(qrels)` and could not fail. Three of the
    control's seven published fields were therefore asserted rather than measured, inside a
    block reporting `"passed": true`, in a repository whose premise is that its controls halt
    the run.

    It is deliberately NON-RAISING and is called BEFORE `build()`. `build()` and `title_index`
    keep their own raises as defence in depth, but they raise before they could ever return a
    nonzero count — which is why relocating the literal would not have fixed anything. Only a
    path that observes the counts without exploding can let the control fail on its own terms.

    `gold_titles_matched` is the real intersection its name always claimed: judged queries whose
    EVERY gold document id is present in the pooled corpus. That property is what makes the pool
    an exactly-identified subset, and until now nothing in the repository tested it.
    """
    titles, slots = pool_titles(context)
    index, collisions = _index_and_collisions({d: t for d, t in corpus_titles.items() if t in titles})
    resolved, unresolved = resolve(titles, index)
    # Derives the pooled document ids itself rather than taking them from `build()`. That is what
    # lets this run BEFORE build(), which matters: build() raises on an unresolved title, so a
    # control called afterwards could never observe a nonzero count and would be measuring a
    # quantity that is zero by control flow rather than by fact.
    pool_doc_ids = set(resolved.values())
    matched = sum(1 for docs in qrels.values() if docs and all(d in pool_doc_ids for d in docs))
    return {
        "questions": len(context),
        "title_slots": slots,
        "unresolved": len(unresolved),
        "collisions": len(collisions),
        "gold_titles_matched": matched,
        "gold_queries": len(qrels),
    }


def resolve(titles: set[str], index: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    """Map pooled titles onto corpus document ids. Returns (title -> doc_id, unresolved)."""
    resolved = {t: index[t] for t in titles if t in index}
    return resolved, titles - set(resolved)


def build(corpus: dict[str, str], corpus_titles: dict[str, str],
          context: dict[str, list[str]]) -> tuple[dict[str, str], dict[str, str]]:
    """
    The pooled corpus, keyed by BEIR document id and carrying BEIR text.

    Returns (pool_corpus, title_to_doc_id). Raises if any pooled title is missing from
    the corpus: an unresolved title means the pool is not a subset, which is the whole
    claim this construction rests on.
    """
    titles, _slots = pool_titles(context)
    index = title_index({d: t for d, t in corpus_titles.items() if t in titles})
    resolved, unresolved = resolve(titles, index)
    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} pooled titles are absent from the corpus, so the pool is "
            f"not a subset of it. Examples: {sorted(unresolved)[:5]}"
        )
    return {doc_id: corpus[doc_id] for doc_id in resolved.values()}, resolved
