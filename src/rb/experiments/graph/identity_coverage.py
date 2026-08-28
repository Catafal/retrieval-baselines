"""
Stage 0's only measurement — protocols/005-identity.md section 6.

Reports what the redirect registry covers, BEFORE any retrieval number exists. The ordering is
the point: a coverage figure read after a score is a figure that can be argued with, and the
protocol froze section 4's construction rules precisely so this file cannot become a knob.

WHAT IS AND IS NOT HERE. No R@2, no BM25, no Holm family, no prediction. Section 8 of the
protocol lists those as out of scope for this stage, and the absence is deliberate rather than
unfinished.

THE FLOOR IS AN MDE. No coverage percentage is declared sufficient in advance. What is reported
instead is the smallest concentration effect the alias-affected subset could resolve at 80%
power. A threshold invites renegotiation once the number lands; a minimum detectable effect is a
property of the sample size and computes the same way whatever the answer turns out to be.
"""

import json
from collections import Counter
from pathlib import Path

from rb import datasets
from rb.experiments.graph import llm_extractor as llm
from rb.experiments.graph import pool, pool2wiki, redirects
from rb.experiments.graph.extraction_score import normalise
from rb.experiments.graph.extractor import extract as spacy_extract
from rb.experiments.graph.extractor import extract_many as spacy_extract_many
from rb.experiments.graph.extractor import node_strings
from rb.experiments.graph.linker import build_registry
from rb.stats import mde_two_proportion

OUT = Path("results/005")


def _load(corpus: str):
    """(passages, queries, pool_titles) for one corpus, by the same route every scored run uses."""
    if corpus == "hotpotqa":
        ctx = pool.load_distractor_context()
        passages, resolved = pool.build(datasets.load_corpus("hotpotqa"),
                                        datasets.load_titles("hotpotqa"), ctx)
        qrels = datasets.load_qrels("hotpotqa")
        queries = {q: t for q, t in datasets.load_queries("hotpotqa").items() if q in qrels}
        return passages, queries, set(resolved)
    passages, titles, queries, qrels = pool2wiki.build()
    queries = {q: t for q, t in queries.items() if q in qrels}
    return passages, queries, set(titles.values())


def _entities(corpus: str, passages: dict, queries: dict, extractor: str):
    """Whitelisted entity strings per document and per query, for one extractor.

    Both extractors are replayed rather than re-run: spaCy is deterministic and local, and GLM
    reads 004's committed cache. Stage 0 spends nothing and calls no model.
    """
    if extractor == "spacy":
        docs = {d: node_strings(e) for d, e in spacy_extract_many(passages).items()}
        qents = {q: node_strings(spacy_extract(t)) for q, t in queries.items()}
        return docs, qents
    docs = {d: node_strings(e) for d, e in llm.extract_docs_offline(passages).items()}
    qents = {q: node_strings(llm.extract_query_offline(t)) for q, t in queries.items()}
    return docs, qents


def measure(corpus: str, extractor: str, registry: dict[str, str], drops: dict) -> dict:
    """C1 to C6 plus the MDE, for one corpus under one extractor."""
    passages, queries, _ = _load(corpus)
    docs, qents = _entities(corpus, passages, queries, extractor)

    # C1 / C2 — the node set, before and after.
    before = {normalise(s) for ents in docs.values() for s in ents if normalise(s)}
    hit = {n for n in before if n in registry}
    after = {registry.get(n, n) for n in before}

    # C4 — how many distinct pre-merge nodes land on each canonical actually reached. Counted
    # over nodes the graph really has, not over the registry, which mostly describes forms this
    # corpus never produced.
    merged = Counter(registry[n] for n in hit)
    sizes = Counter(merged.values())

    # C3 / C5 — the query side. See protocols/005-identity-amendment-1.
    #
    # A query is alias-affected when one of its entities REACHES A DIFFERENT SET OF DOCUMENTS
    # under typed identity than under string identity. That is the mechanism's own definition:
    # the walk starts at seeds, and what a seed can reach is exactly what identity changed.
    #
    # The narrow reading — queries that themselves name an alias — is kept alongside it because
    # it is what section 6 originally said, and because the gap between the two figures IS the
    # amendment. It misses the commoner case: the query names the canonical, a document named an
    # alias, and the merge makes that document reachable without the query entity ever being a
    # registry key.
    docs_string: dict[str, set] = {}
    docs_typed: dict[str, set] = {}
    for doc, ents in docs.items():
        for surface in ents:
            key = normalise(surface)
            if not key:
                continue
            docs_string.setdefault(key, set()).add(doc)
            docs_typed.setdefault(registry.get(key, key), set()).add(doc)

    q_entity_total = q_entity_hit = 0
    affected = affected_narrow = 0
    for ents in qents.values():
        keys = {normalise(s) for s in ents if normalise(s)}
        q_entity_total += len(keys)
        hits = keys & registry.keys()
        q_entity_hit += len(hits)
        affected_narrow += bool(hits)
        affected += any(
            docs_typed.get(registry.get(k, k), frozenset()) != docs_string.get(k, frozenset())
            for k in keys
        )

    n_queries = len(qents)
    return {
        "corpus": corpus,
        "extractor": extractor,
        "nodes": {
            "before": len(before),
            "after": len(after),
            "resolving_through_alias": len(hit),
            "share_resolving": round(len(hit) / len(before), 4) if before else 0.0,
            "reduction": len(before) - len(after),
        },
        "merge_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "queries": {
            "n": n_queries,
            "entities": q_entity_total,
            "entities_resolving": q_entity_hit,
            "share_entities_resolving": (
                round(q_entity_hit / q_entity_total, 4) if q_entity_total else 0.0),
            "alias_affected": affected,
            "share_alias_affected": round(affected / n_queries, 4) if n_queries else 0.0,
            "alias_affected_narrow": affected_narrow,
            "share_alias_affected_narrow": (
                round(affected_narrow / n_queries, 4) if n_queries else 0.0),
        },
        # The registered floor. Computed on the alias-affected subset because that is the
        # population the mechanism claim lives in; the full query set could resolve a smaller
        # effect but is not where the effect is predicted.
        "mde_at_80_power_on_affected": (
            mde_two_proportion(affected) if affected else None),
        "registry": drops,
    }


STATUS = ("Stage 0 of protocol 005. Coverage only — no retrieval number is produced here, "
          "and section 8 forbids one.")
CORPORA = ("hotpotqa", "2wiki")
ARMS = ("spacy", "glm")


def _part_path(corpus: str, extractor: str) -> Path:
    return OUT / f"identity-coverage-{corpus}-{extractor}.json"


def run_one(corpus: str, extractor: str) -> dict:
    """One (corpus, extractor) cell, written to its own file.

    Split into cells rather than run as one job because the spaCy passes are ten minutes each
    and an interrupted whole-corpus run loses all four. Each cell is independent, so a rerun
    costs one cell rather than the set.
    """
    _, _, titles = _load(corpus)
    registry, drops = build_registry(redirects.load(corpus), titles)
    cell = measure(corpus, extractor, registry, drops)
    OUT.mkdir(parents=True, exist_ok=True)
    _part_path(corpus, extractor).write_text(json.dumps(cell, indent=2) + "\n")
    return cell


def combine() -> dict:
    """Assemble the four cells into the artifact. Every cell must exist; a partial coverage
    figure reported as if it were complete is the failure this refuses to allow."""
    result = {"status": STATUS, "corpora": {}}
    for corpus in CORPORA:
        manifest = json.loads((OUT / f"redirects-{corpus}-manifest.json").read_text())
        arms = {}
        for extractor in ARMS:
            path = _part_path(corpus, extractor)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing. Run `python -m rb.experiments.graph.identity_coverage "
                    f"{corpus} {extractor}` first — coverage is reported for all four cells or "
                    "not at all."
                )
            arms[extractor] = json.loads(path.read_text())
        result["corpora"][corpus] = {
            "snapshot": {k: manifest[k] for k in
                         ("fetched_utc", "sha256", "titles_requested",
                          "titles_with_redirects", "aliases_total", "failed_batches")},
            "arms": arms,
        }
    return result


def run() -> dict:
    """Every cell, then the artifact. Cells already on disk are reused."""
    for corpus in CORPORA:
        for extractor in ARMS:
            if not _part_path(corpus, extractor).exists():
                run_one(corpus, extractor)
    return combine()


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) == 2:
        print(json.dumps(run_one(*args), indent=2))
    else:
        r = run() if "--combine" not in sys.argv else combine()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "identity-coverage.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r, indent=2))
