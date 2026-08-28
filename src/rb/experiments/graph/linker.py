"""
Typed identity — protocols/005-identity.md sections 4 and 5.

WHAT THIS REPLACES. 003 and 004 resolve an entity to a graph node with `normalise`: lowercase,
strip punctuation, collapse whitespace. Two surface forms are the same node when their normalised
strings are equal, and otherwise they are not, however obviously they name one thing. A linker
adds one step in front of that: a normalised surface form may first resolve to a canonical
identity, and the node is keyed by the canonical.

THE INVARIANT THAT MAKES THE COMPARISON AN EXPERIMENT. With an empty registry `link` is
`normalise`, exactly. The string arm and the typed arm are therefore the same code path carrying
different data, not two programs whose difference includes whatever else drifted between them.
Anything that matched before still matches: typed identity is a strict superset (R3).

WHY THE DROPS EXIST. Merging is not free. Two entities that share a name and are not the same
thing produce a path that does not exist, and a graph retriever will happily walk it. R5 and R6
are the registered guards, fixed before coverage was measured, and both are counted rather than
applied silently — a registry that quietly discarded a third of its aliases would report high
precision for a reason the reader could not see.
"""

from rb.experiments.graph.extraction_score import normalise


def build_registry(
    redirects: dict[str, list[str]], pool_titles: set[str]
) -> tuple[dict[str, str], dict]:
    """
    Alias -> canonical, per protocol 005 section 4. Returns (registry, counts).

    `redirects` maps a canonical article title to the titles that redirect to it, as committed by
    `redirects.snapshot`. `pool_titles` is the corpus's own title set, needed by R5.
    """
    normalised_pool = {normalise(t) for t in pool_titles}

    # First pass: collect every canonical each alias could resolve to, so R6 can see ambiguity
    # across the whole corpus rather than in the order the file happened to be written.
    candidates: dict[str, set[str]] = {}
    for canonical, aliases in redirects.items():
        target = normalise(canonical)
        if not target:
            continue
        for alias in aliases:
            key = normalise(alias)
            if not key:
                continue
            candidates.setdefault(key, set()).add(target)

    registry: dict[str, str] = {}
    counts = {
        "aliases_seen": len(candidates),
        "dropped_identity": 0,     # alias normalises to its own canonical: not a merge
        "dropped_self_title": 0,   # R5
        "dropped_ambiguous": 0,    # R6
        "kept": 0,
    }

    for key, targets in candidates.items():
        # R6 first. An ambiguous alias is dropped whatever else is true of it, and checking it
        # before R5 keeps the two counts disjoint so they can be read as a partition.
        if len(targets) > 1:
            counts["dropped_ambiguous"] += 1
            continue
        target = next(iter(targets))
        if key == target:
            counts["dropped_identity"] += 1
            continue
        # R5. The alias is itself a document in this pool, so merging it away would erase that
        # document's own identity — the graph would lose a node it is expected to retrieve.
        if key in normalised_pool:
            counts["dropped_self_title"] += 1
            continue
        registry[key] = target
        counts["kept"] += 1

    counts["canonicals"] = len(set(registry.values()))
    return registry, counts


def link(text: str, registry: dict[str, str]) -> str:
    """
    One surface form to one node key.

    An empty registry makes this `normalise` — see the module docstring. The fallback is the
    normalised string itself rather than a miss, because an entity with no alias is still an
    entity and still needs its node.
    """
    key = normalise(text)
    return registry.get(key, key)


def linker(registry: dict[str, str]):
    """A one-argument linker bound to `registry`, which is the shape build() and _seed() take."""
    return lambda text: link(text, registry)


def apply_df_cap(registry: dict[str, str], docs: dict[str, list[str]],
                 corpus_size: int, fraction: float = 0.01) -> tuple[dict[str, str], dict]:
    """
    R9 — protocols/005-amendment-2-hub-cap.md. Drop any canonical whose MERGED document
    frequency would exceed `fraction` of the corpus, reverting all its aliases to string identity.

    Why a cap at all: node specificity is 1/document-frequency and HippoRAG's ablation shows it is
    load-bearing. A merge that lands a tenth of the corpus on one node drives that node's
    specificity to nearly zero, and every query naming it seeds into a node that says almost
    nothing about which document is wanted. `america -> united states` is a correct redirect and,
    at ~6,669 documents, exactly that failure.

    Applied AFTER build_registry rather than inside it, so R1-R8 stay exactly as they were tagged
    and the capped arm differs from the uncapped one by this rule alone.
    """
    cap = int(corpus_size * fraction)

    merged: dict[str, set] = {}
    for doc, ents in docs.items():
        for surface in ents:
            key = normalise(surface)
            if key:
                merged.setdefault(registry.get(key, key), set()).add(doc)

    over = {c for c, ds in merged.items() if len(ds) > cap}
    kept = {k: c for k, c in registry.items() if c not in over}
    return kept, {
        "cap_fraction": fraction,
        "cap_documents": cap,
        "canonicals_over_cap": len(over),
        "aliases_reverted": len(registry) - len(kept),
        "aliases_kept": len(kept),
    }
