"""
The deterministic entity extractor — protocols/003-graph-arm.md §2 and §8.

WHY spaCy AND NOT AN LLM. Three reasons, all in the protocol. It runs on a laptop so
`make reproduce` stays honest. It removes the "we measured our LLM" confound rather than
reporting around it. And a weak extractor still yields an interpretable result, because the
prediction is about a DIFFERENTIAL between query classes and a weak extractor degrades both
sides. The LLM extractor is the pre-registered upgrade, and it is experiment 004.

WHAT THIS IS NOT. spaCy performs named-entity recognition. HippoRAG and REBEL perform OpenIE
triple extraction — a different task producing a different graph. That is exactly why no
published retrieval row can gate this arm, and why §8.3's gate is bridge reachability instead.

PINNED, because the protocol requires the extractor fixed by version and model revision before
anything is scored. A different spaCy or a different model is a different experiment.
"""

import functools

from rb.experiments.graph.entity_types import WHITELIST, assert_partition

# Fixed before the first scored run. The model version is the one whose published OntoNotes
# per-type scores are transcribed in entity_types.py, so the reference figures there describe
# the artefact actually being run rather than a nearby one.
SPACY_VERSION = "3.8.13"
MODEL_NAME = "en_core_web_sm"
MODEL_VERSION = "3.8.0"

# Only the entity recogniser is needed. Parser and lemmatiser cost time on 66,581 passages and
# contribute nothing to a node set, so they are disabled — a speed decision, not a quality one,
# and it cannot change which entities are produced.
_DISABLED = ("parser", "lemmatizer", "tagger", "attribute_ruler")


@functools.lru_cache(maxsize=1)
def _nlp():
    """Load once. Verifies the pin rather than trusting the environment: an environment that
    has drifted must fail loudly here rather than quietly produce different entities."""
    import spacy

    if spacy.__version__ != SPACY_VERSION:
        raise RuntimeError(
            f"spaCy {spacy.__version__} installed, protocol pins {SPACY_VERSION}. "
            "A different version is a different experiment."
        )
    nlp = spacy.load(MODEL_NAME, exclude=list(_DISABLED))
    got = nlp.meta.get("version")
    if got != MODEL_VERSION:
        raise RuntimeError(
            f"{MODEL_NAME} {got} loaded, protocol pins {MODEL_VERSION}."
        )
    # The whitelist must partition the labels this model actually emits. Checked HERE
    # because this is the only place spaCy is guaranteed loaded, and it is on the path
    # of every scored run. entity_types.py cannot check it at import without importing
    # spaCy, which would destroy the "frozen before install" property that makes the
    # list credible. A type in neither set would be dropped from both sides of the §8.2
    # comparison and the symptom would be a number, not an error.
    assert_partition(nlp.get_pipe("ner").labels)
    return nlp


def manifest() -> dict:
    """What a stranger needs to tell an environment difference from a finding."""
    return {
        "extractor": "spacy-ner",
        "spacy_version": SPACY_VERSION,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "disabled_components": list(_DISABLED),
        "whitelist": sorted(WHITELIST),
    }


def extract(text: str) -> list[tuple[str, str]]:
    """(surface form, label) for one passage, unfiltered.

    Unfiltered on purpose: the whitelist is applied by
    `entity_types.filter_entities`, in one place, to BOTH the reference set and the
    predictions. Filtering here would make it easy to filter one side only, which is the
    single easiest way to make the §8.2 diagnostic wrong.
    """
    return [(ent.text, ent.label_) for ent in _nlp()(text).ents]


def extract_many(texts: dict[str, str], batch_size: int = 256) -> dict[str, list[tuple[str, str]]]:
    """
    doc_id -> entities, over a whole corpus.

    Uses spaCy's pipe rather than a loop: on 66,581 passages the per-call overhead dominates
    otherwise. Order is preserved by zipping against the same sorted key list the pipe consumed,
    so the mapping cannot silently misalign — the failure mode that would attach every document's
    entities to its neighbour and still look plausible.
    """
    ids = sorted(texts)
    docs = _nlp().pipe([texts[i] for i in ids], batch_size=batch_size)
    return {i: [(e.text, e.label_) for e in d.ents] for i, d in zip(ids, docs)}


def node_strings(entities: list[tuple[str, str]], kept=WHITELIST) -> list[str]:
    """
    The graph's node set for one document: whitelisted entities, deduplicated by surface form.

    Deduplicated because the graph keys nodes by string — a repeated mention is one node, not
    two — which is also why the reference set is annotated as a set.
    """
    seen, out = set(), []
    for text, label in entities:
        if label not in kept:
            continue
        s = text.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
