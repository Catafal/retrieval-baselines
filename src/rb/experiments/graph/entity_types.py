"""
The entity-type whitelist — protocols/003-graph-arm.md section 8.2.

FROZEN BEFORE spaCy IS INSTALLED. Not merely before it is run: at the time this file was
committed, spaCy was absent from the environment entirely, so no output of the extractor
could have informed the list. That is checkable from the repository history rather than
asserted here.

WHY RESTRICT AT ALL. An extracted entity earns its place only if it can (a) be matched
from a query and (b) connect two documents. Measured against the corpus itself: of all
13,783 gold-document titles in HotpotQA — which ARE the bridge entities the experiment is
about — exactly 2 (0.01%) are purely numeric. Dates and cardinals do not bridge anything.
Including them would measure a capability the graph does not use.

WHY THIS IS NOT GERRYMANDERING, WITH THE NUMBERS. The obvious objection is that a
whitelist chosen by the author flatters the extractor. The opposite is true here, and it
is measurable in advance from spaCy's own published per-type OntoNotes scores for the
pinned model. The types this list EXCLUDES are among spaCy's strongest; the types it
KEEPS include its four weakest:

    excluded   MONEY 0.910  PERCENT 0.896  DATE 0.867  CARDINAL 0.841  ORDINAL 0.820
    kept       GPE 0.904  NORP 0.903  PERSON 0.880  ORG 0.805  LOC 0.668
               LANGUAGE 0.690  LAW 0.417  EVENT 0.406  WORK_OF_ART 0.393  FAC 0.349
               PRODUCT 0.309

Overall published ents_f is 0.843. Restricting to this whitelist should therefore push
the measured number DOWN, not up. A restriction that costs the extractor points is not a
restriction chosen to flatter it.

ONE HONEST QUALIFICATION. The whitelist borrows spaCy's own OntoNotes label vocabulary,
so "frozen before install" defends against contamination by THIS run's predictions, not
against general prior knowledge of the schema. The list is grounded in the title
distribution above rather than in any calibration of the extractor, and that is the claim
being made — not naive blindness.
"""

# Types that can serve as graph nodes: things a query can name and two documents can share.
WHITELIST = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
})

# Measures and quantities. Excluded for the reason above, recorded explicitly so the
# exclusion is a decision in the artifact rather than an absence a reader must infer.
EXCLUDED = frozenset({
    "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL",
})

# spaCy's published OntoNotes 5 scores for the pinned model, transcribed from
# explosion/spacy-models meta for en_core_web_sm 3.8.0. REPORTED AS CONTEXT, NEVER AS A
# GATE: OntoNotes is newswire and this corpus is Wikipedia intros, so it is a figure
# measured under different conditions — the same reason the first BM25 closure control
# failed a correct implementation, and the reason HippoRAG's retrieval rows cannot gate
# this arm either.
SPACY_ONTONOTES_ENTS_F = 0.8433


def assert_partition(labels=None) -> None:
    """
    Check the whitelist partition. Two claims, checkable only in different places, and the
    argument is what separates them.

    WITHOUT `labels` — disjointness only. That is honestly all this module can verify on its
    own: it must not import spaCy, because "frozen before spaCy was installed" is the property
    that makes the list credible, and it stays checkable from git history only while the module
    has no dependency on the thing it predates. Runs at import, costs nothing.

    WITH `labels` — additionally that the two sets exactly cover the model's real label
    inventory. A type in neither set is silently dropped from both sides of the §8.2
    comparison; a type declared here but absent from the model means this list describes a
    different artefact than the one running.

    The second form is called from `extractor._nlp()`, which already loads the model and
    already asserts the version pin. That placement is deliberate. The obvious alternative — a
    hand-written SPACY_NER_LABELS constant checked at import — is self-defeating: an editor who
    breaks the partition simply edits the third constant to match, because the check would only
    compare three siblings in this file against each other and never against reality. Nor can
    the real check live in a test that skips when spaCy is absent: this repository has no CI,
    so a skipped test is enforced nowhere. Checking at model load puts it on the path every
    scored run takes, with nothing to drift.
    """
    overlap = WHITELIST & EXCLUDED
    if overlap:
        raise RuntimeError(f"a type cannot be both kept and excluded: {sorted(overlap)}")
    if labels is None:
        return
    declared = WHITELIST | EXCLUDED
    actual = frozenset(labels)
    uncovered = actual - declared
    if uncovered:
        raise RuntimeError(
            f"the model emits entity types this whitelist classifies neither way: "
            f"{sorted(uncovered)}. A type in neither set is dropped from both sides."
        )
    phantom = declared - actual
    if phantom:
        raise RuntimeError(
            f"this whitelist declares entity types the model never emits: "
            f"{sorted(phantom)}. The list describes a different model than the one loaded."
        )


def filter_entities(entities, kept=WHITELIST):
    """
    Keep only whitelisted types, from an iterable of (text, label) pairs.

    APPLIED TO BOTH SIDES. Filtering only the gold set while scoring the extractor's raw
    output would turn every correctly-found DATE into a false positive and collapse
    precision for a reason that has nothing to do with extraction quality. Symmetric
    filtering is the whole point of this function existing rather than being inlined.
    """
    return [(text, label) for text, label in entities if label in kept]


# Enforced at import, not left as a function nobody calls. This catches the half checkable
# here — an edit making a type both kept and excluded fails the moment the module loads. The
# COVERAGE half needs the model's real label inventory and runs in `extractor._nlp()`; see
# assert_partition's docstring for why it lives there rather than behind a hand-written
# constant or a test that skips when spaCy is missing.
assert_partition()
