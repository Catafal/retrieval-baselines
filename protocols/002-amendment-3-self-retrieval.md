# Amendment 3 to protocol-002 — what the self-retrieval control actually does

**Status: frozen on tagging as `protocol-002-amendment-3`.** Written after the control halted a
run and before that run is repeated. The halt and the reasoning are recorded here rather than the
control being quietly adjusted until the run passed, which is the obvious way to abuse this.

## 1. What happened

The second encoder (`BAAI/bge-base-en-v1.5`, amendment 2) failed the self-retrieval control on
Quora: 1 of 100 sampled documents did not return itself at rank 1. The control halted the run and
nothing was written, which is the behaviour amendment 1 section 9 specifies.

MiniLM passed 100 of 100 on all three corpora. The second encoder passed 100 of 100 on SciFact.
The single failure is Quora with the second encoder only.

## 2. Diagnosis, evidenced before any change

The failing document is `100`, "How to make friends ?". Its nearest neighbours in the corpus are
"How can you make a friend?", "How can you make friends with people?", "How do I make friend?"
and "How do I make friends." Quora is a duplicate-question corpus, so a document having a
near-identical twin is the normal case rather than a defect.

The second encoder also applies an asymmetric query prefix (amendment 2 section 3), so a
document's own query embedding is deliberately not identical to its document embedding. Combine a
corpus of paraphrases with an encoder that shifts queries, and a paraphrase edging out the
document itself is expected rather than diagnostic. On a 40,000-document subset the same document
does return itself first; it is at full corpus scale, with more near-duplicates competing, that it
loses by a small margin.

## 3. The more important finding: the control does not do what amendment 1 claimed

Amendment 1 section 9 said self-retrieval "catches an encoder wired up wrongly in a way the
shuffle would not, for example query and document encoders transposed."

Measured, on 100 Quora documents with the second encoder:

| configuration | self-retrieval failures |
|---|---|
| correctly wired | 0 / 100 |
| query and document encoders transposed | 0 / 100 |

The control does not catch transposition. For this encoder the two paths differ only by a query
prefix, so swapping them leaves a document still very close to its own nearest neighbour. For
MiniLM the two paths are byte-identical, so it cannot catch it there either. That claim in
amendment 1 was wrong and is corrected here rather than left standing.

## 4. What the control is redefined to be

**Purpose:** an alignment check. It verifies that document ids, embedding rows and the similarity
computation line up, so a document can be found by its own text. That is a real failure mode and
the control does detect it, since a misaligned index fails almost every sample rather than one.

**Threshold:** at least 98 of 100 sampled documents must return themselves at rank 1. Chosen
because a genuine alignment failure produces near-total failure, not one or two, so the threshold
preserves everything the control detects while tolerating a near-duplicate corpus. It is not
chosen to make this particular run pass: at 1 failure the run passes with a margin of one, and any
run failing 3 or more still halts.

**What it explicitly does NOT catch:** query and document encoders transposed. That is covered
instead by `tests/test_dense.py`, which exercises the actual `retrieve()` call sites with a stub
encoder whose two paths disagree, and which fails when they are swapped. The protocol should not
claim a runtime control does something a unit test does.

## 5. Everything else unchanged

The embedding shuffle control, the closure gate, gold presence and the empty-query control are
untouched. The threshold change applies to self-retrieval only.

## 6. Disclosure

The entry states that this control halted a run, that the halt was investigated rather than
worked around, and that investigating it turned up a weaker guarantee than the protocol had
claimed. A control that fires and turns out to be mis-specified is worth more written down than
quietly retuned.
