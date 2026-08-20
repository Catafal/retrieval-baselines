"""
The extraction-quality annotation sample — protocols/003-graph-arm.md section 8.2.

WHY A SAMPLE AT ALL. The graph arm's closure control cannot be a published retrieval
number. HippoRAG and REBEL build their graphs with OpenIE triple extraction; this arm
uses spaCy named-entity recognition, which is a DIFFERENT task. Gating our arm against
their retrieval rows would repeat the mistake the first BM25 closure control made, where
a correct implementation failed because the reference was measured under different
conditions. So the graph arm is gated on what it can honestly be gated on: how well it
extracts, measured against annotation, reported before any retrieval score exists.

WHY IT IS DRAWN BEFORE THE EXTRACTOR RUNS. A sample chosen after seeing extraction
output is a sample chosen to flatter it. The seed is fixed in the protocol, the draw is
deterministic, and the drawn ids are committed, so anyone can check that the annotated
passages are the ones the seed selects.
"""

import json
import random
from pathlib import Path

# protocols/003-graph-arm.md section 8.2. Both frozen there before the draw.
SAMPLE_SEED = 20260820
SAMPLE_SIZE = 100


def draw(pool_corpus: dict[str, str], size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[str]:
    """
    Deterministic sample of document ids from the pooled corpus.

    Sorted before sampling because dict order is an accident of construction: sampling
    an unsorted iteration order would make the draw depend on how the pool was built
    rather than on the seed, and it would stop being reproducible the moment the loader
    changed. Same reasoning as the doc-id tie-break every ranking in this repo uses.
    """
    return random.Random(seed).sample(sorted(pool_corpus), size)


def write_template(pool_corpus: dict[str, str], titles: dict[str, str], out: Path,
                   size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> Path:
    """
    Write the sample as a JSONL annotation template, one passage per line.

    `entities` is left empty ON PURPOSE and is not pre-filled by any model. The point of
    this file is to be an independent reference for the extractor; seeding it with a
    model's guesses would make it a measure of agreement rather than of quality, and the
    number it produced would be worthless in exactly the way this repository exists to
    avoid.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        for doc_id in draw(pool_corpus, size, seed):
            f.write(json.dumps({
                "doc_id": doc_id,
                "title": titles.get(doc_id, ""),
                "text": pool_corpus[doc_id],
                "entities": [],
                "annotated": False,
            }) + "\n")
    return out
