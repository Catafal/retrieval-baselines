"""
Prediction D compares two corpora. It must not compare two retrievers.

`analysis2wiki` quotes HotpotQA's figures as module constants so a 2Wiki run cannot silently
alter them — a good property that became a defect the moment the module was pointed at a second
arm. The constants are 003's spaCy numbers, so running the GLM arm left prediction D computing
GLM-on-2Wiki against spaCy-on-HotpotQA, and it reported `supported: true` on that mismatch.
Corrected, the same run reports 3.97 shrink points against a registered threshold of 10 and is
refuted.

The failure is worth a test rather than a comment because it is invisible in the output: both
numbers are real, both are published, and the crossover reads as a finding either way.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _crossover(name: str) -> dict:
    path = ROOT / "results" / "003" / "2wiki" / name
    return json.loads(path.read_text())["prediction_d_crossover"]


def test_each_arms_crossover_quotes_its_own_hotpotqa_figure():
    spacy = _crossover("analysis.json")
    glm = _crossover("analysis-graph-glm.json")

    spacy_summary = json.loads((ROOT / "results/003/pool/graph/summary.json").read_text())
    glm_summary = json.loads((ROOT / "results/003/pool/graph-glm/summary.json").read_text())

    assert spacy["hotpotqa_graph_recall_2"] == round(spacy_summary["ranked"]["recall_2"], 4)
    assert glm["hotpotqa_graph_recall_2"] == round(glm_summary["ranked"]["recall_2"], 4)
    assert spacy["hotpotqa_graph_recall_2"] != glm["hotpotqa_graph_recall_2"], (
        "if these are equal the arms are not distinguished and the guard proves nothing"
    )


def test_the_glm_crossover_is_refuted_not_supported():
    """Pins the corrected verdict, since the uncorrected one said the opposite."""
    glm = _crossover("analysis-graph-glm.json")
    assert glm["supported"] is False
    assert glm["shrink_points"] < glm["registered_threshold_points"]
