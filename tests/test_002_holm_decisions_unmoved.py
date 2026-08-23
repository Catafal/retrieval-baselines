"""
Correcting 002's p-values must not change what 002 concluded.

On 2026-08-23 `paired_bootstrap` moved from the naive two-sided statistic to the add-one
estimator, and all eight of 002's committed analysis artifacts were regenerated. Every
p-value in them shifted slightly; 75 of them had been exactly 0.0, which is a value no
finite resampling procedure can produce.

That is defensible ONLY if it is a reporting correction — if no Holm decision moves. This
test is the standing evidence for that claim, so a future change to the estimator, the Holm
implementation, or the round count cannot quietly flip a published decision.

`tests/fixtures/002_published_holm_decisions.json` freezes the decisions as they were
PUBLISHED, read from the artifacts at commit HEAD before the regeneration. Those are
historical facts and must never be edited to make this test pass. If this test fails, the
correction has changed a conclusion and the entry needs rewriting, not the fixture.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "002_published_holm_decisions.json"


def _holm_blocks(obj, path=""):
    """Every dict in the tree carrying at least one key starting with 'holm'."""
    if isinstance(obj, dict):
        if any(k.startswith("holm") for k in obj):
            yield path, obj
        for k, v in obj.items():
            yield from _holm_blocks(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _holm_blocks(v, f"{path}[{i}]")


def _artifact_for(fixture_key: str) -> Path:
    # Fixture keys are the artifact paths with separators flattened, e.g.
    # results_002_quora_lexical_factorial.json -> results/002/quora/lexical_factorial.json
    stem = fixture_key[: -len(".json")]
    _, exp, dataset, *rest = stem.split("_", 3)
    return REPO / "results" / exp / dataset / (rest[0] + ".json")


def test_no_published_holm_decision_moved():
    published = json.loads(FIXTURE.read_text())
    assert published, "fixture is empty — it is meant to freeze 24 families"

    moved, checked = [], 0
    for key, families in published.items():
        artifact = _artifact_for(key)
        assert artifact.exists(), f"{artifact} is gone; 002's artifacts are published"
        current = dict(_holm_blocks(json.loads(artifact.read_text())))

        for path, decisions in families.items():
            assert path in current, f"{key}: Holm family {path} disappeared"
            for name, was in decisions.items():
                checked += 1
                now = current[path].get(name)
                if now != was:
                    moved.append(f"{key}{path}/{name}: {was} -> {now}")

    assert checked == 24, f"expected 24 frozen decisions, compared {checked}"
    assert not moved, "a published Holm decision moved:\n  " + "\n  ".join(moved)


def test_no_002_artifact_still_reports_an_impossible_p_value():
    """The defect itself, asserted against the shipped artifacts rather than the function."""
    zeros = []
    for artifact in sorted((REPO / "results" / "002").rglob("*.json")):
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("p_value", "p", "p_adj") and isinstance(v, (int, float)) and v == 0.0:
                        zeros.append(f"{artifact.name}{path}/{k}")
                    walk(v, f"{path}/{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")

        walk(json.loads(artifact.read_text()))

    assert not zeros, f"{len(zeros)} impossible p-values remain: {zeros[:5]}"
