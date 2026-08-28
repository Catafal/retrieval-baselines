"""
The offline check on 004's ablation artifact must be able to FAIL.

NB-24's mutation sweep found three tests that passed against a deliberately broken
implementation. A checker that reads an artifact and recomputes from the same artifact is
exactly the shape that goes vacuous quietly, so each committed figure is perturbed here and
the check is required to notice.
"""

import json
from pathlib import Path

import pytest

from rb.experiments.graph import reasoning_ablation as ra

ARTIFACT = ra.OUT / "reasoning-ablation.json"


@pytest.fixture
def restore():
    """Snapshot the real artifact and put it back, whatever the test does to it."""
    original = ARTIFACT.read_bytes()
    yield
    ARTIFACT.write_bytes(original)


def test_committed_artifact_passes_its_own_check():
    assert ra.check() == []


@pytest.mark.parametrize(
    "path",
    [
        ("queries", "reasoning_off", "empty_rate_ci95"),
        ("queries", "reasoning_off", "empty_rate"),
        ("queries", "mde_at_80_power"),
        ("passages", "reasoning_on", "tp"),
        ("passages", "spacy", "f1"),
        ("passages", "paired_f1_difference", "ci95"),
    ],
)
def test_check_notices_a_moved_figure(path, restore):
    d = json.loads(ARTIFACT.read_text())
    node = d
    for key in path[:-1]:
        node = node[key]
    value = node[path[-1]]
    if isinstance(value, list):
        node[path[-1]] = [x + 0.01 for x in value]
    elif isinstance(value, int):
        node[path[-1]] = value + 1
    else:
        node[path[-1]] = round(value + 0.01, 4)
    ARTIFACT.write_text(json.dumps(d, indent=2) + "\n")

    assert ra.check(), f"moving {'.'.join(path)} did not fail the check"


def test_pooled_counts_must_match_per_passage(restore):
    """The two halves of the artifact must describe one run, not two."""
    d = json.loads(ARTIFACT.read_text())
    first = next(iter(d["passages"]["per_passage"]["reasoning_off"]))
    d["passages"]["per_passage"]["reasoning_off"][first]["tp"] += 1
    ARTIFACT.write_text(json.dumps(d, indent=2) + "\n")

    assert any("pooled" in m for m in ra.check())
