"""
Which arm won must be frozen, not only whether the win was significant.

WHY THIS FILE EXISTS, AND WHY test_002_holm_decisions_unmoved.py IS NOT ENOUGH.
`paired_bootstrap`'s p-value is symmetric under swapping its two arguments: the add-one
estimator's `min(c_le + 1, c_ge + 1)` is invariant to negating every difference, so
`paired_bootstrap(a, b)` and `paired_bootstrap(b, a)` return the SAME p_value and differ only
in the sign of mean_diff. Holm significance is computed from p_value alone. Therefore swapping
which arm is A and which is B in `run_dense_hybrid_analysis`'s comparison tuples leaves every
`holm_significant` boolean identical, and the Holm fixture cannot detect it — not by oversight,
but structurally.

A mutation sweep confirmed the consequence: swapping `dense_ndcg` and `bm25_ndcg` in the
`dense_vs_full_bm25` tuple survives the entire test suite. So does inverting the
`better_component_name` selection. Those functions had no test of their own at all.

The entry's central claim is a DIRECTION — "the same encoder beats a lazy baseline everywhere
and a real one almost nowhere". A sign error is therefore the defect that would do the most
damage while looking the most normal, and it was the one thing nothing checked.

`tests/fixtures/002_published_comparison_directions.json` freezes the sign of every published
mean_diff. Like the Holm fixture, it records history and must never be edited to make a test
pass.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "002_published_comparison_directions.json"


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def test_no_published_comparison_changed_direction():
    published = json.loads(FIXTURE.read_text())
    assert published, "fixture is empty — it is meant to freeze 24 comparisons"

    flipped, checked = [], 0
    for rel, blocks in published.items():
        artifact = REPO / rel
        assert artifact.exists(), f"{artifact} is gone; 002's artifacts are published"
        current = json.loads(artifact.read_text())

        for key, rows in blocks.items():
            assert key in current, f"{rel}: {key} disappeared"
            assert len(current[key]) == len(rows), f"{rel}/{key}: comparison count changed"
            for i, row in enumerate(rows):
                checked += 1
                now = current[key][i]
                # dense/hybrid artifacts carry `comparison`; the lexical factorial names its
                # pair as from/to rungs. Both encode which arms and in which order.
                now_label = now["comparison"] if "comparison" in now else f'{now["from"]} -> {now["to"]}'
                # The label carries which arms are being compared and in which order, so a
                # relabelled comparison is as much a direction change as a flipped sign.
                assert now_label == row["comparison"], (
                    f"{rel}/{key}[{i}]: label changed, "
                    f"{row['comparison']!r} -> {now_label!r}"
                )
                if _sign(now["mean_diff"]) != row["sign"]:
                    flipped.append(
                        f"{rel}/{key}[{i}] {row['comparison']}: "
                        f"{row['mean_diff']} -> {now['mean_diff']}"
                    )

    assert checked == 24, f"expected 24 frozen directions, compared {checked}"
    assert not flipped, "a published comparison changed direction:\n  " + "\n  ".join(flipped)


def test_the_holm_fixture_provably_cannot_catch_an_arm_swap():
    """
    Pins the symmetry that makes this file necessary, so nobody deletes it as redundant.

    If a future estimator broke this symmetry, the Holm fixture WOULD start catching arm
    swaps and this reasoning would need revisiting — so the property is asserted rather
    than left as a comment.
    """
    from rb.stats import paired_bootstrap

    a = [0.9, 0.2, 0.7, 0.4, 0.55] * 40
    b = [0.3, 0.5, 0.1, 0.6, 0.25] * 40

    forward, reverse = paired_bootstrap(a, b), paired_bootstrap(b, a)

    assert forward["p_value"] == reverse["p_value"], "symmetry broken; revisit this file"
    assert _sign(forward["mean_diff"]) == -_sign(reverse["mean_diff"])
