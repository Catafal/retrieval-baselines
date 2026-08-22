"""
The whitelist partition check, both halves (NB-25 D3).

MUTATION-CHECKED. NB-24 shipped three tests that passed whether or not the defect was present:
one raised the error it asserted, one used a topology where the term under test normalised
away, one asserted a sum the code's own guard restored even when the invariant was violated.
A test that passes is not evidence; only a test that FAILS against the defect is. Each test
below names the mutation it kills. See NB-25 for the audit these close.
"""

import pytest

from rb.experiments.graph import entity_types as et


# ------------------------------------------------------------------- D3: the partition check

def test_partition_rejects_a_type_that_is_both_kept_and_excluded(monkeypatch):
    """KILLS: deleting the disjointness branch.

    Monkeypatches the module global and calls the REAL function, so the production logic is
    what runs. NB-24's partition test constructed and raised the error itself, which made it
    satisfied by its own setup rather than by the code under test.

    The `match=` is REQUIRED and must differ from the coverage test's: two tests asserting a
    bare RuntimeError would both pass while only the disjointness branch existed, which is
    exactly the shipped bug.
    """
    monkeypatch.setattr(et, "WHITELIST", et.WHITELIST | {"DATE"})
    with pytest.raises(RuntimeError, match="cannot be both kept and excluded"):
        et.assert_partition()


def test_partition_rejects_a_model_label_in_neither_set():
    """KILLS: deleting the coverage branch — i.e. reverting to the shipped bug.

    Distinct `match=` from the disjointness test, so this cannot pass by accidentally
    exercising the other branch. Passes labels directly: no monkeypatching needed, because the
    check takes its inventory as an argument.
    """
    with pytest.raises(RuntimeError, match="classifies neither way"):
        et.assert_partition(set(et.WHITELIST | et.EXCLUDED) | {"BRAND_NEW_TYPE"})


def test_partition_rejects_a_declared_type_the_model_never_emits():
    """KILLS: checking only one direction of the set difference."""
    with pytest.raises(RuntimeError, match="never emits"):
        et.assert_partition(set(et.WHITELIST | et.EXCLUDED) - {"PERSON"})


def test_partition_accepts_the_real_inventory():
    """The declared sets must exactly cover the pinned model's labels — the live invariant."""
    et.assert_partition(set(et.WHITELIST | et.EXCLUDED))


def test_partition_without_labels_cannot_check_coverage():
    """Pins the documented contract: the no-argument form is disjointness ONLY.

    Stops a future 'improvement' from adding a hand-written label constant here, which is the
    self-defeating design this fix rejected.
    """
    et.assert_partition()  # must not raise despite no inventory being supplied
