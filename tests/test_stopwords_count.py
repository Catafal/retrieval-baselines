"""
The stopword list's size is asserted, because its own docstring got it wrong.

The list is frozen and committed so that published numbers cannot drift when a library changes.
That protects the list. It did not protect the COUNT of the list: the module docstring claimed 33
words from the first commit, the list has always held 37, and entry 001 published the 33 in a
sentence about how carefully the baseline was specified.

A number written by hand next to the object it describes is not checked by anything. This is the
check.
"""
from rb.stopwords import STOPWORDS

# The published figure, in the entry and in the module docstring. Change this only alongside both.
DOCUMENTED_SIZE = 37


def test_the_list_is_the_size_its_documentation_claims():
    assert len(STOPWORDS) == DOCUMENTED_SIZE, (
        f"the stopword list holds {len(STOPWORDS)} words and is documented as {DOCUMENTED_SIZE}; "
        "update src/rb/stopwords.py's docstring and entry 001 together, or the published "
        "number drifts from the object again"
    )


def test_the_list_has_no_duplicates():
    """A duplicate would make the count right and the list wrong, which is the same defect."""
    words = [w for w in STOPWORDS]
    assert len(words) == len(set(words))
