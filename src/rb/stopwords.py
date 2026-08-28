"""
The stopword list, frozen here rather than imported.

A list that arrives from a library changes when the library changes, which would
silently change published numbers. 37 words, committed, versioned with the results.

The docstring said 33 from the first commit and the list has always had 37. Entry 001 copied
the wrong number into a published sentence, where it survived a retraction, a rebuild and four
review passes before a cross-entry audit counted the object instead of reading the comment. A
frozen list exists so a number cannot drift; a hand-written count of it is the one part that
can, and it did. tests/test_stopwords_count.py now pins it.
"""

STOPWORDS = frozenset("""
a an and are as at be by for from has have how i in is it its of on or that the
their there these this to was what when where which who why will with
""".split())
