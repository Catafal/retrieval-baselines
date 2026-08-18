"""
The stopword list, frozen here rather than imported.

A list that arrives from a library changes when the library changes, which would
silently change published numbers. 33 words, committed, versioned with the results.
"""

STOPWORDS = frozenset("""
a an and are as at be by for from has have how i in is it its of on or that the
their there these this to was what when where which who why will with
""".split())
