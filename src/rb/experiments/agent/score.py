"""
Answer scoring for 006. SQuAD-style normalisation, exact match and token F1.

No judge model anywhere in this path. The whole reason 006 can measure answering at all is
that HotpotQA ships a gold answer string, so correctness is a string comparison and not
another model's opinion. Introducing a judge here would reintroduce exactly the
unmeasurable that kept generation out of the sequence for five experiments.
"""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)
# Agents narrate even when told not to. These are stripped before scoring so that a correct
# answer wrapped in a sentence is not counted as wrong; the containment fallback below does
# the rest. Any leniency here is applied identically to every arm.
_PREFIXES = re.compile(
    r"^(the answer is|answer|based on the (context|passages|facts)[,:]?|"
    r"according to the (context|passages|facts)[,:]?)\s*[:\-]?\s*", re.I)


def normalise(s: str) -> str:
    s = _PREFIXES.sub("", (s or "").strip())
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    """1 when the normalised prediction equals the gold span, or cleanly contains it.

    Containment is included because an agent asked for a span often returns a short
    sentence. It is deliberately one-directional and length-bounded: a prediction may
    contain the gold, never the reverse, and only when it is not padded out with enough
    other text to be a guess-everything answer.
    """
    p, g = normalise(pred), normalise(gold)
    if not g:
        return 0
    if p == g:
        return 1
    if g in p and len(p.split()) <= len(g.split()) + 6:
        return 1
    return 0


def token_f1(pred: str, gold: str) -> float:
    p, g = normalise(pred).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def is_abstention(pred: str) -> bool:
    """UNKNOWN is a distinct outcome from a wrong answer and is reported separately."""
    return normalise(pred) in {"unknown", "i dont know", "i do not know", ""}
