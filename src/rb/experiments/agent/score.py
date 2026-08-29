"""
Answer scoring for 006. SQuAD-style normalisation, exact match and token F1.

No judge model anywhere in this path. The whole reason 006 can measure answering at all is
that HotpotQA ships a gold answer string, so correctness is a string comparison and not
another model's opinion. A judge here would reintroduce exactly the unmeasurable that kept
generation out of the sequence for five experiments.

THE PRIMARY METRIC IS STRICT EQUALITY, AND THAT IS A CORRECTION.

The first version of this file made containment the primary rule: a prediction scored correct
if it contained the gold span within six words. Review broke it in seven ways, of which the
worst were `Iron Man 3` scoring correct against gold `Iron Man`, and
`the answer is not France, it is Germany` scoring correct against gold `France`.

Every one of those failures requires the prediction to be LONGER than the gold span, so the
rule systematically favoured whichever arms narrate most -- which by construction are the two
arms in the primary contrast. A scoring rule that inflates both sides of the comparison being
tested, by an unknown net amount, cannot be the headline.

So `exact_match` is now normalised equality, which is what HotpotQA's own evaluation does, and
the lenient rule survives as `exact_match_lenient`, reported beside it as a sensitivity check.
Both ship, because they disagree precisely where the arms differ in narration.
"""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)

# Agents narrate even when told not to. Stripped before scoring so a correct answer wrapped in
# a stock opener is not counted wrong. Applied identically to every arm.
_PREFIXES = re.compile(
    r"^(the answer is|answer|based on the (context|passages|facts)[,:]?|"
    r"according to the (context|passages|facts)[,:]?)\s*[:\-]?\s*", re.I)

# A prediction that argues against a span must not be credited for containing it.
_NEGATION = re.compile(r"\b(not|isn't|is not|wasn't|was not|rather than|instead of|"
                       r"don't know|do not know|cannot determine|no information)\b", re.I)


def normalise(s: str) -> str:
    s = _PREFIXES.sub("", (s or "").strip())
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def normalise_strict(s: str) -> str:
    """Normalisation WITHOUT the narration strip.

    `em_strict` must be a genuinely independent check, and it is not one if it shares the
    leniency layer with the primary metric -- which was the case until review pointed it out.
    """
    s = (s or "").strip().lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    """PRIMARY. Normalised equality, after stripping stock narration openers."""
    g = normalise(gold)
    return int(bool(g) and normalise(pred) == g)


def exact_match_verbatim(pred: str, gold: str) -> int:
    """The strictest variant: no narration strip at all."""
    g = normalise_strict(gold)
    return int(bool(g) and normalise_strict(pred) == g)


def _contains_tokens(hay: list[str], needle: list[str]) -> bool:
    """Contiguous TOKEN subsequence, not substring.

    Substring containment credited `no` inside `North Dakota` and `196` inside
    `between 1962 and 1964`. Tokens make those impossible.
    """
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def exact_match_lenient(pred: str, gold: str) -> int:
    """SENSITIVITY. Credits a prediction that states the gold span inside a short answer.

    Three guards, each closing a demonstrated false positive: token subsequence rather than
    substring; a tight length bound; and a refusal to credit a prediction that negates or
    hedges. It still cannot tell `Iron Man 3` from `Iron Man`, which is why it is not primary.
    """
    p, g = normalise(pred), normalise(gold)
    if not g:
        return 0
    if p == g:
        return 1
    if _NEGATION.search(p):
        return 0
    pt, gt = p.split(), g.split()
    return int(len(pt) <= len(gt) + 3 and _contains_tokens(pt, gt))


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
