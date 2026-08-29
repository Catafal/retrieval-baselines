"""
Experiment 006's arms. One prompt builder per condition, and one shared instruction.

THE PROMPTS ARE IDENTICAL EXCEPT FOR THE CONTEXT BLOCK AND TOOL ACCESS. This is the design's
main defence against the strawman failure: the easiest way to manufacture a graph win is to
write an encouraging prompt for the arm you like and a flat one for the arm you do not.
ANSWER_RULE below is one string, shared verbatim by every arm including grep, and the arms
differ only in what precedes it. A diff of two arms' prompts should show the context block
and nothing else.

Token budgets are equalised at the point of injection, not left to the corpus. The prior art
claims a fixed ~400-token graph injection against a 660-1180 token search transcript, so a
comparison that let the passage arms inject 3,000 tokens would be measuring context volume
and reporting it as structure. `fit_budget` truncates every injected arm to the same budget,
and the realised budget per arm is recorded so the equalisation can be audited rather than
believed.
"""

from dataclasses import dataclass

# Roughly 4 characters per token for English prose. Deliberately crude: it is applied
# identically to every arm, so its error cancels in the between-arm comparison, and the
# MEASURED input tokens from the CLI are what the protocol reports.
CHARS_PER_TOKEN = 4

# The prior art's graph injection is ~400 tokens. Every injected arm gets the same.
DEFAULT_BUDGET_TOKENS = 400

ANSWER_RULE = (
    "Answer with the shortest possible answer span: a name, a date, or a short noun phrase. "
    "Output ONLY that span, with no preamble, no explanation and no full sentence. "
    "If you cannot determine the answer, output exactly UNKNOWN."
)

SYSTEM = (
    "You answer multi-hop factual questions. " + ANSWER_RULE
)

# The grep arm needs to be told the corpus exists and that chaining searches is allowed --
# withholding that would hobble it. It is told nothing about strategy that the injected arms
# are not also told, and it gets the same ANSWER_RULE.
SYSTEM_GREP = (
    "You answer multi-hop factual questions by searching a document corpus. "
    "The corpus is a directory of markdown files, one document per file, each beginning with "
    "its title as a heading. Answering usually requires finding one document, extracting a "
    "name or term from it, and then searching for a second document about that term. "
    "Search as many times as you need. " + ANSWER_RULE
)

# Read-only: everything a competent human searcher would use, and nothing that reaches the
# network or writes. Granting less would be the hobbled-toolkit strawman.
GREP_TOOLS = "Grep,Glob,Read"


@dataclass
class Injection:
    text: str
    tokens_est: int
    items: int
    truncated: bool


def fit_budget(blocks: list[str], budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> Injection:
    """Take whole blocks in order until the budget is spent. Never splits a block.

    Whole blocks only, because half a passage is a different object from a passage and would
    make the arms incomparable in kind as well as in size. An arm whose first block already
    exceeds the budget keeps that one block, so every arm always gets at least its top hit.
    """
    limit = budget_tokens * CHARS_PER_TOKEN
    kept, used = [], 0
    for b in blocks:
        if kept and used + len(b) > limit:
            break
        kept.append(b)
        used += len(b)
    return Injection("\n\n".join(kept), used // CHARS_PER_TOKEN, len(kept),
                     truncated=len(kept) < len(blocks))


def _wrap(question: str, context: str | None, header: str | None) -> str:
    if context is None:
        return f"Question: {question}"
    return f"{header}\n\n{context}\n\n---\n\nQuestion: {question}"


def closed_book(question: str) -> tuple[str, Injection]:
    """No corpus, no tools. Measures what the model already knows, per question per tier."""
    return _wrap(question, None, None), Injection("", 0, 0, False)


def passages(question: str, docs: list[tuple[str, str]],
             budget: int = DEFAULT_BUDGET_TOKENS) -> tuple[str, Injection]:
    """Shared builder for every passage-injection arm: bm25, dense, oracle.

    One builder rather than three, so the three arms cannot drift apart in formatting and
    have a formatting difference read as a retrieval difference.
    """
    inj = fit_budget([f"[{t}]\n{txt}" for t, txt in docs], budget)
    return _wrap(question, inj.text, "Context passages:"), inj


def graph_facts(question: str, facts_text: str,
                budget: int = DEFAULT_BUDGET_TOKENS) -> tuple[str, Injection]:
    """The prior art's shape: typed triples plus entity notes, no passages."""
    inj = fit_budget([facts_text], budget)
    return _wrap(question, inj.text, "Recalled facts from memory:"), inj


def grep(question: str) -> tuple[str, Injection]:
    """The agent finds its own evidence. Corpus reaches it via --add-dir, not the prompt."""
    return _wrap(question, None, None), Injection("", 0, 0, False)
