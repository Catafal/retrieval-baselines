"""
Experiment 006's driver: turn a frozen question sample into the full job list, run it, score it.

Two things here are deliberate and are registered in the protocol.

INTERLEAVING. Jobs are shuffled with a fixed seed before submission rather than emitted
arm-by-arm. A run of thousands of calls takes hours, and service conditions drift over hours.
Emitting all of one arm and then all of another would let that drift land entirely on one arm
and be read as an arm effect. The shuffle is seeded so the order is reproducible.

ONE BUILDER PER ARM, ONE PROMPT SHAPE. Every arm's prompt comes from arms.py and shares
ANSWER_RULE verbatim; the arms differ in their context block and their tool grant, nothing
else. The grep arm is the only one that receives --add-dir and a turn cap.
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

from rb.experiments.agent import arms, graphmem, runner, score
from rb.experiments.agent.corpus import Question

TIERS = ("haiku", "sonnet", "opus")

# Registered in the protocol before any scored call. Generous: a 2-hop chain needs roughly
# four tool calls (search, read, search, read); 20 turns is five times that, so exhaustion
# means the agent was lost, not that the cap was tight.
GREP_MAX_TURNS = 20


def build_jobs(questions: list[Question], pool: dict[str, str], corpus_dir: Path,
               retrieved: dict[str, dict[str, list[str]]], facts: dict[str, str],
               tiers=TIERS, budget: int = arms.DEFAULT_BUDGET_TOKENS) -> list[dict]:
    """One job per (question, arm, tier).

    `retrieved` maps arm name -> query id -> ranked titles, precomputed so that no retrieval
    happens inside the scored loop. `facts` maps query id -> the graph's recalled text.
    """
    jobs = []
    for q in questions:
        variants: list[tuple[str, str, arms.Injection, str, str | None, int | None]] = []

        p, inj = arms.closed_book(q.question)
        variants.append(("closed-book", p, inj, arms.SYSTEM, None, None))

        p, inj = arms.grep(q.question)
        variants.append(("grep", p, inj, arms.system_grep(str(corpus_dir)),
                         str(corpus_dir), GREP_MAX_TURNS))

        for name in ("bm25", "dense"):
            docs = [(t, pool[t]) for t in retrieved[name].get(q.id, [])]
            p, inj = arms.passages(q.question, docs, budget)
            variants.append((name, p, inj, arms.SYSTEM, None, None))

        # The ceiling: the gold passages themselves. If an arm approaches this, retrieval is
        # no longer the bottleneck; if the grep arm cannot approach it, the grep arm is broken
        # rather than the corpus being hard. It is the guard against a strawman control.
        docs = [(t, pool[t]) for t in q.gold if t in pool]
        p, inj = arms.passages(q.question, docs, budget)
        variants.append(("oracle", p, inj, arms.SYSTEM, None, None))

        p, inj = arms.graph_facts(q.question, facts.get(q.id, ""), budget)
        variants.append(("graph-facts", p, inj, arms.SYSTEM, None, None))

        for tier in tiers:
            for name, prompt, inj, system, add_dir, mt in variants:
                jobs.append({"query_id": q.id, "arm": name, "model": tier, "prompt": prompt,
                             "system": system, "allowed_tools":
                                 arms.GREP_TOOLS if name == "grep" else "",
                             "add_dir": add_dir, "max_turns": mt,
                             "injected_tokens_est": inj.tokens_est})
    return jobs


def shuffle_jobs(jobs: list[dict], seed: int) -> list[dict]:
    j = list(jobs)
    random.Random(seed).shuffle(j)
    return j


def retrieve_all(questions: list[Question], pool: dict[str, str], top_n: int = 8
                 ) -> dict[str, dict[str, list[str]]]:
    """BM25 and dense rankings, computed once, outside the scored loop.

    top_n is deliberately larger than the budget will fit: fit_budget does the cutting, so
    the number of passages an arm receives is set by the shared token budget rather than by
    a per-arm k. That is what makes the arms budget-comparable.
    """
    from rb.experiments.ladder.retrievers.dense import DenseRetriever, SentenceTransformerEncoder
    from rb.experiments.ladder.retrievers.lexical import full_bm25
    from rb.experiments.ladder.run import BGE_MODEL_NAME, BGE_REVISION

    queries = {q.id: q.question for q in questions}
    out = {}
    for name, r in (("bm25", full_bm25()),
                    ("dense", DenseRetriever(SentenceTransformerEncoder(BGE_MODEL_NAME,
                                                                        BGE_REVISION)))):
        run = r.retrieve(pool, queries, top_k=top_n)
        out[name] = {qid: list(d)[:top_n] for qid, d in run.items()}
    return out


def recall_all(questions: list[Question], db: Path, hops: int = 3, top_k: int = 8) -> dict:
    return {q.id: graphmem.recall(db, q.question, hops, top_k).as_text() for q in questions}


def score_calls(calls, gold: dict[str, str], baseline: float = 0.0) -> list[dict]:
    """Attach both EM variants and the abstention flag to every call.

    Both EM variants ship because the lenient one -- which credits a prediction that contains
    the gold span -- can only ever help an arm that narrates, and the arms differ in how much
    they narrate by construction. Reporting one without the other would let a scoring choice
    decide the result.
    """
    rows = []
    for c in calls:
        d = asdict(c) if not isinstance(c, dict) else dict(c)
        g = gold[d["query_id"]]
        d["em"] = score.exact_match(d["answer"], g)
        d["em_lenient"] = score.exact_match_lenient(d["answer"], g)
        d["em_strict"] = score.exact_match_verbatim(d["answer"], g)
        d["f1"] = round(score.token_f1(d["answer"], g), 4)
        # Context the ARM supplied, with the harness's fixed overhead removed by the analysis.
        d["context_tokens"] = (d["input_tokens"] + d["cache_read_tokens"]
                               + d["cache_creation_tokens"])
        d["abstained"] = int(score.is_abstention(d["answer"]))
        d["gold"] = g
        d["context_tokens_net"] = max(0.0, d["context_tokens"] - baseline)
        rows.append(d)
    return rows


def load_scored(path: Path, gold: dict[str, str]) -> list[dict]:
    """Score twice: once to find the harness's zero-context baseline, once to net it out.

    The baseline is what a closed-book call costs before any corpus reaches the model -- the
    CLI's own system prompt and tool schemas. Reporting it as context the model read would
    credit every arm with ~23k tokens of harness and drown the difference P4 is about.
    """
    calls = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    first = score_calls(calls, gold)
    cb = [r for r in first if r["arm"] == "closed-book"
          and r.get("outcome") not in {"timeout", "api_error", "no_json"}]
    baseline = sum(r["context_tokens"] for r in cb) / len(cb) if cb else 0.0
    return score_calls(calls, gold, baseline)
