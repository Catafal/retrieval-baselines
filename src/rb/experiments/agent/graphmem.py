"""
Experiment 006's graph-memory arm — a faithful replication of graph-memory-starter.

001-005's graph is entity CO-OCCURRENCE: spaCy or GLM emits (surface_form, label) pairs and
two entities are adjacent when they appear in the same passage. It has no predicates, and
llm_extractor.py's own docstring records that OpenIE triples were deliberately excluded
because they would change the node AND edge semantics at once.

The starter's mechanism is a different artifact: typed relations walked by a recursive CTE,
returning FACTS rather than documents. Reproducing its claim therefore requires building
that artifact, not reusing 003's. This module is the smallest faithful version:

  extract()  one LLM call per document -> nodes, edges, aliases  (src/extract-prompt.md)
  build()    three tables: entities, relations, aliases          (src/schema.sql)
  recall()   seed by name/alias occurring in the question, walk k hops, return top_k triples
             plus the descriptions of the entities involved      (src/recall.py)

THE EXTRACTOR IS PINNED TO THE WEAKEST TIER UNDER TEST, AND THAT IS THE POINT.

The graph is built by an LLM reading every document. That is inference-time compute moved to
build time, and if the builder were a stronger model than the answerer, the graph arm would
be smuggling a smarter model's reasoning into a weaker model's answer and reporting the
result as "structure helps". Pinning extraction to haiku -- the weakest of the three tiers
being scored -- makes that impossible: no answering arm can receive reasoning from a model
better than the worst one in the experiment. It costs the graph arm its best case and is
registered for exactly that reason.

Build cost is measured and reported alongside the answering results. A method that wins at
query time by spending more at build time has not won until both are on the page.
"""

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from rb.experiments.agent import runner

# The starter's vocabulary is its corpus's (PERSON, ROLE, POLICY, PROCESS, DOCUMENT), which
# describes a company's approval chain. HotpotQA is encyclopaedic, so the vocabulary is
# re-drawn for the domain while the SHAPE -- a closed type set and a closed predicate set --
# is kept, because a closed vocabulary is what makes the edges walkable rather than free text.
ENTITY_TYPES = ["PERSON", "ORG", "PLACE", "WORK", "EVENT", "GROUP", "OTHER"]
PREDICATES = [
    "member_of", "located_in", "created_by", "part_of", "born_in", "died_in",
    "spouse_of", "parent_of", "employed_by", "succeeded_by", "occurred_in",
    "has_role", "associated_with",
]

# One f-string would collapse the JSON braces before .format() saw them, so the template is a
# function: the vocabulary is interpolated once, here, and the document is substituted by
# concatenation rather than by a second round of brace parsing.
_JSON_SHAPE = (
    '{"nodes": [{"name": "", "type": "", "description": ""}],\n'
    ' "edges": [{"source": "", "predicate": "", "target": ""}],\n'
    ' "aliases": [{"entity": "", "alias": ""}]}'
)


def prompt(title: str, text: str) -> str:
    return (
        "Read the document below. Extract a knowledge graph using ONLY this vocabulary.\n\n"
        f"Entity types: {', '.join(ENTITY_TYPES)}\n"
        f"Relationships: {', '.join(PREDICATES)}\n\n"
        "Return ONLY JSON, no prose, no code fence:\n"
        f"{_JSON_SHAPE}\n\n"
        "Rules:\n"
        "- Use the most complete form of each name. Add short forms as aliases.\n"
        "- Put conditions (dates, numbers, qualifiers) in the entity description.\n"
        "- Extract only facts stated in the document.\n"
        "- Every edge endpoint must appear in nodes.\n\n"
        f"Document title: {title}\n\n{text}\n"
    )


SCHEMA = """
CREATE TABLE IF NOT EXISTS entities  (id TEXT PRIMARY KEY, name TEXT, type TEXT,
                                      description TEXT, source_doc TEXT);
CREATE TABLE IF NOT EXISTS relations (source_id TEXT, target_id TEXT,
                                      predicate TEXT, source_doc TEXT);
CREATE TABLE IF NOT EXISTS aliases   (entity_id TEXT, alias TEXT);
CREATE INDEX IF NOT EXISTS r_src ON relations(source_id);
CREATE INDEX IF NOT EXISTS r_tgt ON relations(target_id);
"""

# Identical to the starter's src/recall.py, including the ORDER BY near that makes the
# top_k cut a nearest-hops-first cut rather than an arbitrary one.
WALK = """
WITH RECURSIVE walk(entity_id, depth) AS (
  SELECT id, 0 FROM entities WHERE id IN ({seeds})
  UNION
  SELECT CASE WHEN r.source_id = w.entity_id THEN r.target_id ELSE r.source_id END,
         w.depth + 1
  FROM relations r JOIN walk w ON w.entity_id IN (r.source_id, r.target_id)
  WHERE w.depth < ?
)
SELECT e1.name, r.predicate, e2.name, r.source_doc,
       MIN((SELECT MIN(depth) FROM walk WHERE entity_id = r.source_id),
           (SELECT MIN(depth) FROM walk WHERE entity_id = r.target_id)) AS near
FROM relations r
JOIN entities e1 ON e1.id = r.source_id
JOIN entities e2 ON e2.id = r.target_id
WHERE r.source_id IN (SELECT entity_id FROM walk)
  AND r.target_id IN (SELECT entity_id FROM walk)
ORDER BY near
"""

EXTRACT_SYSTEM = ("You extract structured knowledge graphs from text. "
                  "You output only JSON. You never explain.")


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _eid(etype: str, name: str) -> str:
    """The starter keys entities by uuid5(type + normalised name); the same collapsing rule
    expressed as a plain string key, so two documents naming the same thing share a node."""
    return f"{etype.upper()}::{_norm(name)}"


def _parse(raw: str) -> dict | None:
    """Recover the JSON object from a model reply that may be fenced or narrated."""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def extract(pool: dict[str, str], out: Path, model: str = "haiku",
            workers: int = 6) -> list[runner.Call]:
    """One LLM call per pool document. Resumable: run_all skips what is already recorded."""
    jobs = [{"query_id": title, "arm": "extract", "model": model,
             "prompt": prompt(title, text),
             "system": EXTRACT_SYSTEM, "allowed_tools": ""}
            for title, text in sorted(pool.items())]
    return runner.run_all(jobs, out, workers=workers)


def build(extraction_jsonl: Path, db_path: Path) -> dict:
    """Load the extractions into the three tables. Returns build statistics."""
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)

    n_docs = n_nodes = n_edges = n_alias = n_bad = 0
    dropped_edges = 0
    for line in extraction_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            n_bad += 1
            continue
        doc = rec["query_id"]
        d = _parse(rec.get("answer", ""))
        if not d:
            n_bad += 1
            continue
        n_docs += 1
        local: dict[str, str] = {}          # normalised name -> entity id, for edge lookup
        for nd in d.get("nodes", []) or []:
            name, etype = (nd.get("name") or "").strip(), (nd.get("type") or "OTHER").upper()
            if not name:
                continue
            eid = _eid(etype, name)
            local[_norm(name)] = eid
            db.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?)",
                       (eid, name, etype, (nd.get("description") or "").strip(), doc))
            n_nodes += 1
        for ed in d.get("edges", []) or []:
            s, t = local.get(_norm(ed.get("source", ""))), local.get(_norm(ed.get("target", "")))
            pred = (ed.get("predicate") or "").strip()
            if not (s and t and pred) or s == t:
                # The prompt requires every endpoint to appear in nodes. Edges that break
                # that rule are dropped and counted rather than silently repaired.
                dropped_edges += 1
                continue
            db.execute("INSERT INTO relations VALUES (?,?,?,?)", (s, t, pred, doc))
            n_edges += 1
        for al in d.get("aliases", []) or []:
            eid = local.get(_norm(al.get("entity", "")))
            alias = (al.get("alias") or "").strip()
            if eid and alias:
                db.execute("INSERT INTO aliases VALUES (?,?)", (eid, alias))
                n_alias += 1
    db.commit()
    distinct = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    db.close()
    return {"docs_parsed": n_docs, "docs_unparseable": n_bad, "node_mentions": n_nodes,
            "distinct_entities": distinct, "edges": n_edges,
            "edges_dropped_dangling": dropped_edges, "aliases": n_alias}


@dataclass
class Facts:
    triples: list          # (source, predicate, target, source_doc, depth)
    notes: list
    ms: float
    seeds: int

    def depth_histogram(self) -> dict:
        h = {}
        for *_, d in self.triples:
            h[int(d)] = h.get(int(d), 0) + 1
        return h

    def lines(self) -> list[str]:
        """The injection as separate blocks, so the shared token budget can actually cut it.

        Returned as lines rather than one string because arms.fit_budget never splits a block:
        handing it a single blob meant the graph arm was the one injected arm the 400-token
        budget did not bind on, which review measured at up to 569 tokens.
        """
        if not self.triples:
            return ["memory: no facts recalled for this question"]
        width = max(len(f"{s} --[{p}]--> {t}") for s, p, t, _, _ in self.triples)
        out = [f"memory: {len(self.triples)} facts recalled"]
        out += [f"{f'{s} --[{p}]--> {t}':<{width}}   ({doc})"
                for s, p, t, doc, _ in self.triples]
        if self.notes:
            out.append("where:")
            out += [f"  {n}: {d}" for n, d in self.notes]
        return out

    def as_text(self) -> str:
        return "\n".join(self.lines())


# Slots reserved for triples the walk had to traverse to reach. REGISTERED AT 0, which is the
# prior art's own configuration: `ORDER BY near` with a flat top_k cut and no reservation.
#
# Review found that a flat cut makes `hops` nearly decorative -- answer presence was identical
# at hops=1, 2 and 3 -- and a reservation was written to force deeper facts through. Both were
# then measured on the partial graph BEFORE this constant was fixed, and that measurement is
# disclosed rather than buried: reserve=0 gives 28% of injected triples at depth>=1 and answer
# presence 0.26; reserve=4 gives 55% at depth>=1 and answer presence 0.18.
#
# The reserve is the better retrieval config and it is NOT the registered one. Fidelity decides
# this, not the score: 006 exists to test the prior art's mechanism, and improving on that
# mechanism would be a different experiment. The reserve ships as an exploratory sensitivity,
# and the realised depth histogram is a mandatory reported diagnostic -- because if the walk
# delivers almost nothing beyond depth 1, then "the graph walks the hops" is a claim about the
# design that the design does not keep, and that is a finding rather than a bug.
DEPTH_RESERVE = 0


def recall(db_path: Path, question: str, hops: int = 3, top_k: int = 8,
           depth_reserve: int = DEPTH_RESERVE) -> Facts:
    """Seed on any entity name or alias occurring in the question, then walk. No model call."""
    t0 = time.perf_counter()
    db = sqlite3.connect(db_path)
    q = question.lower()
    seeds = {}
    rows = list(db.execute("SELECT id, name FROM entities"))
    rows += list(db.execute("SELECT entity_id, alias FROM aliases"))
    for eid, text in rows:
        t = (text or "").lower().strip()
        if len(t) < 3:
            continue                      # a 1-2 char alias matches everything
        if re.search(rf"\b{re.escape(t)}\b", q):
            seeds[eid] = True
    if not seeds:
        db.close()
        return Facts([], [], (time.perf_counter() - t0) * 1000, 0)

    marks = ",".join("?" * len(seeds))
    rows = db.execute(WALK.format(seeds=marks), (*seeds, hops)).fetchall()

    # Depth-stratified cut: fill the reserved slots from depth >= 1 first, then take the
    # nearest facts for whatever remains. Ordering within each stratum is unchanged.
    near = [(s, p, t, doc, d) for s, p, t, doc, d in rows if (d or 0) >= 1]
    shallow = [(s, p, t, doc, d) for s, p, t, doc, d in rows if (d or 0) < 1]
    reserved = near[:min(depth_reserve, top_k)]
    triples = reserved + shallow[:top_k - len(reserved)]
    if len(triples) < top_k:                     # corpus had too few shallow facts
        triples += near[len(reserved):len(reserved) + (top_k - len(triples))]

    names = {n for s, _, t, _, _ in triples for n in (s, t)}
    notes = []
    if names:
        m = ",".join("?" * len(names))
        notes = db.execute(
            f"SELECT DISTINCT name, description FROM entities "  # noqa: S608
            f"WHERE name IN ({m}) AND description != '' ORDER BY name", (*names,)).fetchall()
    db.close()
    return Facts(triples, notes, (time.perf_counter() - t0) * 1000, len(seeds))
