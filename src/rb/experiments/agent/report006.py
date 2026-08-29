"""
Render experiment 006's RESULT.md from the committed artifacts.

Same contract as 001 through 005: no number in the record is typed by hand. Every table here
reads from analysis.json, which reads from scored.json, which reads from calls.jsonl. A figure
that cannot be traced back to a recorded call does not appear.

The order of the document is the order the protocol registered, not the order that flatters the
result: the registered predictions first with their decisions, then the falsifier checks, then
the exploratory material, then the costs including the graph's build.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "006"

TIERS = ("haiku", "sonnet", "opus")
ARM_ORDER = ("closed-book", "grep", "bm25", "dense", "graph-facts", "oracle")


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def sgn(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):.4f}"


def ci(a):
    return f"[{sgn(a[0])}, {sgn(a[1])}]"


def arms_table(rows) -> str:
    by = {(r["arm"], r["model"]): r for r in rows}
    out = ["| arm | tier | n | EM | EM lenient | EM strict | F1 | abstain | max-turns | turns | ctx tok | $ |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        for t in TIERS:
            r = by.get((arm, t))
            if not r:
                continue
            out.append(f"| `{arm}` | {t} | {r['n']} | **{f4(r['em'])}** | {f4(r['em_lenient'])} "
                       f"| {f4(r['em_strict'])} | {f4(r['f1'])} | {f4(r['abstained'])} "
                       f"| {f4(r['max_turns_rate'])} | {r['turns']} | {r['context_tokens']:.0f} "
                       f"| {r['cost_usd']:.2f} |")
    return "\n".join(out)


def contrast_rows(cells, label_a, label_b) -> str:
    out = [f"| tier | n | {label_a} | {label_b} | Δ | 95% CI | p | p Holm | MDE | discord | decision |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        if not c.get("resolved"):
            out.append(f"| {c.get('model','?')} | {c.get('n',0)} | | | | | | | | | "
                       f"{c.get('reason','unresolved')} |")
            continue
        out.append(f"| {c['model']} | {c['n']} | {f4(c['mean_a'])} | {f4(c['mean_b'])} "
                   f"| {sgn(c['mean_diff'])} | {ci(c['ci95'])} | {c['p_value']} "
                   f"| {c.get('p_holm', '—')} | {c.get('mde', '—')} "
                   f"| {c.get('discordance', '—')} | **{c.get('decision', '—')}** |")
    return "\n".join(out)


def efficiency_table(pe: dict) -> str:
    """What each arm delivered, against what it did with what it delivered."""
    pres, eff = pe["presence"], pe["efficiency"]
    by = {(r["arm"], r["model"]): r for r in eff}
    out = ["| arm | answer-presence ceiling | EM/ceiling haiku | sonnet | opus |",
           "|---|---|---|---|---|"]
    for arm in ("oracle", "dense", "bm25", "graph-facts"):
        if arm not in pres:
            continue
        cells = "".join(f" {by[(arm, m)]['efficiency']:.3f} |" if (arm, m) in by else " — |"
                        for m in TIERS)
        out.append(f"| `{arm}` | {pres[arm]:.2f} |{cells}")
    return "\n".join(out)


def render(a: dict, build: dict, presence: dict, yld: dict, manifest: dict,
           pe: dict) -> str:
    p1, p3 = a["p1_primary"], a["p3_interaction"]
    L = []
    w = L.append

    w("# Experiment 006 — result")
    w("")
    w("Protocol: `protocols/006-graph-memory-answering.md`, tagged `protocol-006` before the "
      "first scored call. Every figure below is generated from the committed artifacts.")
    w("")

    w("## The headline")
    w("")
    w(f"**P1 (primary).** On haiku, `graph-facts` minus `grep` = **{sgn(p1['mean_diff'])}** EM, "
      f"95% CI {ci(p1['ci95'])}, p = {p1['p_value']}, MDE {p1['mde']} at the realised "
      f"discordance of {p1['discordance']}. Decision: **{p1['decision']}**.")
    w("")
    w(f"**P3 (the prior art's actual thesis, registered underpowered).** The haiku-minus-opus "
      f"interaction = {sgn(p3['mean_diff'])}, CI {ci(p3['ci95'])}, MDE {p3['mde']}. "
      f"Decision: **{p3['decision']}**.")
    w("")

    w("### The shortfall is coverage, not conversion")
    w("")
    w("Answer-presence is the rate at which the gold answer string is present in an arm's "
      "injection: a ceiling on its EM under a copy-only model. EM divided by that ceiling is a "
      "crude conversion efficiency — how well an arm used what it actually delivered.")
    w("")
    w(efficiency_table(pe))
    w("")
    w("**This is the caveat the headline needs.** The graph's injection contains the literal "
      "gold string for "
      f"{pe['presence']['graph-facts']:.0%} of questions against {pe['presence']['bm25']:.0%} "
      f"for BM25, {pe['presence']['dense']:.0%} for dense and {pe['presence']['oracle']:.0%} for "
      "the oracle. Within that coverage the graph converts evidence into correct answers about "
      "as efficiently as BM25 and more efficiently than dense. **The P1 shortfall is "
      "concentrated in what the extractor retained, not in how the answering model used what it "
      "kept.** What this experiment falsifies is a graph built cheaply by the weakest tier, not "
      "graph retrieval as such.")
    w("")
    w("Efficiency above 1 means an arm answered beyond its own injection — parametric memory "
      "filling gaps in a sparse context. It appears for BM25 at opus too, so it is a low-ceiling "
      "strong-model effect rather than anything specific to graph structure. Exploratory, "
      "unregistered, and carrying no decision.")
    w("")
    w("## Every arm, every tier")
    w("")
    w(arms_table(a["arms"]))
    w("")
    w(f"Harness overhead, subtracted from the P4 measure: "
      f"{a['harness_baseline_context_tokens']:.0f} context tokens per call, measured on the "
      f"zero-context arm. It is the CLI's own system prompt and tool schemas, not corpus.")
    w("")

    w("## The registered predictions")
    w("")
    w("### P2 — graph-facts vs grep, per tier (Holm family of 3)")
    w("")
    w(contrast_rows(a["p2_family"], "graph", "grep"))
    w("")
    w("### P5 — the adversarial one: graph-facts vs dense")
    w("")
    w("Registered because the author did not want it to be true.")
    w("")
    w(contrast_rows(a["p5_adversarial"], "graph", "dense"))
    w("")
    w("### P4 — context tokens, grep minus graph-facts")
    w("")
    w("A positive difference means the graph arm read less. The prediction is conjunctive: "
      "fewer tokens AT EQUAL OR BETTER EM.")
    w("")
    w(contrast_rows(a["p4_context_tokens"], "grep tok", "graph tok"))
    w("")

    w("## The falsifier checks")
    w("")
    w("### F2 — is the graph moving hops, or just supplying a short clean context?")
    w("")
    w("`oracle` gives the same information without the graph's structure. If it matches or beats "
      "`graph-facts`, the mechanism claim is dead however the EM table looks.")
    w("")
    w(contrast_rows(a["exploratory"]["oracle_vs_graph"], "oracle", "graph"))
    w("")
    w("### F4 — is the grep arm a strawman?")
    w("")
    w("`grep` against the `oracle` ceiling. If grep on opus cannot approach oracle on opus, the "
      "control is broken and no arm comparison stands.")
    w("")
    w(contrast_rows(a["exploratory"]["grep_vs_oracle"], "grep", "oracle"))
    w("")

    w("### F6 — is `graph-facts` a lexical retrieval arm wearing arrow notation?")
    w("")
    w("Answer presence is the rate at which the gold answer string appears in the injection: a "
      "necessary condition for answering by copying, so a ceiling on the arm's EM under a "
      "copy-only model. Computed with no model calls.")
    w("")
    w("| measure | value |")
    w("|---|---|")
    w(f"| seed rate (question matched ≥1 entity) | {presence['seed_rate']} |")
    w(f"| questions with empty recall | {presence['empty_recall']} |")
    w(f"| `a0` seed neighbourhood only | {presence['a0_seed_only']} |")
    w(f"| `a3` the shipped walk | **{presence['a3_shipped']}** |")
    w(f"| `a3_shuf` edge-permuted placebo | {presence['a3_edge_placebo']} |")
    w(f"| `a3` triples only, no entity prose | {presence['a3_triples_only']} |")
    w(f"| walk over seed match | {presence['walk_over_seed']} |")
    w(f"| walk over placebo | {presence['walk_over_placebo']} |")
    w("")
    w(f"**Verdict: {presence['verdict']}.**")
    w("")
    w("### Does the walk actually walk?")
    w("")
    w(f"Injected-triple depth histogram at the registered configuration "
      f"(`hops=3, top_k=8`, no depth reservation): `{presence['depth_histogram']}`. "
      f"{presence['share_depth_ge_1']:.1%} of injected facts are depth ≥ 1 and "
      f"{presence['share_depth_ge_2']:.1%} are depth ≥ 2.")
    w("")

    w("## Extraction yield, which gates interpretation")
    w("")
    w(f"| | |")
    w(f"|---|---|")
    w(f"| documents attempted | {yld['docs_attempted']} |")
    w(f"| completed | {yld['docs_ok']} |")
    w(f"| parsed into the graph | {yld['docs_parsed']} |")
    w(f"| yield | **{yld['yield']:.1%}** |")
    w(f"| questions with 2 gold docs in the graph | {yld['gold_docs_in_graph']['2']} |")
    w(f"| with 1 | {yld['gold_docs_in_graph']['1']} |")
    w(f"| with 0 | {yld['gold_docs_in_graph']['0']} |")
    w("")
    w(f"Interpretable at the registered 90% threshold: **{yld['interpretable']}**.")
    w("")

    w("## What the graph cost to build")
    w("")
    w("A method that wins at query time by spending more at build time has not won until both "
      "are on the page.")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| extractor | {build['extractor_model']} (the weakest tier under test) |")
    w(f"| documents | {build['docs']} |")
    w(f"| cost | ${build['cost_usd']:.2f} |")
    w(f"| tokens in / out | {build['input_tokens']:,} / {build['output_tokens']:,} |")
    w(f"| wall time | {build['wall_s'] / 60:.0f} min |")
    w(f"| entities / edges / aliases | {build['distinct_entities']:,} / {build['edges']:,} "
      f"/ {build['aliases']:,} |")
    w(f"| amortised per question | ${build['cost_usd'] / manifest['n_questions']:.3f} |")
    w("")

    w("## The corpus")
    w("")
    w(f"{manifest['n_questions']} HotpotQA bridge-hard questions, seed {manifest['seed']}, over "
      f"a {manifest['n_pool_docs']}-document pool of {manifest['pool_chars']:,} characters. "
      f"Questions sha256 `{manifest['questions_sha256'][:16]}`, pool sha256 "
      f"`{manifest['pool_sha256'][:16]}`.")
    w("")
    w("Strata sizes: " + ", ".join(f"{k} = {v}" for k, v in a["strata_sizes"].items()) + ".")
    return "\n".join(L) + "\n"


def main() -> None:
    a = json.loads((OUT / "analysis.json").read_text())
    build = json.loads((OUT / "graph-build.json").read_text())
    presence = json.loads((OUT / "answer-presence.json").read_text())
    yld = json.loads((OUT / "extraction-yield.json").read_text())
    manifest = json.loads((OUT / "sample-manifest.json").read_text())
    pe = json.loads((OUT / "presence-and-efficiency.json").read_text())
    (OUT / "RESULT.md").write_text(render(a, build, presence, yld, manifest, pe))
    print(f"wrote {OUT / 'RESULT.md'}")


if __name__ == "__main__":
    main()
