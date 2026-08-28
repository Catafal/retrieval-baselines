VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup test reproduce reproduce-002-lexical reproduce-003-controls reproduce-003-pool-control reproduce-003-analysis reproduce-003-2wiki-analysis reproduce-003-arms-summary reproduce-003-oracle reproduce-003-closure reproduce-003-diagnostics reproduce-003-ablation reproduce-004-ablation reproduce-004-pilot-gate clean

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

test:
	PYTHONPATH=src $(VENV)/bin/pytest -q tests

# Full run: controls, grep baseline and BM25 anchor on all three datasets.
#
# FROZEN. This target's meaning is a published promise (see README.md /
# protocols/001-grep-baseline.md): `make reproduce` reproduces 001, exactly this,
# and nothing more. New experiments get their own targets below rather than
# growing this one, so checking 001 never silently becomes an hours-long job.
reproduce:
	PYTHONPATH=src $(PY) -m rb.run --dataset scifact
	PYTHONPATH=src $(PY) -m rb.bm25_control scifact
	PYTHONPATH=src $(PY) -m rb.run --dataset quora
	PYTHONPATH=src $(PY) -m rb.bm25_control quora
	PYTHONPATH=src $(PY) -m rb.run --dataset hotpotqa
	PYTHONPATH=src $(PY) -m rb.bm25_control hotpotqa

# Experiment 002, lexical rungs only (the cheap ones — minutes, not hours). Dense
# and hybrid need a pinned encoder and are run manually per protocols/002-ladder.md,
# gated on the pre-registration tag, not from a Makefile target anyone could run
# by accident.
reproduce-002-lexical:
	PYTHONPATH=src $(PY) -m rb.experiments.ladder.run --dataset scifact --rung coordination
	PYTHONPATH=src $(PY) -m rb.experiments.ladder.run --dataset scifact --rung lexical-factorial
	PYTHONPATH=src $(PY) -m rb.experiments.ladder.run --dataset quora --rung coordination
	PYTHONPATH=src $(PY) -m rb.experiments.ladder.run --dataset quora --rung lexical-factorial
	PYTHONPATH=src $(PY) -m rb.experiments.ladder.run --dataset hotpotqa --rung coordination
	PYTHONPATH=src $(PY) -m rb.experiments.ladder.run --dataset hotpotqa --rung lexical-factorial

# Experiment 003's two closure controls (§8.2 diagnostic, §8.3 gate) plus the seed-match
# rate and the graph summary. Minutes on a warm entity cache; ~10 minutes cold.
#
# Separate target for the same reason 002 has one: `reproduce` is a published promise about
# 001 and must not silently become a long job. The SCORED arms are deliberately absent — they
# take hours and are gated on the pre-registration tag, so they are run deliberately and by
# hand, not from a target anyone could trigger by accident.
#
# To verify this reproduces the committed artifacts, delete data/003-pool-entities.json first:
# a warm cache replays its own contents and would pass even against a broken extractor.
reproduce-003-controls:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.run_controls

# Experiment 003's section 9 pool control, as amended by 003-amendment-4. Writes
# results/003/pool-control.json. Does NOT touch the five arms' summary.json files — see
# src/rb/experiments/graph/run_pool_control.py for why they are left as published.
reproduce-003-pool-control:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.run_pool_control

# Experiment 003's registered analysis (section 7). Rescores from the committed per_query.jsonl
# rather than re-running retrieval, so it is cheap and needs no model. Writes
# results/003/analysis.json, results/003/headroom-control.json and, per amendment 5,
# results/003/decomposition.json (POST-HOC -- its own file, never merged into analysis.json).
# Experiment 004's reasoning ablation — the measurement its entry leads on. COSTS REAL API CALLS:
# it deliberately bypasses the extraction cache, because a cached read would answer with whichever
# reasoning setting was bought rather than the one under test. Roughly $0.30 and a few minutes.
# Needs OPENROUTER_API_KEY. Everything else in 004 replays from the committed cache for free.
# Experiment 004's section 8 pilot gate. Replays from the committed extraction cache and makes no
# API call, unlike reproduce-004-ablation above. This artifact feeds a generated block in the
# entry and previously had no producer at all.
reproduce-004-pilot-gate:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.pilot_gate

reproduce-004-ablation:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.reasoning_ablation

reproduce-003-analysis:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.analysis

# Experiment 003's SECOND corpus (2WikiMultiHopQA), amendment 6. Adjudicates predictions C, D
# and E from the committed per_query.jsonl; does not re-run retrieval and does not touch the
# HotpotQA artifacts. Writes results/003/2wiki/analysis.json.
reproduce-003-2wiki-analysis:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.analysis2wiki

# The cross-arm table for both corpora. Derived from each arm's committed summary.json.
# Replaces a file that previously had no producer and carried a retracted figure.
reproduce-003-arms-summary:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.arms_summary

# The oracle-extractor ceiling: corpus titles as entities, a perfect extractor AND linker.
# A DIAGNOSTIC, not a registered arm -- it uses gold titles a real system does not have.
# Writes results/003/oracle-entity-graph.json, which previously had no producer at all.
reproduce-003-oracle:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.oracle

# Section 8.1's harness-closure GATE: our BM25 against HippoRAG's published BM25 row on a nested
# 1,000-question subset, tolerance 0.05. Halts on failure. Previously had no producer at all,
# which for a gate means it could not be checked by anyone including its author.
reproduce-003-closure:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.closure

# Section 8.2's two side-analyses: inter-rater agreement over the three-model reference panel,
# and the characterisation of the extractor's false positives and negatives.
reproduce-003-diagnostics:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.diagnostics

# The scoring ablation: summed (registered) against mean document scoring. Rebuilds the graph and
# runs PPR over all 7,405 queries, so it is the slow one -- roughly fifteen minutes.
reproduce-003-ablation:
	PYTHONPATH=src $(PY) -m rb.experiments.graph.ablation

clean:
	rm -rf data/*/rg_corpus.txt
