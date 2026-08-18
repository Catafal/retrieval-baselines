VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup test reproduce reproduce-002-lexical clean

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

clean:
	rm -rf data/*/rg_corpus.txt
