VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup test reproduce clean

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

test:
	PYTHONPATH=src $(VENV)/bin/pytest -q tests

# Full run: controls, grep baseline and BM25 anchor on all three datasets.
reproduce:
	PYTHONPATH=src $(PY) -m rb.run --dataset scifact
	PYTHONPATH=src $(PY) -m rb.bm25_control scifact
	PYTHONPATH=src $(PY) -m rb.run --dataset quora
	PYTHONPATH=src $(PY) -m rb.bm25_control quora
	PYTHONPATH=src $(PY) -m rb.run --dataset hotpotqa
	PYTHONPATH=src $(PY) -m rb.bm25_control hotpotqa

clean:
	rm -rf data/*/rg_corpus.txt
