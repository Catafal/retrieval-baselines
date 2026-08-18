"""
BEIR dataset acquisition and loading.

Corpora are downloaded, never redistributed (see LICENCES.md). What this repo
commits instead is the SHA-256 of every zip it consumed, in manifests/datasets.json.
That is the difference between "I ran it on SciFact" and "I ran it on the bytes
whose hash is this" — only the second is checkable by a stranger.

BEIR layout inside each zip:
    <name>/corpus.jsonl     {"_id", "title", "text"}
    <name>/queries.jsonl    {"_id", "text"}
    <name>/qrels/test.tsv   query-id \t corpus-id \t score   (with header row)
"""

import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# The three datasets fixed by protocol.md. Nothing else may be added without a
# protocol amendment, because dataset choice after seeing results is the classic
# way to manufacture a finding.
DATASETS = ("scifact", "quora", "hotpotqa")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MANIFEST = Path(__file__).resolve().parents[2] / "manifests" / "datasets.json"


def _sha256(path: Path) -> str:
    """Stream the file so a 5 GB corpus does not have to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str) -> Path:
    """Fetch <name>.zip into data/, skipping if already present. Returns the zip path."""
    if name not in DATASETS:
        raise ValueError(f"{name} is not in the pre-registered set {DATASETS}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{name}.zip"
    if zip_path.exists():
        return zip_path

    url = BEIR_URL.format(name=name)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return zip_path


def extract(name: str) -> Path:
    """Unzip into data/<name>/, recording the zip's hash in the manifest."""
    zip_path = download(name)
    out_dir = DATA_DIR / name
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
    _record_manifest(name, zip_path)
    return out_dir


def _record_manifest(name: str, zip_path: Path) -> None:
    """
    Append or verify this dataset's hash.

    A hash that changes between runs means the upstream file changed under us,
    which invalidates every number already published against it. Loud failure is
    the only correct behaviour here.
    """
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    digest = _sha256(zip_path)
    if name in manifest and manifest[name]["sha256"] != digest:
        raise RuntimeError(
            f"{name}.zip hash changed: manifest has {manifest[name]['sha256']}, "
            f"file is {digest}. Published results reference the manifest hash."
        )
    manifest[name] = {"sha256": digest, "bytes": zip_path.stat().st_size}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_corpus(name: str) -> dict[str, str]:
    """doc_id -> "title text", the single string every retriever in this repo sees."""
    path = extract(name) / "corpus.jsonl"
    corpus = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            title = (d.get("title") or "").strip()
            text = (d.get("text") or "").strip()
            corpus[d["_id"]] = f"{title} {text}".strip()
    return corpus


def load_queries(name: str) -> dict[str, str]:
    path = extract(name) / "queries.jsonl"
    with open(path, encoding="utf8") as f:
        return {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}


def load_qrels(name: str, split: str = "test") -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: relevance}. Only positive judgements are kept."""
    path = extract(name) / "qrels" / f"{split}.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            score = int(score)
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    return qrels


def load(name: str, split: str = "test"):
    """
    Corpus, queries and qrels, with queries restricted to those that have judgements.

    BEIR ships more queries than qrels for several datasets. Scoring a query with
    no gold document would silently drive recall toward zero and look like a
    finding about the retriever.
    """
    qrels = load_qrels(name, split)
    queries = {qid: q for qid, q in load_queries(name).items() if qid in qrels}
    return load_corpus(name), queries, qrels
