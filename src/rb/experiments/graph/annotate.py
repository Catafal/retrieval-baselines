"""
The annotation tool for Experiment 003 §8.2.

WHY THIS EXISTS RATHER THAN EDITING JSONL BY HAND. 100 records edited in a text editor is a
session that can be lost at passage 60 to one malformed line, with no way to tell which records
survived. The failure mode is worse than any subtlety in the annotation scheme, and it is removed
by roughly sixty lines of code. So: save after EVERY record, write atomically, and resume at the
first unannotated passage.

WHAT IT DELIBERATELY DOES NOT DO. It does not suggest entities, highlight candidate spans, or
pre-fill anything. The `entities` field must be produced by the annotator alone: a tool that
proposes and a human who confirms measures agreement with the proposer, which would make the
resulting number worthless in exactly the way this experiment exists to avoid. The tool shows
text and records what is typed.

BUILT BEFORE ANY ANNOTATION EXISTS. This file is committed while every record still has
`annotated: false`, so nothing in it could have been shaped by the answers it collects.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

RULE_CARD_VERSION = "v1"

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "results" / "003" / "extraction-sample.jsonl"


def load(path: Path = SAMPLE) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def save_atomic(rows: list[dict], path: Path = SAMPLE) -> None:
    """
    Write via a temp file in the same directory, then rename.

    os.replace is atomic on POSIX, so an interruption mid-write leaves the previous complete file
    rather than a truncated one. Writing in place would risk losing every annotation already made
    to a single crash — the exact failure this tool was built to prevent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def parse_entities(raw: str) -> list[str]:
    """
    Split a typed line into a SET of entity strings, order-stable.

    Comma-separated, because that is what a person types. Surface form is preserved exactly per
    rule card item 9 — no case folding, no canonicalisation — so normalisation happens once at
    scoring time under a rule fixed in code rather than differently on different days. Duplicates
    collapse (rule card: a set, matching how the graph deduplicates nodes); the order of first
    appearance is kept so the file stays readable and diffable.
    """
    seen, out = set(), []
    for part in raw.split(","):
        text = part.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def progress(rows: list[dict]) -> tuple[int, int]:
    return sum(1 for r in rows if r.get("annotated")), len(rows)


def next_index(rows: list[dict]) -> int | None:
    """First unannotated record, so an interrupted session resumes where it stopped."""
    for i, r in enumerate(rows):
        if not r.get("annotated"):
            return i
    return None


def record(row: dict, entities: list[str]) -> dict:
    """Stamp one annotation. The rule-card version travels WITH the record: if the card is ever
    revised mid-session, the cutover is visible per-passage instead of being reconstructed."""
    row["entities"] = entities
    row["annotated"] = True
    row["annotated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["rule_card"] = RULE_CARD_VERSION
    return row


def _render(row: dict, i: int, total: int, done: int) -> str:
    bar = f"[{done}/{total} done]"
    return (
        f"\n{'=' * 78}\n{bar}  passage {i + 1} of {total}   doc_id {row['doc_id']}\n"
        f"TITLE: {row['title']}\n{'-' * 78}\n{row['text']}\n{'=' * 78}\n"
    )


def main() -> None:  # pragma: no cover - interactive loop
    rows = load()
    print(f"Rule card {RULE_CARD_VERSION} — protocols/003-annotation-rules.md")
    print("Type entities comma-separated. Blank = none. 's' = skip. 'q' = save and quit.\n")
    while True:
        i = next_index(rows)
        if i is None:
            print("\nAll passages annotated.")
            break
        done, total = progress(rows)
        print(_render(rows[i], i, total, done))
        try:
            raw = input("entities> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped. Everything already answered is saved.")
            break
        if raw == "q":
            print("Saved.")
            break
        if raw == "s":
            rows.append(rows.pop(i))  # move to the end, revisit later
            save_atomic(rows)
            continue
        record(rows[i], parse_entities(raw))
        save_atomic(rows)
    done, total = progress(rows)
    print(f"{done}/{total} annotated.")


if __name__ == "__main__":  # pragma: no cover
    main()
