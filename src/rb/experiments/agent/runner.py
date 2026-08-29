"""
Experiment 006 — invoking Claude Code as the model under test.

001-005 measured retrieval: does an arm rank the right documents. 006 measures answering:
does an agent get the question right, and what did it cost. The subject is therefore an
agent PROCESS, not an API call, which is why this shells out to the `claude` CLI. That makes
the harness part of the experimental condition, and the flag set below is not incidental --
each flag closes a leak that was measured, not assumed.

  --setting-sources ""   The user's global ~/.claude/CLAUDE.md reaches the model even from an
                         empty temp directory, because settings load at user level and not
                         from cwd. Probed directly: without this flag the model quotes that
                         file back verbatim. An experiment whose every arm silently inherits
                         the author's standing instructions is not measuring the arms.
  --restricted           --add-dir GRANTS access to the corpus; it does not CONFINE the agent
                         to it. Probed directly: a grep agent given --add-dir <corpus> read a
                         file outside that directory. Without --restricted the grep arm can
                         reach the rest of the filesystem, including this repository and the
                         gold answers in it.
  --strict-mcp-config    No MCP server reaches an arm.
  fresh temp cwd         Cache reuse turned out to be content-addressed and TTL-based, NOT
                         keyed on the working directory: two calls in DIFFERENT temp dirs
                         reused the cache fully, three CONCURRENT calls all missed it. So a
                         shared directory buys nothing and costs isolation (shared .claude/
                         session state across concurrent workers). Fresh dir per call stays.

EXIT CODE 1 IS NOT NECESSARILY A FAILURE. On --max-turns exhaustion the CLI exits non-zero
with an empty stderr and a complete JSON result on stdout, carrying the usage, cost and turn
count. The first version of this file returned early on returncode != 0 and threw all of that
away, recording the run as an unexplained "exit 1". stdout is now parsed first and the exit
code is only consulted when there is no JSON to read.

OUTCOMES ARE TYPED, because three different things were previously collapsed into one empty
answer that scored as wrong: a model that ran out of turns mid-search, a process the harness
killed, and an API error. Only the first is a fact about the model. See Outcome below.
"""

import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

CLAUDE = shutil.which("claude") or "claude"
TIMEOUT_S = 900

# Applied to every arm without exception. Any difference between arms must come from the
# prompt, the tool grant or the corpus -- never from the harness configuration.
BASE_FLAGS = ["--output-format", "json", "--setting-sources", "", "--strict-mcp-config",
              "--restricted"]


class Outcome:
    """How a call ended. Registered in the protocol because each is scored differently.

    OK                  the model committed an answer; scored normally.
    MAX_TURNS           the agent was cut off mid-investigation. A real capability/cost
                        outcome, not noise: scored as wrong AND reported separately, because
                        the arm that does more tool round-trips is mechanically more exposed
                        to it and hiding that would flatter the injected arms.
    API_ERROR, TIMEOUT  infrastructure, not retrieval. Retried once, then excluded from the
                        denominator and reported as a rate. Scoring these as wrong would
                        penalise the grep arm for taking longer, which is a property of the
                        harness rather than of the model's reasoning.
    """
    OK = "ok"
    MAX_TURNS = "max_turns"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    NO_JSON = "no_json"


@dataclass
class Call:
    query_id: str
    arm: str
    model: str
    answer: str = ""
    outcome: str = Outcome.OK
    error: str | None = None
    # --- what the model actually saw and did ---
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    # --- provenance, so a reader can audit without re-running ---
    resolved_model: str = ""
    session_id: str = ""
    service_tier: str = ""
    stop_reason: str = ""
    terminal_reason: str = ""
    permission_denials: int = 0
    attempt: int = 1
    injected_tokens_est: int = 0
    cmd: list = field(default_factory=list)


def context_tokens(c: Call) -> int:
    """Everything the model conditioned on this call, cached or not.

    input_tokens alone understates it badly: on a cache hit the corpus the model read is
    counted under cache_read instead. The harness's own fixed overhead is subtracted
    separately by the analysis, using the measured zero-context baseline, so that the
    reported figure is context the ARM supplied rather than context the CLI did.
    """
    return c.input_tokens + c.cache_read_tokens + c.cache_creation_tokens


def _parse_result(stdout: str) -> dict | None:
    try:
        d = json.loads(stdout)
        return d if isinstance(d, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _fill(c: Call, d: dict) -> Call:
    u = d.get("usage", {}) or {}
    c.answer = (d.get("result") or "").strip()
    c.num_turns = d.get("num_turns", 0) or 0
    c.input_tokens = u.get("input_tokens", 0) or 0
    c.output_tokens = u.get("output_tokens", 0) or 0
    c.thinking_tokens = (u.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0
    c.cache_read_tokens = u.get("cache_read_input_tokens", 0) or 0
    c.cache_creation_tokens = u.get("cache_creation_input_tokens", 0) or 0
    c.service_tier = u.get("service_tier", "") or ""
    c.cost_usd = d.get("total_cost_usd", 0.0) or 0.0
    c.duration_ms = d.get("duration_ms", 0) or 0
    c.session_id = d.get("session_id", "") or ""
    c.stop_reason = d.get("stop_reason", "") or ""
    c.terminal_reason = d.get("terminal_reason", "") or ""
    c.permission_denials = len(d.get("permission_denials", []) or [])
    # The resolved snapshot id, e.g. haiku -> claude-haiku-4-5-20251001. The alias alone is
    # not a pin: it moves when a new model ships.
    mu = d.get("modelUsage") or {}
    c.resolved_model = next(iter(mu), "")

    if d.get("subtype") == "error_max_turns" or d.get("terminal_reason") == "max_turns":
        c.outcome = Outcome.MAX_TURNS
        c.error = "max_turns"
    elif d.get("is_error"):
        c.outcome = Outcome.API_ERROR
        c.error = d.get("api_error_status") or "is_error"
    return c


def _once(query_id: str, arm: str, model: str, prompt: str, system: str,
          allowed_tools: str, add_dir: str | None, max_turns: int | None,
          attempt: int) -> Call:
    cwd = tempfile.mkdtemp(prefix="rb006-")
    cmd = [CLAUDE, "-p", prompt, "--model", model, *BASE_FLAGS,
           "--system-prompt", system, "--allowedTools", allowed_tools]
    if add_dir:
        cmd += ["--add-dir", add_dir]
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]

    c = Call(query_id=query_id, arm=arm, model=model, attempt=attempt,
             cmd=[x if x != prompt else "<PROMPT>" for x in cmd])
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=TIMEOUT_S, stdin=subprocess.DEVNULL)
        d = _parse_result(p.stdout)
        if d is None:
            c.outcome = Outcome.NO_JSON
            c.error = f"exit {p.returncode}: {(p.stderr or p.stdout)[-400:]}"
        else:
            _fill(c, d)
    except subprocess.TimeoutExpired:
        c.outcome, c.error = Outcome.TIMEOUT, f"timeout after {TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001 - the driver records failures, it does not raise
        c.outcome, c.error = Outcome.NO_JSON, f"{type(e).__name__}: {e}"
    finally:
        c.duration_ms = c.duration_ms or int((time.time() - t0) * 1000)
        shutil.rmtree(cwd, ignore_errors=True)
    return c


# Infrastructure failures get one retry; a model that ran out of turns does not, because
# that is the measurement.
RETRYABLE = {Outcome.TIMEOUT, Outcome.API_ERROR, Outcome.NO_JSON}


def invoke(query_id: str, arm: str, model: str, prompt: str, system: str,
           allowed_tools: str = "", add_dir: str | None = None,
           max_turns: int | None = None, injected_tokens_est: int = 0) -> Call:
    c = _once(query_id, arm, model, prompt, system, allowed_tools, add_dir, max_turns, 1)
    if c.outcome in RETRYABLE:
        time.sleep(2)
        c = _once(query_id, arm, model, prompt, system, allowed_tools, add_dir, max_turns, 2)
    c.injected_tokens_est = injected_tokens_est
    return c


def cli_version() -> str:
    try:
        return subprocess.run([CLAUDE, "--version"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_all(jobs: list[dict], out_path: Path, workers: int = 4, on_done=None) -> list[Call]:
    """Run jobs concurrently, appending each result to a jsonl as it lands.

    Resumable: a job already present in the output is skipped, so a killed run continues
    rather than restarting and re-spending. Callers are expected to hand jobs in an
    interleaved order across arms -- see driver.shuffle_jobs -- so that a drift in service
    conditions over the run lands on every arm equally instead of on whichever arm was
    scheduled last.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["query_id"], d["arm"], d["model"]))
    todo = [j for j in jobs if (j["query_id"], j["arm"], j["model"]) not in done]

    results: list[Call] = []
    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for c in ex.map(lambda j: invoke(**j), todo):
            fh.write(json.dumps(asdict(c)) + "\n")
            fh.flush()
            results.append(c)
            if on_done:
                on_done(c, len(results), len(todo))
    return results
