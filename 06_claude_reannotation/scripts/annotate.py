"""Annotate items with Claude Code headless mode (`claude -p`): one item per call, resumable.

Each call: frozen system prompt (protocol/system_<layer>[_rev].txt) replaces Claude Code's default
prompt; all tools off; output forced to protocol/schema_<layer>.json; run from a neutral directory
outside the repo so no project memory or CLAUDE.md is injected. Uses the logged-in subscription.

Usage:
  python annotate.py --input ../input/pilot_100.jsonl --layer L1 --run pilot_medium --effort medium
  python annotate.py ... --variant rev       # order-reversed codebook (order-invariance test)
  python annotate.py ... --limit 3           # smoke test

Writes runs/<run>/<layer>.jsonl (parsed, one line per item), <layer>_raw.jsonl (full CLI output),
config_<layer>.json and <layer>.log. Re-running the same command skips items already done.
Exit code 75 = stopped on a usage limit (resume later with the same command).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from common import L1, L2, NEUTRAL_CWD, PROTOCOL, RUNS, read_jsonl, sha256_file, user_text

MODEL = "claude-opus-5-5"
USAGE_LIMIT_RE = re.compile(r"(session|usage|weekly|daily|rate) limit|hit your .{0,20}limit|limit reached|limit resets|resets \d", re.I)
RESET_RE = re.compile(r"resets\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.I)


def parse_reset(msg):
    """'... resets 12:10pm (America/Los_Angeles)' -> next such local time (ISO), or None if not parseable."""
    m = RESET_RE.search(msg or "")
    if not m:
        return None
    h, mi, ap = int(m.group(1)) % 12, int(m.group(2) or 0), m.group(3).lower()
    h += 12 if ap == "pm" else 0
    now = datetime.now()
    t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return t.isoformat(timespec="seconds")
TRANSIENT_MARKERS = ("529", "overloaded", "429", "rate_limit", "rate limit", "timeout", "timed out",
                     "econnreset", "socket", "network", "502", "503", "500 ", "internal server error")


def validate(layer, obj):
    if not isinstance(obj, dict):
        return "not an object"
    if layer == "CODE":
        return None if isinstance(obj.get("value"), bool) else "missing/non-bool value"
    if layer == "L1":
        if obj.get("L1") not in L1:
            return f"illegal L1 {obj.get('L1')!r}"
        if obj.get("L1_secondary") not in L1 + [None]:
            return f"illegal L1_secondary {obj.get('L1_secondary')!r}"
        return None
    for lab in L2:
        if not isinstance(obj.get(lab), bool):
            return f"missing/non-bool {lab}"
    return None


def parse_cli(stdout):
    """Return (cli_json, structured_obj_or_None, error_text_or_None)."""
    try:
        cj = json.loads(stdout)
    except Exception:
        return None, None, f"cli output not json: {stdout[:300]!r}"
    if cj.get("is_error"):
        return cj, None, str(cj.get("result") or cj.get("subtype") or "is_error")
    obj = cj.get("structured_output")
    if obj is None:
        txt = (cj.get("result") or "").strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            txt = txt[txt.find("{"):]
        try:
            obj = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception:
            return cj, None, f"result not json: {txt[:300]!r}"
    return cj, obj, None


class Runner:
    def __init__(self, args):
        self.args = args
        self.layer = args.layer
        suffix = "_rev" if args.variant == "rev" else ""
        self.sys_path = PROTOCOL / (args.prompt or f"system_{self.layer}{suffix}.txt")
        self.schema_path = PROTOCOL / ("schema_code.json" if self.layer == "CODE" else f"schema_{self.layer}.json")
        self.schema = json.dumps(json.loads(self.schema_path.read_text(encoding="utf-8")), ensure_ascii=False)
        self.run_dir = RUNS / args.run
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tag = args.tag or self.layer
        self.out_path = self.run_dir / f"{self.tag}.jsonl"
        self.raw_path = self.run_dir / f"{self.tag}_raw.jsonl"
        self.log_path = self.run_dir / f"{self.tag}.log"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_reason = None
        self.n_done = self.n_fail = 0
        self.t0 = time.time()
        self.env = dict(os.environ, DISABLE_AUTOUPDATER="1")

    def log(self, msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, file=sys.stderr, flush=True)
        with self.lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def cmd(self):
        return ["claude", "-p", "--model", MODEL, "--effort", self.args.effort,
                "--system-prompt-file", str(self.sys_path),
                "--json-schema", self.schema,
                "--tools", "",
                "--output-format", "json",
                "--no-session-persistence", "--safe-mode", "--strict-mcp-config"]

    def call(self, it):
        msg = user_text(it.get("prev"), it.get("sentence"), it.get("next"))
        last_err, cj = None, None
        for attempt in range(1, self.args.max_attempts + 1):
            if self.stop.is_set():
                return None
            t = time.time()
            try:
                p = subprocess.run(self.cmd(), input=msg, cwd=NEUTRAL_CWD, capture_output=True, text=True,
                                   timeout=self.args.timeout, env=self.env)
                stdout, stderr = p.stdout, p.stderr
            except subprocess.TimeoutExpired:
                stdout, stderr = "", "timeout"
            cj, obj, err = parse_cli(stdout) if stdout.strip() else (None, None, f"empty stdout; stderr={stderr[:300]!r}")
            if err is None:
                err = validate(self.layer, obj)
            with self.lock, self.raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"item_id": it["item_id"], "attempt": attempt, "stdout": stdout,
                                    "stderr": stderr[-2000:], "wall_s": round(time.time() - t, 2)},
                                   ensure_ascii=False) + "\n")
            if err is None:
                return self.record(it, obj, cj, attempt, None)
            last_err = err
            low = (err + " " + stderr).lower()
            if USAGE_LIMIT_RE.search(err + " " + stderr):
                self.stop_reason = err
                if not self.stop.is_set():
                    self.stop.set()
                    reset = parse_reset(err + " " + stderr)
                    (self.run_dir / f"{self.tag}.reset").write_text(reset or "", encoding="utf-8")
                    self.log(f"USAGE LIMIT -> stopping (reset {reset or 'unknown'}): {err[:200]}")
                return None
            backoff = min(300, 10 * 2 ** (attempt - 1)) if any(m in low for m in TRANSIENT_MARKERS) else 3
            self.log(f"{it['item_id']} attempt {attempt} failed: {err[:200]} (sleep {backoff}s)")
            time.sleep(backoff)
        return self.record(it, None, cj, self.args.max_attempts, last_err)

    def record(self, it, obj, cj, attempt, err):
        cj = cj or {}
        row = {
            "item_id": it["item_id"], "layer": self.layer, "ok": err is None, "error": err,
            "parsed": obj, "attempts": attempt,
            "model_used": list((cj.get("modelUsage") or {}).keys()),
            "usage": cj.get("usage"), "cost_usd": cj.get("total_cost_usd"),
            "duration_ms": cj.get("duration_ms"), "duration_api_ms": cj.get("duration_api_ms"),
            "num_turns": cj.get("num_turns"), "ts": datetime.now().isoformat(timespec="seconds"),
        }
        with self.lock:
            with self.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if err is None:
                self.n_done += 1
            else:
                self.n_fail += 1
            n = self.n_done + self.n_fail
        if n % self.args.log_every == 0:
            el = time.time() - self.t0
            self.log(f"progress {n}/{self.todo} ok={self.n_done} fail={self.n_fail} "
                     f"elapsed={el/60:.1f}min rate={n/el*60:.1f}/min")
        return row

    def run(self):
        items = read_jsonl(self.args.input)
        done = {r["item_id"] for r in read_jsonl(self.out_path) if r.get("ok")}
        if not self.args.retry_failed:
            done |= {r["item_id"] for r in read_jsonl(self.out_path) if not r.get("ok")}
        todo = [it for it in items if it["item_id"] not in done]
        if self.args.limit:
            todo = todo[: self.args.limit]
        self.todo = len(todo)
        cfg = {
            "model": MODEL, "effort": self.args.effort, "variant": self.args.variant, "layer": self.layer,
            "system_prompt": self.sys_path.name, "system_prompt_sha256": sha256_file(self.sys_path),
            "schema_sha256": sha256_file(self.schema_path), "input": str(self.args.input),
            "input_sha256": sha256_file(self.args.input), "workers": self.args.workers,
            "cli_version": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip(),
            "cwd": str(NEUTRAL_CWD),
            "command": [c if c != self.schema else self.schema_path.name for c in self.cmd()],
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        cfg_path = self.run_dir / f"config_{self.tag}.json"
        if cfg_path.exists():
            prev = json.loads(cfg_path.read_text(encoding="utf-8"))
            for k in ["model", "effort", "variant", "system_prompt_sha256", "schema_sha256"]:
                assert prev[k] == cfg[k], f"run dir {self.run_dir} was started with a different {k}"
            cfg["resumed_from"] = prev.get("started")
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        self.log(f"start {self.args.run}/{self.layer}: {len(todo)} to do ({len(done)} already done), "
                 f"effort={self.args.effort} variant={self.args.variant} workers={self.args.workers}")
        NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=self.args.workers) as ex:
            list(ex.map(self.call, todo))
        self.log(f"end: ok={self.n_done} fail={self.n_fail} stopped={self.stop.is_set()}")
        return 75 if self.stop.is_set() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--layer", required=True, choices=["L1", "L2", "CODE"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--variant", default="std", choices=["std", "rev"])
    ap.add_argument("--tag", default=None, help="output file stem (default: layer); e.g. L2-07 for a single-code run")
    ap.add_argument("--prompt", default=None, help="system prompt file in protocol/ (overrides layer/variant default)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    sys.exit(Runner(ap.parse_args()).run())


if __name__ == "__main__":
    main()
