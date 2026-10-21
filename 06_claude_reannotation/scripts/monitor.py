"""Progress and ETA for the re-annotation jobs.

Usage (from the repo root or anywhere):
  python3 06_claude_reannotation/scripts/monitor.py            # one snapshot
  python3 06_claude_reannotation/scripts/monitor.py --watch 30 # refresh every 30 s (Ctrl-C to quit)
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
JOBS = [  # (run, layer, label) in launch order
    ("defs_v101", "L2", "定义回收 L2 v10.1"),
    ("holdout_low", "L1", "留出集 L1"),
    ("holdout_low", "L2", "留出集 L2 v10.1"),
    ("pilot_low_v101", "L2", "试点 L2 v10.1（开发集）"),
    ("full_low", "L1", "全量 L1"),
    ("full_low", "L2", "全量 L2 v10.1"),
    ("full_code_L2-07", "L2-07", "全量 L2-07 专用提示（次要配置）"),
    ("full_low_r2", "L1", "多数票 第2次 L1"),
    ("full_low_r2", "L2", "多数票 第2次 L2"),
    ("full_low_r3", "L1", "多数票 第3次 L1"),
    ("full_low_r3", "L2", "多数票 第3次 L2"),
    ("full_code_L2-07_r2", "L2-07", "多数票 第2次 L2-07 专用"),
    ("full_code_L2-07_r3", "L2-07", "多数票 第3次 L2-07 专用"),
]
_total_cache = {}


def total_of(cfg):
    p = cfg.get("input")
    if p not in _total_cache:
        path = (RUNS.parent / "scripts" / p).resolve() if not os.path.isabs(p) else Path(p)
        with open(path, encoding="utf-8") as f:
            _total_cache[p] = sum(1 for _ in f)
    return _total_cache[p]


def stats(run, layer):
    d = RUNS / run
    cfgp, outp = d / f"config_{layer}.json", d / f"{layer}.jsonl"  # layer doubles as the output tag
    status = (d / f"{layer}.status").read_text().strip() if (d / f"{layer}.status").exists() else "waiting"
    if not cfgp.exists():
        return dict(status="waiting", done=0, fail=0, total=None, rate=None, eta=None)
    total = total_of(json.loads(cfgp.read_text(encoding="utf-8")))
    ok, fail, ts = set(), set(), []
    if outp.exists():
        for line in outp.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("ok"):
                ok.add(r["item_id"]); fail.discard(r["item_id"]); ts.append(r["ts"])
            elif r["item_id"] not in ok:
                fail.add(r["item_id"])
    rate = eta = None
    recent = sorted(ts)[-200:]
    # use only the latest uninterrupted stretch: a pause (shutdown, usage limit) would deflate the rate
    times = [datetime.fromisoformat(x) for x in recent]
    gaps = [k for k in range(1, len(times)) if (times[k] - times[k - 1]).total_seconds() > 600]
    if gaps:
        recent = recent[gaps[-1]:]
    if len(recent) >= 10:
        t0 = datetime.fromisoformat(recent[0]); t1 = datetime.fromisoformat(recent[-1])
        mins = max((t1 - t0).total_seconds() / 60, 1e-6)
        rate = (len(recent) - 1) / mins
        left = total - len(ok)
        if status.startswith("running") and rate > 0:
            eta = left / rate
    return dict(status=status.split(" ")[0], done=len(ok), fail=len(fail), total=total, rate=rate, eta=eta)


def bar(done, total, width=24):
    if not total:
        return " " * width
    k = int(width * done / total)
    return "█" * k + "░" * (width - k)


def hm(minutes):
    if minutes is None:
        return "—"
    h, m = divmod(int(minutes), 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def snapshot():
    procs = subprocess.run(["pgrep", "-fc", "annotate.py"], capture_output=True, text=True).stdout.strip() or "0"
    lines = [f"Claude 重标注进度  {datetime.now():%Y-%m-%d %H:%M:%S}   运行中的 annotate 进程: {procs}", ""]
    lines.append(f"{'任务':<22}{'状态':<20}{'进度':<26}{'完成/总数':>14}{'失败':>6}{'速度/分':>9}{'剩余':>9}")
    for run, layer, label in JOBS:
        s = stats(run, layer)
        tot = s["total"] or 0
        pct = f"{100*s['done']/tot:5.1f}%" if tot else "   —"
        rate = f"{s['rate']:.1f}" if s["rate"] else "—"
        lines.append(f"{label:<20}{s['status']:<20}{bar(s['done'], tot)} {pct}"
                     f"{s['done']:>8}/{tot if tot else '—':<6}{s['fail']:>5}{rate:>9}{hm(s['eta']):>9}")
    maj = [stats(r, l) for r, l, _ in JOBS if r.startswith(("full_low_r", "full_code_L2-07_r"))]
    done = sum(f["done"] for f in maj); tot = 6 * 20900
    etas = [f["eta"] for f in maj if f["eta"]]
    lines += ["", f"多数票补跑合计 {done:,}/{tot:,}（{100*done/tot:.1f}%）"
              + (f"；正在运行的任务中最慢的约剩 {hm(max(etas))}" if etas else "")]
    lines += ["", "状态说明：running 运行中 / paused-usage-limit 碰到额度上限、30 分钟后自动续跑 / done 完成 / waiting 排队",
              "L2 全量在留出集与开发集检查跑完后自动开始；剩余时间按最近 200 条的速度估算。"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    if not a.watch:
        print(snapshot()); return
    try:
        while True:
            os.system("clear")
            print(snapshot())
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
