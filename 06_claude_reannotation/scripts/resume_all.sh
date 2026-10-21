#!/usr/bin/env bash
# Resume every unfinished majority-vote job after a shutdown or crash. Safe to run more than once:
# each job skips items already in its output file, and nothing is started twice.
# Low-load settings (2026-10-19, after 20 concurrent claude processes froze the desktop):
#   10 concurrent calls in total, all under `nice -n 15` so the desktop and VS Code stay responsive.
cd "$(dirname "$0")"
if pgrep -f "^python3 annotate\.py" > /dev/null; then
  echo "annotation jobs are already running — not starting a second copy"; exit 1
fi
IN=../input/corpus_20900.jsonl
S="nice -n 15 ./supervise.sh"
(
  $S $IN L1   full_low_r2        low system_L1.txt        1       &
  $S $IN L1   full_low_r3        low system_L1.txt        1       &
  $S $IN L2   full_low_r2        low system_L2_v101.txt   2       &
  $S $IN L2   full_low_r3        low system_L2_v101.txt   2       &
  $S $IN CODE full_code_L2-07_r2 low system_code_L2-07.txt 2 L2-07 &
  $S $IN CODE full_code_L2-07_r3 low system_code_L2-07.txt 2 L2-07 &
  wait
  echo "MAJ3 DONE $(date -Is)" >> ../runs/maj3_done.log
) >> ../runs/maj3_resume.log 2>&1 < /dev/null &
disown
setsid nohup ./after_maj3.sh > /dev/null 2>&1 < /dev/null &
sleep 5
echo "resumed $(date -Is) with 10 concurrent calls at low priority; progress: python3 $(pwd)/monitor.py --watch 30"
