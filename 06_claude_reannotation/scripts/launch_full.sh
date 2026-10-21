#!/usr/bin/env bash
# Full re-annotation, detached from any terminal or Claude session:
#   track 1: full corpus L1 (v9.1) starts immediately
#   track 2: v10.1 checks (definition recovery, pilot dev run), holdout L1 + L2 (run once),
#            then full corpus L2 (v10.1)
# Monitor with:  python3 scripts/monitor.py --watch 30
cd "$(dirname "$0")"
IN=../input
S=./supervise.sh
(
  $S $IN/corpus_20900.jsonl L1 full_low low system_L1.txt 4
) >> ../runs/full_track1.log 2>&1 &
(
  $S $IN/definitions_15.jsonl L2 defs_v101 low system_L2_v101.txt 4
  $S $IN/holdout_39.jsonl L1 holdout_low low system_L1.txt 4
  $S $IN/holdout_39.jsonl L2 holdout_low low system_L2_v101.txt 4
  $S $IN/pilot_100.jsonl L2 pilot_low_v101 low system_L2_v101.txt 4
  $S $IN/corpus_20900.jsonl L2 full_low low system_L2_v101.txt 4
) >> ../runs/full_track2.log 2>&1 &
wait
