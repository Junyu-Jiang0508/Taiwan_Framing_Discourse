#!/usr/bin/env bash
# Pilot: definition-recovery test, then pilot_100 at three effort levels (L1 and L2 in parallel).
set -u
cd "$(dirname "$0")"
IN=../input
python3 annotate.py --input $IN/definitions_15.jsonl --layer L1 --run defs_medium --effort medium --workers 4 &
python3 annotate.py --input $IN/definitions_15.jsonl --layer L2 --run defs_medium --effort medium --workers 4 &
wait
for E in low medium high; do
  python3 annotate.py --input $IN/pilot_100.jsonl --layer L1 --run pilot_$E --effort $E --workers 4 &
  python3 annotate.py --input $IN/pilot_100.jsonl --layer L2 --run pilot_$E --effort $E --workers 4 &
  wait
done
echo "PILOT DONE"
