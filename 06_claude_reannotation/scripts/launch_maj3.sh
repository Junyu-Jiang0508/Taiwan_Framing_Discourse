#!/usr/bin/env bash
# 3-run majority vote (2 of 3): two more full runs of L1 (v9.1) and L2 (v10.1), same protocol as full_low;
# then two more runs of the L2-07 code-specific prompt (pre-registered secondary configuration).
cd "$(dirname "$0")"
IN=../input/corpus_20900.jsonl
S=./supervise.sh
for R in r2 r3; do
  ( $S $IN L1 full_low_$R low system_L1.txt 3 ) >> ../runs/maj3_L1_$R.log 2>&1 &
  ( $S $IN L2 full_low_$R low system_L2_v101.txt 3 ) >> ../runs/maj3_L2_$R.log 2>&1 &
done
# code-specific L2-07 reruns start after the first code-specific run has finished
( until grep -q '^done' ../runs/full_code_L2-07/L2-07.status 2>/dev/null; do sleep 120; done
  for R in r2 r3; do $S $IN CODE full_code_L2-07_$R low system_code_L2-07.txt 4 L2-07 & done; wait ) >> ../runs/maj3_code.log 2>&1 &
wait
echo "MAJ3 DONE $(date -Is)" >> ../runs/maj3_done.log
