#!/usr/bin/env bash
# Runs the majority merge automatically once every majority-vote run has finished.
cd "$(dirname "$0")"
until grep -q "MAJ3 DONE" ../runs/maj3_done.log 2>/dev/null; do sleep 300; done
python3 merge_majority.py --tag "$(date +%Y%m%d)" > ../runs/maj3_merge.log 2>&1
echo "MERGED $(date -Is) rc=$?" >> ../runs/maj3_done.log
