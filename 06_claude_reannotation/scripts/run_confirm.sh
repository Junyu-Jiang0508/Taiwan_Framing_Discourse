#!/usr/bin/env bash
# After choosing an effort level on the pilot: stability (rerun + order-reversed) and the holdout.
# Usage: ./run_confirm.sh medium
set -u
E=${1:?effort level}
cd "$(dirname "$0")"
IN=../input
for L in L1 L2; do
  python3 annotate.py --input $IN/stability_30.jsonl --layer $L --run stab_rerun_$E --effort $E --workers 3 &
  python3 annotate.py --input $IN/stability_30.jsonl --layer $L --run stab_rev_$E --effort $E --variant rev --workers 3 &
done
wait
for L in L1 L2; do
  python3 annotate.py --input $IN/holdout_39.jsonl --layer $L --run holdout_$E --effort $E --workers 4 &
done
wait
echo "CONFIRM DONE"
