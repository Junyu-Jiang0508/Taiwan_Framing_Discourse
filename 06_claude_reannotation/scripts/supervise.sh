#!/usr/bin/env bash
# Keep one annotate.py job running to completion across usage-limit stops.
# Usage: supervise.sh <input.jsonl> <layer> <run> <effort> <prompt_file> [workers] [tag]
# Status is written to runs/<run>/<layer>.status (read by monitor.py).
set -u
cd "$(dirname "$0")"
IN=$1; LAYER=$2; RUN=$3; EFFORT=$4; PROMPT=$5; W=${6:-4}; TAG=${7:-$LAYER}
mkdir -p ../runs/$RUN
ST=../runs/$RUN/$TAG.status
fails=0
while true; do
  echo "running $(date -Is)" > $ST
  python3 annotate.py --input $IN --layer $LAYER --run $RUN --effort $EFFORT --prompt $PROMPT --workers $W --tag $TAG --retry-failed --log-every 200
  rc=$?
  if [ $rc -eq 0 ]; then
    # items that failed all attempts get one more pass
    python3 annotate.py --input $IN --layer $LAYER --run $RUN --effort $EFFORT --prompt $PROMPT --workers $W --tag $TAG --retry-failed --log-every 200
    echo "done $(date -Is)" > $ST; exit 0
  elif [ $rc -eq 75 ]; then
    # sleep until the reset time the CLI reported (+2 min); 30 min if it could not be parsed
    RS=$(cat ../runs/$RUN/$TAG.reset 2>/dev/null); WAIT=1800
    if [ -n "$RS" ]; then WAIT=$(( $(date -d "$RS" +%s) - $(date +%s) + 120 )); [ $WAIT -lt 60 ] && WAIT=60; fi
    echo "paused-usage-limit $(date -Is) retry-at $(date -Is -d "+$WAIT sec")" > $ST; sleep $WAIT
  else
    fails=$((fails+1)); echo "error rc=$rc $(date -Is) (crash $fails/5)" > $ST
    [ $fails -ge 5 ] && exit $rc
    sleep 300
  fi
done
