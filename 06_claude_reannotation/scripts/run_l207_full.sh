#!/usr/bin/env bash
# Secondary configuration (pre-registered): L2-07 from the code-specific prompt, full corpus, run 1.
cd "$(dirname "$0")"
exec ./supervise.sh ../input/corpus_20900.jsonl CODE full_code_L2-07 low system_code_L2-07.txt 4 L2-07
