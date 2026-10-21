#!/usr/bin/env bash
# P4 experiments on dev data / corpus (never on gold labels): code-specific prompts, self-consistency, party-cue masking.
cd "$(dirname "$0")"
IN=../input
A="python3 annotate.py --effort low --workers 3 --log-every 50"
$A --input $IN/pilot_100.jsonl --layer CODE --run p4_code --tag L2-07 --prompt system_code_L2-07.txt &
$A --input $IN/pilot_100.jsonl --layer CODE --run p4_code --tag L2-01 --prompt system_code_L2-01.txt &
$A --input $IN/pilot_100.jsonl --layer L2 --run pilot_low_v101_r2 --prompt system_L2_v101.txt &
$A --input $IN/pilot_100.jsonl --layer L2 --run pilot_low_v101_r3 --prompt system_L2_v101.txt &
wait
$A --input $IN/mask_300.jsonl --layer L1 --run p4_mask --prompt system_L1.txt &
$A --input $IN/mask_300.jsonl --layer L2 --run p4_mask --prompt system_L2_v101.txt &
$A --input $IN/mask_300_orig.jsonl --layer L1 --run p4_mask_ctrl --prompt system_L1.txt &
$A --input $IN/mask_300_orig.jsonl --layer L2 --run p4_mask_ctrl --prompt system_L2_v101.txt &
wait
echo P4 DONE
