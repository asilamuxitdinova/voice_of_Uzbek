#!/usr/bin/env bash
# Step 1.1b - compare ready-made Uzbek Whisper models on the same 100 FLEURS test clips.
# Usage:  source .venv/bin/activate && bash asr_compare.sh [N]
# Each model is ~1.5 GB (downloaded once to ~/.cache/huggingface).
#
# Caveat: navai-uz/* models were trained with FLEURS data, so their FLEURS score may look
# better than real-world use. Test on your own recordings too (python asr_baseline.py --audio ...).
set -uo pipefail
cd "$(dirname "$0")"
N="${1:-100}"

MODELS=(
  navai-uz/whisper-medium-uzbek
  Abduqayum/whisper-uzbek-medium-callcenter
  navai-uz/whisper-small-uzbek
)

for m in "${MODELS[@]}"; do
  echo "=================== $m"
  python asr_baseline.py --model "$m" --n "$N" || echo "FAILED: $m"
done

echo
echo "=================== Summary"
grep -H "^WER\|^CER\|avg seconds" results/asr_*.txt
