#!/usr/bin/env bash
# Step 1.1b - compare ready-made Uzbek Whisper models on the same test data.
#
# Usage (inside .venv):
#   bash asr_compare.sh                              # 100 FLEURS clips
#   bash asr_compare.sh 100 --beams 5                # with beam search
#   bash asr_compare.sh 0 --audio-dir test_audios    # your own recordings (N is ignored)
# Any extra options are passed to asr_baseline.py. Each model is ~1.5 GB (downloaded once).
#
# Caveat: navai-uz/* models were trained with FLEURS data, so their FLEURS score may look
# better than real-world use. Your own recordings (--audio-dir) are the honest test.
set -uo pipefail
cd "$(dirname "$0")"
N="${1:-100}"
shift || true

MODELS=(
  navai-uz/whisper-medium-uzbek
  Abduqayum/whisper-uzbek-medium-callcenter
  navai-uz/whisper-small-uzbek
)

for m in "${MODELS[@]}"; do
  echo "=================== $m $*"
  python asr_baseline.py --model "$m" --n "$N" "$@" || echo "FAILED: $m"
done

echo
echo "=================== Summary"
grep -H "^WER\|^CER\|avg seconds\|^speed" results/asr_*.txt
