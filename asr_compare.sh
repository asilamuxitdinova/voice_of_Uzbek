#!/usr/bin/env bash
# Step 1.1b - compare ready-made Uzbek Whisper models on the same test data.
#
# Usage (inside .venv):
#   bash asr_compare.sh                              # 100 FLEURS clips
#   bash asr_compare.sh 100 --beams 5                # with beam search
#   bash asr_compare.sh 0 --audio-dir test_audios    # your own recordings (N is ignored)
# Any extra options are passed to asr_baseline.py. Models are 0.9-2.4 GB each (downloaded once).
# --beams only affects the Whisper models (GigaAM is CTC, greedy).
#
# Caveat: navai-uz/* models were trained with FLEURS data, so their FLEURS score may look
# better than real-world use. Your own recordings (--audio-dir) are the honest test.
set -uo pipefail
cd "$(dirname "$0")"
N="${1:-100}"
shift || true

MODELS=(
  ai-sage/GigaAM-Multilingual            # large_ctc, 600M (MIT)
  ai-sage/GigaAM-Multilingual@ctc        # 220M, faster
  navai-uz/whisper-medium-uzbek
  Kotib/uzbek_stt_v1                     # whisper-medium, ~1,600 h, handles Russian loanwords
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
