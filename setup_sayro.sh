#!/usr/bin/env bash
# Separate Python environment for uzlm/sayro-tts-1.7B (Qwen3-TTS based).
# Separate because the qwen-tts package pins transformers==4.57.3, which would
# downgrade the main .venv used by Whisper and the other scripts.
#
# Usage:  bash setup_sayro.sh      (then: source .venv-tts/bin/activate)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv-tts ]; then
  if command -v uv >/dev/null 2>&1; then
    # Python 3.12: some qwen-tts dependencies (onnxruntime, sox) may not have 3.14 wheels yet
    uv venv .venv-tts --python 3.12 --seed
  else
    python3 -m venv .venv-tts
  fi
fi
# shellcheck disable=SC1091
source .venv-tts/bin/activate
python --version
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U qwen-tts soundfile huggingface_hub

python - <<'EOF'
import torch, qwen_tts
assert torch.cuda.is_available(), "CUDA not available in .venv-tts"
print("OK: qwen-tts installed, GPU:", torch.cuda.get_device_name(0))
EOF

echo
echo "Before running tts_sayro.py:"
echo "  1. Open https://huggingface.co/uzlm/sayro-tts-1.7B (logged in), request access and read the license"
echo "  2. source .venv-tts/bin/activate && hf auth login   (same token as before is fine)"
echo "  3. python tts_sayro.py"
echo
echo "If 'pip install qwen-tts' failed on Python 3.14: install uv (curl -LsSf https://astral.sh/uv/install.sh | sh),"
echo "then: rm -rf .venv-tts && bash setup_sayro.sh"
