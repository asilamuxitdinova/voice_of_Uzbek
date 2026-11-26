#!/usr/bin/env bash
# Phase 0: prepare the server. Run from the repo root:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 System packages =="
sudo apt update
sudo apt install -y ffmpeg git git-lfs build-essential cmake python3-venv python3-dev

echo "== 2/5 Disk space (need ~100-150 GB free for later phases) =="
df -h .

echo "== 3/5 Python virtual environment =="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "== 4/5 PyTorch (CUDA) + libraries =="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

echo "== 5/5 GPU check =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA is NOT available - check the driver / torch install"
print("OK, GPU:", torch.cuda.get_device_name(0))
EOF

mkdir -p results
echo
echo "Setup finished. Next steps:"
echo "  source .venv/bin/activate"
echo "  hf auth login"
