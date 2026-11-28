#!/usr/bin/env bash
# One-time setup for GigaAM fine-tuning (Phase 2). Run from the repo root inside .venv:
#   source .venv/bin/activate && bash setup_gigaam_train.sh
#
# Clones the official GigaAM repo (MIT) at a reviewed commit into third_party/GigaAM and installs
# what training needs. The gigaam package is installed WITHOUT its pinned dependencies, because it
# pins onnx/onnxruntime versions (only used for model export) that may not exist for Python 3.14.
set -euo pipefail
cd "$(dirname "$0")"

GIGAAM_COMMIT="e9bbf2e773eedd217c6f0d3b89b9728c41332b7d"   # reviewed 2026-11-27
DIR=third_party/GigaAM

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "Activate the venv first: source .venv/bin/activate"; exit 1
fi

if [ ! -d "$DIR/.git" ]; then
  mkdir -p third_party
  git clone https://github.com/salute-developers/GigaAM "$DIR"
fi
git -C "$DIR" fetch -q origin "$GIGAAM_COMMIT" 2>/dev/null || git -C "$DIR" fetch -q origin
git -C "$DIR" checkout -q "$GIGAAM_COMMIT"
echo "GigaAM at $(git -C "$DIR" log -1 --format='%h %cd')"

pip install --no-deps -e "$DIR"
pip install hydra-core omegaconf sentencepiece soundfile tqdm pyarrow \
            pytorch-lightning editdistance tensorboard

python - <<'EOF'
import torch, pytorch_lightning, editdistance, gigaam  # noqa: F401
print("OK: gigaam", getattr(gigaam, "__version__", ""), "| lightning", pytorch_lightning.__version__,
      "| CUDA", torch.cuda.is_available(), "| bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
EOF
echo
echo "Next: python prepare_data.py   (build the dataset)"
echo "      python train_gigaam.py --trial   (short speed test)"
