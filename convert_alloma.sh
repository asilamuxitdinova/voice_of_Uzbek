#!/usr/bin/env bash
# Convert uzlm/alloma-8B-Instruct (Hugging Face, 16 GB) into a Q4_K_M GGUF (~5 GB) for kronk.
#
# Usage (from the repo root, with kronk free to keep running):
#     bash convert_alloma.sh
# Options via environment variables:
#     MODEL_ID=uzlm/alloma-3B-Instruct bash convert_alloma.sh   # other size
#     QUANT=Q5_K_M bash convert_alloma.sh                        # other quantization
#     KEEP_HF=1 bash convert_alloma.sh                           # keep the HF weights (needed later for QLoRA)
#
# Peak disk use: ~35 GB for the 8B model (HF weights + f16 GGUF). Intermediates are deleted at the end.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="${MODEL_ID:-uzlm/alloma-8B-Instruct}"
QUANT="${QUANT:-Q4_K_M}"
NAME="$(basename "$MODEL_ID" | tr 'A-Z' 'a-z')"        # e.g. alloma-8b-instruct
WORK="${WORK:-$HOME/convert_work}"
LLAMA="$WORK/llama.cpp"
HF_DIR="$WORK/$NAME-hf"
F16="$WORK/$NAME-f16.gguf"
OUT="$HOME/models/$NAME-$QUANT.gguf"
KRONK_DIR="$HOME/.kronk/models/uzlm/$NAME"

mkdir -p "$WORK" "$HOME/models"

echo "== 1/7 Disk space check =="
avail=$(df --output=avail -BG "$WORK" | tail -1 | tr -dc '0-9')
echo "Free: ${avail} GB (need ~35 GB for 8B)"
if [ "$avail" -lt 35 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "Not enough space. Free some up (see PROGRESS.md tips) or run with FORCE=1."
  exit 1
fi

echo "== 2/7 Get llama.cpp =="
if [ ! -d "$LLAMA" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
else
  git -C "$LLAMA" pull --ff-only || true
fi

echo "== 3/7 Build llama-quantize (CPU only, takes a few minutes) =="
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA/build" --target llama-quantize -j"$(nproc)"
fi

echo "== 4/7 Patch the converter for alloma's tokenizer =="
# alloma keeps the Llama-3 pre-tokenizer regex but replaced part of the vocabulary with Uzbek
# tokens. llama.cpp identifies tokenizers by a hash of the vocab, so it would refuse to convert.
# We tell it to treat an unknown hash as the standard Llama-3 tokenizer ("llama-bpe").
# This only affects this private llama.cpp copy in $WORK.
python3 - "$LLAMA" <<'EOF'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
targets = [root / "conversion" / "base.py", root / "convert_hf_to_gguf.py"]
pat = re.compile(r'^(\s*)raise NotImplementedError\("BPE pre-tokenizer was not recognized[^\n]*$', re.M)
for f in targets:
    if not f.exists():
        continue
    s = f.read_text()
    if "patched-for-alloma" in s:
        print(f"already patched: {f}"); break
    s2, n = pat.subn(r'\1res = "llama-bpe"  # patched-for-alloma: Llama-3 regex with custom Uzbek vocab', s, count=1)
    if n:
        f.write_text(s2); print(f"patched: {f}"); break
else:
    print("WARNING: patch point not found - conversion may fail with 'pre-tokenizer was not recognized'")
EOF

echo "== 5/7 Download $MODEL_ID =="
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q numpy safetensors   # converter deps (torch + transformers are already in the venv)
hf download "$MODEL_ID" --local-dir "$HF_DIR"

echo "== 6/7 Convert to GGUF (f16), then quantize to $QUANT =="
python "$LLAMA/convert_hf_to_gguf.py" "$HF_DIR" --outtype f16 --outfile "$F16"
if [ "${KEEP_HF:-0}" != "1" ]; then
  rm -rf "$HF_DIR"; echo "deleted HF weights (set KEEP_HF=1 to keep them)"
fi
"$LLAMA/build/bin/llama-quantize" "$F16" "$OUT" "$QUANT"
rm -f "$F16"

echo "== 7/7 Register in kronk (hard link, no extra space) =="
mkdir -p "$KRONK_DIR"
ln -f "$OUT" "$KRONK_DIR/"
ls -lh "$OUT"

echo
echo "Done: $OUT"
echo "Next:"
echo "  kronk model index --local && kronk model list --local"
echo "  (restart 'kronk server start' in tmux if the new model does not show up in /v1/models)"
echo "  python llm_baseline.py --model <alloma-id-from-list> --apst"
