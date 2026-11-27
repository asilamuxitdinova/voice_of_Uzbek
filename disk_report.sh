#!/usr/bin/env bash
# READ-ONLY disk usage report. Deletes nothing.
# Usage (on the server):  bash disk_report.sh | tee results/disk_report.txt
# Then paste the output to Claude (or push results/disk_report.txt) to decide what is safe to remove.
# Some system folders need sudo to measure; you may be asked for your password once.
set -uo pipefail

section() { printf '\n==================== %s\n' "$1"; }
sz() { du -sh "$@" 2>/dev/null | sort -rh; }

section "Disk overview"
df -h / "$HOME" 2>/dev/null | uniq

section "Biggest top-level folders on the disk (whole system, same disk only)"
sudo du -xh --max-depth=1 / 2>/dev/null | sort -rh | head -15

section "Biggest folders in your home directory"
du -xh --max-depth=1 "$HOME" 2>/dev/null | sort -rh | head -20

section "Biggest folders two levels deep in home"
du -xh --max-depth=2 "$HOME" 2>/dev/null | sort -rh | head -25

section "Model files (*.gguf, *.safetensors, *.bin, *.pt, *.ckpt, *.onnx > 200 MB)"
find "$HOME" -xdev -type f \( -name '*.gguf' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.onnx' -o -name '*.pth' \) -size +200M -printf '%s\t%n links\t%p\n' 2>/dev/null \
  | sort -rn | awk -F'\t' '{printf "%8.1f GB  %s  %s\n", $1/1e9, $2, $3}' | head -40
echo "(\"2 links\" = hard-linked into kronk: deleting one name does NOT free space until both are gone)"

section "Hugging Face cache (downloaded models and datasets)"
if command -v hf >/dev/null 2>&1; then
  hf cache scan 2>/dev/null | head -40 || true
fi
sz "$HOME/.cache/huggingface"/* "$HOME/.cache/huggingface/hub"/* 2>/dev/null | head -25

section "Other caches"
sz "$HOME/.cache"/* | head -15
command -v pip >/dev/null 2>&1 && echo "pip cache: $(pip cache dir 2>/dev/null) ($(du -sh "$(pip cache dir 2>/dev/null)" 2>/dev/null | cut -f1))"
[ -d "$HOME/.cache/uv" ] && echo "uv cache: $(du -sh "$HOME/.cache/uv" 2>/dev/null | cut -f1)"
[ -d "$HOME/.local/share/Trash" ] && echo "Trash: $(du -sh "$HOME/.local/share/Trash" 2>/dev/null | cut -f1)"

section "Python environments (.venv, conda)"
find "$HOME" -xdev -maxdepth 4 -type d \( -name '.venv' -o -name 'venv' -o -name '.venv-*' -o -name 'miniconda3' -o -name 'anaconda3' \) 2>/dev/null \
  | while read -r d; do du -sh "$d" 2>/dev/null; done | sort -rh

section "Build folders and temporary work (llama.cpp builds, convert_work, Go/Rust/node caches)"
sz "$HOME/convert_work" "$HOME"/*/llama.cpp "$HOME/go" "$HOME/.cargo" "$HOME/.rustup" "$HOME/.npm" "$HOME"/*/node_modules 2>/dev/null

section "kronk models"
sz "$HOME/.kronk" "$HOME/.kronk/models"/*/* 2>/dev/null | head -20

section "This project"
sz "$HOME/voice_of_Uzbek"/* "$HOME/voice_of_Uzbek"/.venv* 2>/dev/null | head -15

section "System: Docker, apt cache, logs, snaps (needs sudo)"
command -v docker >/dev/null 2>&1 && sudo docker system df 2>/dev/null
sz /var/cache/apt /var/log /var/lib/docker /var/lib/snapd /tmp 2>/dev/null
sudo journalctl --disk-usage 2>/dev/null

section "Big single files anywhere on the disk (> 1 GB)"
sudo find / -xdev -type f -size +1G -printf '%s\t%p\n' 2>/dev/null | sort -rn | awk -F'\t' '{printf "%8.1f GB  %s\n", $1/1e9, $2}' | head -30

echo
echo "Done. Nothing was deleted."
