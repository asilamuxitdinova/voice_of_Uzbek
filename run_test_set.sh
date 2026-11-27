#!/usr/bin/env bash
# Run on your LAPTOP (or directly on the server - it detects that and skips the upload).
# Sends only the test recordings (test1.mp3 ... test20.mp3 and any matching
# test1.txt references) to the server, runs the ASR models on just those files, and copies the
# results back into results/ here. Nothing needs to be pushed from the server.
#
# Usage:
#   bash run_test_set.sh                                   # default models (below)
#   bash run_test_set.sh ai-sage/GigaAM-Multilingual       # only the models you name
#   DRY_RUN=1 bash run_test_set.sh                         # print what would happen, do nothing
# Settings (environment variables):
#   SERVER=kronk            SSH host (from ~/.ssh/config)
#   REMOTE_DIR=voice_of_Uzbek   repo folder in the server's home directory
#   SET=test_set            folder name for these recordings on the server
#   PATTERN='test[0-9]*'    which files in test_audios/ to send
#   EXTRA='--beams 1'       extra options for asr_baseline.py
#   LOCAL=1                 force on-server mode (no ssh), if hostname detection doesn't match
#
# Before running: push your latest code, so the server's `git pull` gets it.
set -euo pipefail
cd "$(dirname "$0")"

SERVER="${SERVER:-kronk}"
REMOTE_DIR="${REMOTE_DIR:-voice_of_Uzbek}"
SET="${SET:-test_set}"
PATTERN="${PATTERN:-test[0-9]*}"
EXTRA="${EXTRA:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(
    ai-sage/GigaAM-Multilingual
    Abduqayum/whisper-uzbek-medium-callcenter
    navai-uz/whisper-medium-uzbek
    Kotib/uzbek_stt_v1
  )
fi

run() {  # print the command; execute it unless DRY_RUN=1
  printf '+ %s\n' "$*"
  [ "$DRY_RUN" = "1" ] || "$@"
}

# 1. Collect the files
shopt -s nullglob
files=(test_audios/$PATTERN.*)
shopt -u nullglob
if [ "${#files[@]}" -eq 0 ]; then
  echo "No files matching test_audios/$PATTERN.* - nothing to send."
  exit 1
fi
n_audio=0; n_txt=0
for f in "${files[@]}"; do
  case "${f##*.}" in txt) n_txt=$((n_txt + 1)) ;; *) n_audio=$((n_audio + 1)) ;; esac
done
echo "== Found $n_audio recordings and $n_txt reference .txt files"

# Running ON the server itself? Then there is nothing to upload: copy locally and run here.
# Detected when this machine's hostname equals $SERVER (e.g. "kronk"), or forced with LOCAL=1.
ON_SERVER=0
if [ "${LOCAL:-0}" = "1" ] || [ "$(hostname -s 2>/dev/null || hostname)" = "$SERVER" ]; then
  ON_SERVER=1
fi

# The python commands that run the models (same in both modes)
models_cmd=""
for m in "${MODELS[@]}"; do
  models_cmd+="; echo; echo '=================== $m'"
  models_cmd+="; python asr_baseline.py --model $(printf %q "$m") --audio-dir $SET $EXTRA || echo 'FAILED: $m'"
done

if [ "$ON_SERVER" = "1" ]; then
  echo "== Running on the server itself ($(hostname -s 2>/dev/null)) - no upload needed"
  run mkdir -p "$SET"
  run cp -u "${files[@]}" "$SET/"
  local_cmd="set -e"
  [ -n "${VIRTUAL_ENV:-}" ] || local_cmd+="; source .venv/bin/activate"
  echo "== Running ${#MODELS[@]} model(s) on $SET/"
  run bash -c "$local_cmd$models_cmd"
else
  # 2. Warn if the server would not get the latest code
  if [ -n "$(git status --porcelain -- '*.py' '*.sh' requirements.txt 2>/dev/null)" ]; then
    echo "WARNING: you have uncommitted code changes - the server won't have them."
  fi
  git fetch -q 2>/dev/null || true
  ahead=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)
  if [ "$ahead" != "0" ]; then
    echo "WARNING: $ahead local commit(s) not pushed - run 'git push' first, or the server runs old code."
  fi

  # 3. Upload (rsync only sends new/changed files; scp as fallback)
  echo "== Uploading to $SERVER:~/$REMOTE_DIR/$SET/"
  run ssh "$SERVER" "mkdir -p ~/$REMOTE_DIR/$SET"
  if command -v rsync >/dev/null 2>&1 && { [ "$DRY_RUN" = "1" ] || ssh "$SERVER" "command -v rsync" >/dev/null 2>&1; }; then
    run rsync -a --info=progress2 "${files[@]}" "$SERVER:$REMOTE_DIR/$SET/"
  else
    run scp "${files[@]}" "$SERVER:$REMOTE_DIR/$SET/"
  fi

  # 4. Run the models on the server, only on this folder
  remote="set -e; cd ~/$REMOTE_DIR"
  remote+="; git pull --ff-only || echo 'WARNING: git pull failed - running the code that is already there'"
  remote+="; source .venv/bin/activate"
  echo "== Running ${#MODELS[@]} model(s) on the server"
  run ssh "$SERVER" "$remote$models_cmd"

  # 5. Copy the results back
  echo "== Copying results back to results/"
  mkdir -p results
  run scp "$SERVER:$REMOTE_DIR/results/asr_*_${SET}.txt" results/
fi

echo
echo "== Summary"
if [ "$DRY_RUN" != "1" ]; then
  grep -H "^WER\|^CER\|^speed\|^files" results/asr_*_"${SET}".txt || true
fi
echo "Open results/asr_*_${SET}.txt and fill in MEANING OK for each file."
if [ "$n_txt" -eq 0 ]; then
  echo "Tip: add test1.txt ... test20.txt (what was actually said) next to the mp3 files to get WER."
fi
