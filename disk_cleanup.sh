#!/usr/bin/env bash
# Frees disk space WITHOUT losing anything you can't get back:
#   1. pip / uv / Go build caches  (download leftovers; refill automatically when needed)
#   2. models stored twice (a separate copy in ~/.kronk/models identical to the one in ~/models)
#      -> the kronk copy is replaced by a hard link: same file, same name, kronk keeps working,
#         but it is stored only once. Files are compared byte-for-byte first.
# It does NOT touch: models you use, other projects' files, virtual environments, the HF cache.
#
# Usage (on the server):
#   bash disk_cleanup.sh          # DRY RUN: shows what it would do and how much it would free
#   bash disk_cleanup.sh --yes    # actually do it
set -uo pipefail

APPLY=0
[ "${1:-}" = "--yes" ] && APPLY=1
[ "$APPLY" = "1" ] && echo "== APPLYING changes" || echo "== DRY RUN (nothing is changed). Run with --yes to apply."

size_of() { du -sb "$1" 2>/dev/null | cut -f1; }
# Bytes that deleting would really free: files with only one name. uv hard-links its cache into
# virtual environments, so those files have 2+ names and deleting the cache copy frees nothing.
freeable_of() { find "$1" -type f -links 1 -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
free_bytes() { df -B1 --output=avail "$HOME" | tail -1 | tr -dc '0-9'; }
human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0} bytes"; }
total=0
before=$(free_bytes)
echo "Free before: $(human "$before")"

echo
echo "== 1. Caches"
clean_cache() {  # name, folder, command to clean it
  local name="$1" dir="$2" cmd="$3"
  [ -d "$dir" ] || { echo "  $name: not present"; return; }
  local b f; b=$(size_of "$dir"); f=$(freeable_of "$dir"); total=$((total + f))
  echo "  $name: $(human "$b") on disk, $(human "$f") really freeable  ($dir)"
  [ "$f" -lt "$((b / 2))" ] && echo "    (most of it is shared with virtual environments via hard links - deleting won't free that part)"
  if [ "$APPLY" = "1" ]; then
    eval "$cmd" >/dev/null 2>&1 || rm -rf "${dir:?}"/*
  fi
}
clean_cache "uv cache"       "$HOME/.cache/uv"       "uv cache clean"
clean_cache "pip cache"      "$HOME/.cache/pip"      "pip cache purge"
clean_cache "Go build cache" "$HOME/.cache/go-build" "go clean -cache"

echo
echo "== 2. Models stored twice (~/models vs ~/.kronk/models)"
found=0
while IFS= read -r -d '' k; do
  name=$(basename "$k")
  m="$HOME/models/$name"
  [ -f "$m" ] || continue
  if [ "$(stat -c %i "$k")" = "$(stat -c %i "$m")" ]; then
    continue  # already the same file (hard link)
  fi
  found=1
  b=$(stat -c %s "$k")
  echo "  $name: two separate copies ($(human "$b") wasted)"
  echo "    comparing byte-for-byte (takes a moment)..."
  if cmp -s "$k" "$m"; then
    total=$((total + b))
    echo "    identical -> kronk copy will become a hard link to ~/models/$name"
    if [ "$APPLY" = "1" ]; then
      ln -f "$m" "$k" && echo "    done"
    fi
  else
    echo "    NOT identical - leaving both alone"
  fi
done < <(find "$HOME/.kronk/models" -type f -name '*.gguf' -print0 2>/dev/null)
[ "$found" = "0" ] && echo "  none"

echo
if [ "$APPLY" = "1" ]; then
  sync; after=$(free_bytes)
  echo "== Estimated $(human "$total"); REAL change in free space: $(human $((after - before)))"
  echo "Free after:  $(human "$after")"
  if [ $((after - before)) -lt $((total / 2)) ]; then
    echo "Less than expected? A running program may still hold a deleted file open (e.g. kronk with a model)."
    echo "Check: sudo lsof +L1 | grep -i gguf   - restart that program to release the space."
  fi
else
  echo "== Would free about $(human "$total"). Run: bash disk_cleanup.sh --yes"
fi
