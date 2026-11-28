"""Phase 2 - build the GigaAM fine-tuning dataset from open Uzbek speech corpora.

For each source it downloads ONE parquet file at a time, converts every clip to 16 kHz mono FLAC,
cleans the transcript into GigaAM's alphabet, appends it to the TSV manifests, and deletes the
download before fetching the next file. Peak extra disk use is one parquet file (~0.5 GB).

Re-running continues where it stopped (finished parquet files are remembered in state.json).
It stops by itself when free disk space drops below --min-free-gb.

    python prepare_data.py                          # all sources, default hour caps
    python prepare_data.py --sources usc news       # only some sources
    python prepare_data.py --max-hours uzbekvoice=100 news=50
    python prepare_data.py --max-rows 200 --out data/test_prep   # quick test run

Output (default data/gigaam_uz/):
    audio/<source>/<id>.flac
    train.tsv  val.tsv  test_usc.tsv     (columns: path, duration, transcription_raw, source)

Kept OUT of training on purpose: FLEURS (our test set), test_audios/ and test_set/ (own recordings).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf

from uz_text import numbers_to_words

SR = 16000

# name -> (HF dataset repo, text column, {parquet split prefix: our split}, license)
SOURCES = {
    "usc": ("murodbek/uzbek-speech-corpus", "sentence",
            {"train": "train", "validation": "val", "test": "test_usc"}, "CC-BY-4.0 (ISSAI USC, MIT)"),
    "news": ("islomov/news_youtube_uzbek_speech_dataset", "text", {"train": "train"}, "Apache-2.0"),
    "it": ("islomov/it_youtube_uzbek_speech_dataset", "text", {"train": "train"}, "Apache-2.0"),
    "podcasts": ("islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset", "text",
                 {"train": "train"}, "Apache-2.0"),
    "uzbekvoice": ("DavronSherbaev/uzbekvoice-filtered", "text",
                   {"train": "train", "validate": "val"}, "Apache-2.0"),
}
# default caps keep the whole set around 35-40 GB of FLAC (~550-600 h)
DEFAULT_MAX_HOURS = {"usc": 200, "news": 200, "it": 200, "podcasts": 200, "uzbekvoice": 150}

ALLOWED = re.compile(r"^[a-z' ]+$")  # GigaAM's Latin letters + apostrophe + space
APOSTROPHES = ("ʻ", "’", "‘", "`", "ʼ", "´", "ʹ")
COLUMNS = ["path", "duration", "transcription_raw", "source"]


def clean_text(text: str):
    """Transcript -> GigaAM training target, or None if the row should be skipped."""
    if not text:
        return None
    t = text.lower()
    for a in APOSTROPHES:
        t = t.replace(a, "'")
    if re.search(r"[Ѐ-ӿ]", t):  # Cyrillic (mostly Russian quotes) - skip the clip
        return None
    t = numbers_to_words(t)
    t = re.sub(r"[^\w\s']", " ", t)          # punctuation -> space
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)  # apostrophes only inside words
    t = re.sub(r"\s+", " ", t).strip()
    if not t or not ALLOWED.match(t):        # leftover digits, foreign letters, ...
        return None
    return t


def decode_audio(blob: dict):
    """Parquet audio struct {bytes, path} -> mono float32 at 16 kHz, or None."""
    data = blob.get("bytes") if isinstance(blob, dict) else None
    if not data:
        return None
    try:
        array, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        array = array.mean(axis=1)
    except Exception:  # format libsndfile can't read -> ffmpeg
        try:
            raw = subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
                 "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                input=data, check=True, capture_output=True).stdout
        except Exception:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()
    if sr != SR:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=SR)
    return array.astype(np.float32)


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def is_val(source: str, clip_id: str) -> bool:
    """Deterministic ~1% validation hold-out for sources without their own split."""
    return int(hashlib.md5(f"{source}/{clip_id}".encode()).hexdigest(), 16) % 100 == 0


class Writer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.files, self.writers = {}, {}

    def write(self, split, row):
        if split not in self.writers:
            path = os.path.join(self.out_dir, f"{split}.tsv")
            new = not os.path.exists(path)
            self.files[split] = open(path, "a", encoding="utf-8", newline="")
            self.writers[split] = csv.writer(self.files[split], delimiter="\t", lineterminator="\n")
            if new:
                self.writers[split].writerow(COLUMNS)
        self.writers[split].writerow(row)

    def flush(self):
        for f in self.files.values():
            f.flush()

    def close(self):
        for f in self.files.values():
            f.close()


def process_source(name, args, state, writer):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo, text_col, split_map, _ = SOURCES[name]
    cap_h = args.max_hours.get(name, DEFAULT_MAX_HOURS[name])
    st = state.setdefault(name, {"done_files": [], "hours": {}, "kept": 0, "skipped": 0})
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.endswith(".parquet"))
    audio_dir = os.path.join(args.out, "audio", name)
    os.makedirs(audio_dir, exist_ok=True)
    dl_dir = os.path.join(args.out, "_download")

    for fname in files:
        split_key = os.path.basename(fname).split("-")[0]  # "train-00003-of-00034.parquet" -> "train"
        split = split_map.get(split_key)
        if split is None or fname in st["done_files"]:
            continue
        if split == "train" and st["hours"].get("train", 0) >= cap_h:
            print(f"[{name}] reached {cap_h} h of training audio - skipping the rest")
            break
        if free_gb(args.out) < args.min_free_gb:
            raise SystemExit(f"Stopping: only {free_gb(args.out):.1f} GB free (< --min-free-gb "
                             f"{args.min_free_gb}). Re-run later to continue where it stopped.")

        t0 = time.time()
        print(f"[{name}] downloading {fname} ...", flush=True)
        local = hf_hub_download(repo, fname, repo_type="dataset", local_dir=dl_dir)
        kept = skipped = 0
        reasons = {}
        pf = pq.ParquetFile(local)
        audio_col = next(c for c in pf.schema_arrow.names if c in ("audio", "path"))
        id_col = "id" if "id" in pf.schema_arrow.names else None
        stop = False
        for batch in pf.iter_batches(batch_size=256, columns=[c for c in (audio_col, text_col, id_col) if c]):
            rows = batch.to_pylist()
            for i, r in enumerate(rows):
                text = clean_text(r.get(text_col) or "")
                clip_id = str(r.get(id_col) if id_col else f"{os.path.basename(fname)}_{kept + skipped}")
                clip_id = re.sub(r"[^\w.-]", "_", clip_id)[:120]
                array = decode_audio(r.get(audio_col)) if text else None
                dur = len(array) / SR if array is not None else 0
                reason = ("text" if text is None else "audio" if array is None
                          else "short" if dur < args.min_dur else "long" if dur > args.max_dur
                          else "fast" if len(text) / dur > 25 else None)  # >25 chars/s = mismatch
                if reason:
                    skipped += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                this_split = "val" if (split == "train" and is_val(name, clip_id)) else split
                rel = os.path.join("audio", name, clip_id + ".flac")
                sf.write(os.path.join(args.out, rel), array, SR, format="FLAC", subtype="PCM_16")
                writer.write(this_split, [rel, f"{dur:.3f}", text, name])
                st["hours"][this_split] = st["hours"].get(this_split, 0) + dur / 3600
                kept += 1
                if args.max_rows and kept + skipped >= args.max_rows:
                    stop = True
                    break
            if stop:
                break
        writer.flush()
        st["kept"] += kept
        st["skipped"] += skipped
        if not args.max_rows:
            st["done_files"].append(fname)
        save_state(args.out, state)
        os.remove(local)  # free the download before the next file
        why = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
        print(f"[{name}] {fname}: kept {kept}, skipped {skipped}" + (f" ({why})" if why else "") +
              f" in {time.time() - t0:.0f}s | "
              f"train {st['hours'].get('train', 0):.1f} h | free {free_gb(args.out):.1f} GB", flush=True)
        if args.max_rows:
            break
    shutil.rmtree(dl_dir, ignore_errors=True)


def save_state(out, state):
    with open(os.path.join(out, "state.json"), "w") as f:
        json.dump(state, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/gigaam_uz")
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--max-hours", nargs="*", default=[], metavar="SOURCE=HOURS",
                   help=f"training-hour cap per source (defaults: {DEFAULT_MAX_HOURS})")
    p.add_argument("--min-dur", type=float, default=0.5)
    p.add_argument("--max-dur", type=float, default=30.0,
                   help="most YouTube clips are 20-30 s; train_gigaam.py uses the same 30 s limit")
    p.add_argument("--min-free-gb", type=float, default=25.0,
                   help="stop when free disk falls below this (leave room for checkpoints)")
    p.add_argument("--max-rows", type=int, default=0, help="test mode: only N rows per source")
    args = p.parse_args()
    args.max_hours = {k: float(v) for k, v in (x.split("=") for x in args.max_hours)}

    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    writer = Writer(args.out)
    try:
        for name in args.sources:
            print(f"\n===== {name}: {SOURCES[name][0]} (license {SOURCES[name][3]})")
            process_source(name, args, state, writer)
    finally:
        writer.close()
        save_state(args.out, state)

    print("\n===== Summary (hours)")
    for name, st in state.items():
        hrs = ", ".join(f"{k} {v:.1f}" for k, v in sorted(st["hours"].items()))
        print(f"  {name:11s} {hrs}  | kept {st['kept']}, skipped {st['skipped']}")
    print(f"Manifests: {args.out}/train.tsv, val.tsv, test_usc.tsv")


if __name__ == "__main__":
    main()
