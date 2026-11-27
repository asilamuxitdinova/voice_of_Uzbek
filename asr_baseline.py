"""Step 1.1 - ASR baseline: how well does stock Whisper understand Uzbek?

Measures Word Error Rate (WER) on the Google FLEURS Uzbek test set.

Usage:
    python asr_baseline.py                       # 100 FLEURS samples, whisper-large-v3-turbo
    python asr_baseline.py --n 300               # more samples = more reliable number
    python asr_baseline.py --model openai/whisper-small
    python asr_baseline.py --audio question.m4a      # transcribe your own file (any format)
"""
import argparse
import os
import re
import time

import torch
import jiwer
from transformers import pipeline


def load_audio(path: str, sr: int = 16000):
    """Load any audio file as mono float32 at `sr` Hz.

    wav/flac/ogg/mp3 are read directly; anything else (phone .m4a/.aac/.opus, video files)
    is decoded with ffmpeg. Returns (array, sr).
    """
    import subprocess

    import numpy as np

    if not os.path.isfile(path):
        raise SystemExit(f"Audio file not found: {os.path.abspath(path)}\n"
                         f"Copy it to the server first (scp ...), or pass the full path.")
    try:
        import librosa
        array, _ = librosa.load(path, sr=sr, mono=True)
        return array, sr
    except Exception:
        pass  # format not supported by libsndfile -> fall back to ffmpeg
    try:
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
             "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
            check=True, capture_output=True,
        ).stdout
    except FileNotFoundError:
        raise SystemExit("ffmpeg is not installed: sudo apt install -y ffmpeg")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffmpeg could not decode {path}:\n{e.stderr.decode(errors='replace')}")
    return np.frombuffer(raw, dtype=np.float32).copy(), sr


def norm(text: str) -> str:
    """Lowercase, unify the oʻ/gʻ apostrophes, drop punctuation."""
    text = text.lower()
    for a in ("ʻ", "’", "‘", "`", "ʼ", "´"):
        text = text.replace(a, "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_fleurs(split: str, n: int, lang: str = "uz_uz"):
    """Download FLEURS files directly from the Hub (no `datasets` library needed).

    Returns a list of (wav_path, transcription). Rows in the TSV are grouped by
    sentence (several speakers read the same one), so we take every k-th row to
    get n samples spread over as many different sentences as possible.
    """
    import csv
    import tarfile
    from huggingface_hub import hf_hub_download

    repo = dict(repo_id="google/fleurs", repo_type="dataset")
    tsv = hf_hub_download(filename=f"data/{lang}/{split}.tsv", **repo)
    with open(tsv, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 4]
    step = max(1, len(rows) // n)
    rows = rows[::step][:n]

    out_dir = os.path.join("data", f"fleurs_{lang}_{split}")
    os.makedirs(out_dir, exist_ok=True)
    wanted = {r[1] for r in rows}
    missing = {w for w in wanted if not os.path.exists(os.path.join(out_dir, w))}
    if missing:
        tar_path = hf_hub_download(filename=f"data/{lang}/audio/{split}.tar.gz", **repo)
        with tarfile.open(tar_path, "r:gz") as tar:
            for m in tar:
                name = os.path.basename(m.name)
                if m.isfile() and name in missing:
                    with tar.extractfile(m) as src, open(os.path.join(out_dir, name), "wb") as dst:
                        dst.write(src.read())

    samples = [(os.path.join(out_dir, r[1]), r[3]) for r in rows]
    samples = [s for s in samples if os.path.exists(s[0])]
    if not samples:
        raise SystemExit("No FLEURS audio found after download - check the error above.")
    return samples


def load_asr(model_name: str):
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        dtype=torch.float16,
        device="cuda:0",
    )


def transcribe(asr, array, sr) -> str:
    out = asr(
        {"raw": array, "sampling_rate": sr},
        generate_kwargs={"language": "uzbek", "task": "transcribe"},
    )
    return out["text"].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--n", type=int, default=100, help="number of FLEURS test samples")
    p.add_argument("--audio", help="transcribe this audio file instead of running the benchmark")
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    asr = load_asr(args.model)

    if args.audio:
        array, sr = load_audio(args.audio)
        print("TEXT:", transcribe(asr, array, sr))
        return

    print("Loading FLEURS uz_uz test split (first run downloads ~520 MB) ...")
    ds = load_fleurs("test", args.n)

    refs, hyps = [], []
    t0 = time.time()
    for i, (wav_path, ref) in enumerate(ds, 1):
        array, sr = load_audio(wav_path)
        hyps.append(norm(transcribe(asr, array, sr)))
        refs.append(norm(ref))
        if i % 20 == 0:
            print(f"  {i}/{len(ds)} done")
    elapsed = time.time() - t0

    wer = jiwer.wer(refs, hyps) * 100
    cer = jiwer.cer(refs, hyps) * 100

    os.makedirs("results", exist_ok=True)
    out_path = f"results/asr_{args.model.replace('/', '_')}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"model: {args.model}\nsamples: {len(ds)}\nWER: {wer:.1f}%\nCER: {cer:.1f}%\n")
        f.write(f"avg seconds per sample: {elapsed / len(ds):.2f}\n\n")
        for r, h in zip(refs, hyps):
            f.write(f"REF: {r}\nHYP: {h}\n\n")

    for r, h in list(zip(refs, hyps))[:5]:
        print(f"\nREF: {r}\nHYP: {h}")
    print(f"\nWER: {wer:.1f}%   CER: {cer:.1f}%   ({elapsed / len(ds):.2f}s per sample)")
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
