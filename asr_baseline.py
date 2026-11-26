"""Step 1.1 - ASR baseline: how well does stock Whisper understand Uzbek?

Measures Word Error Rate (WER) on the Google FLEURS Uzbek test set.

Usage:
    python asr_baseline.py                       # 100 FLEURS samples, whisper-large-v3-turbo
    python asr_baseline.py --n 300               # more samples = more reliable number
    python asr_baseline.py --model openai/whisper-small
    python asr_baseline.py --audio my_recording.wav   # just transcribe your own file
"""
import argparse
import os
import re
import time

import torch
import jiwer
from transformers import pipeline


def norm(text: str) -> str:
    """Lowercase, unify the oʻ/gʻ apostrophes, drop punctuation."""
    text = text.lower()
    for a in ("ʻ", "’", "‘", "`", "ʼ", "´"):
        text = text.replace(a, "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_asr(model_name: str):
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16,
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
        import librosa
        array, sr = librosa.load(args.audio, sr=16000, mono=True)
        print("TEXT:", transcribe(asr, array, sr))
        return

    from datasets import load_dataset
    print("Loading FLEURS uz_uz test split (first run downloads it) ...")
    ds = load_dataset("google/fleurs", "uz_uz", split="test", trust_remote_code=True)
    ds = ds.select(range(min(args.n, len(ds))))

    refs, hyps = [], []
    t0 = time.time()
    for i, ex in enumerate(ds, 1):
        a = ex["audio"]
        hyps.append(norm(transcribe(asr, a["array"], a["sampling_rate"])))
        refs.append(norm(ex["transcription"]))
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
