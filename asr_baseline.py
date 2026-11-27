"""Step 1.1 - ASR evaluation: how well does a Whisper model understand Uzbek?

Three modes:
    python asr_baseline.py --model M                    # WER on 100 FLEURS Uzbek test clips
    python asr_baseline.py --model M --audio q.m4a      # transcribe one file (any format)
    python asr_baseline.py --model M --audio-dir test_audios
        # transcribe every audio file in the folder; if "name.txt" (what was actually said)
        # sits next to "name.m4a", WER is computed too. Model is loaded only once.

Options:
    --beams 1     greedy decoding for Whisper models (faster). Default is 5 beams, which is also
                  what transformers uses when nothing is set, so all earlier results used 5
    --n 300       more FLEURS clips = more reliable number

Two scores are reported: "basic" (the original) and "fair" (numbers as words,
Cyrillic->Latin), see uz_text.py. Use the fair score from now on.
"""
import argparse
import os
import re
import time

import torch
import jiwer
from transformers import pipeline

from uz_text import norm_basic, norm_fair


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


norm = norm_basic  # backwards compatibility

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".amr", ".webm", ".mp4"}


def scores(refs, hyps):
    """(basic WER, fair WER, fair CER) in %, skipping empty references."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if norm_basic(r)]
    if not pairs:
        return None
    rb = [norm_basic(r) for r, _ in pairs]
    hb = [norm_basic(h) for _, h in pairs]
    rf = [norm_fair(r) for r, _ in pairs]
    hf = [norm_fair(h) for _, h in pairs]
    return jiwer.wer(rb, hb) * 100, jiwer.wer(rf, hf) * 100, jiwer.cer(rf, hf) * 100


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


# GigaAM-Multilingual (Conformer CTC, MIT). Pick the size with "@": the default is large_ctc (600M).
#   ai-sage/GigaAM-Multilingual            -> large_ctc
#   ai-sage/GigaAM-Multilingual@ctc        -> 220M (faster)
# It runs the repo's own modeling_gigaam.py (trust_remote_code), so the code is pinned to the
# commits that were reviewed on 2026-11-27. To use newer code, review it and update the hashes.
GIGAAM_REVISIONS = {
    "large_ctc": "3905cd51c3ed4e88c8edf33f3302969ba480a327",
    "ctc": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
}


def split_long_audio(array, sr: int, max_s: float = 24.0, min_s: float = 12.0):
    """Split audio into pieces <= max_s, cutting at the quietest 0.1 s window between min_s and max_s."""
    import numpy as np

    pieces, start, n = [], 0, len(array)
    max_len, min_len, win = int(max_s * sr), int(min_s * sr), int(0.1 * sr)
    while n - start > max_len:
        region = array[start + min_len: start + max_len]
        frames = len(region) // win
        energy = (region[: frames * win].reshape(frames, win) ** 2).mean(axis=1)
        cut = start + min_len + int(np.argmin(energy)) * win + win // 2
        pieces.append(array[start:cut])
        start = cut
    pieces.append(array[start:])
    return pieces


class GigaAMASR:
    """Wraps GigaAM so it can be used like the Whisper pipeline (array in, text out)."""

    def __init__(self, model_name: str):
        from transformers import AutoModel

        repo, _, variant = model_name.partition("@")
        variant = variant or "large_ctc"
        revision = GIGAAM_REVISIONS.get(variant, variant)
        if variant not in GIGAAM_REVISIONS:
            print(f"WARNING: '{variant}' is not a reviewed revision - its remote code was not checked.")
        # transformers scans the remote code for imports and refuses to load it if any is missing.
        # pyannote is only referenced for type hints and the long-audio mode (which we replace with
        # split_long_audio), so skip it in that check instead of installing a large dependency.
        from unittest.mock import patch

        import transformers.dynamic_module_utils as dmu

        original_get_imports = dmu.get_imports

        def get_imports_without_optional(filename):
            return [m for m in original_get_imports(filename) if m != "pyannote"]

        try:
            with patch("transformers.dynamic_module_utils.get_imports", get_imports_without_optional):
                self.model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
        except ImportError as e:
            raise SystemExit(f"{e}\nGigaAM needs: pip install hydra-core omegaconf sentencepiece")
        self.model = self.model.to("cuda").eval()

    def __call__(self, array, sr: int) -> str:
        import tempfile

        import soundfile as sf

        texts = []
        for piece in split_long_audio(array, sr):  # GigaAM refuses audio longer than 25 s
            if len(piece) < sr // 10:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, piece, sr, subtype="PCM_16")
                result = self.model.transcribe(tmp.name)
            texts.append(getattr(result, "text", str(result)).strip())
        return " ".join(t for t in texts if t)


def is_gigaam(model_name: str) -> bool:
    return "gigaam" in model_name.lower()


def load_asr(model_name: str):
    if is_gigaam(model_name):
        return GigaAMASR(model_name)
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        dtype=torch.float16,
        device="cuda:0",
        chunk_length_s=30,  # recordings longer than 30 s are split into chunks
    )


DEFAULT_BEAMS = 5  # what transformers uses by default; all Whisper results before 2026-11-27 used it


def transcribe(asr, array, sr, beams: int = DEFAULT_BEAMS) -> str:
    if isinstance(asr, GigaAMASR):
        return asr(array, sr)  # CTC greedy decoding; --beams does not apply
    # Always pass num_beams: transformers' ASR pipeline silently defaults to 5 beams.
    gen = {"language": "uzbek", "task": "transcribe", "num_beams": beams}
    out = asr({"raw": array, "sampling_rate": sr}, generate_kwargs=gen)
    return out["text"].strip()


def out_name(model: str, beams: int, suffix: str = "") -> str:
    name = f"results/asr_{re.sub(r'[/@]', '_', model)}"
    if beams != DEFAULT_BEAMS and not is_gigaam(model):
        name += f"_beam{beams}"
    return name + (f"_{suffix}" if suffix else "") + ".txt"


def run_fleurs(asr, args):
    print("Loading FLEURS uz_uz test split (first run downloads ~520 MB) ...")
    ds = load_fleurs("test", args.n)
    refs, hyps = [], []
    t0 = time.time()
    for i, (wav_path, ref) in enumerate(ds, 1):
        array, sr = load_audio(wav_path)
        hyps.append(transcribe(asr, array, sr, args.beams))
        refs.append(ref)
        if i % 20 == 0:
            print(f"  {i}/{len(ds)} done")
    elapsed = time.time() - t0
    wer_b, wer_f, cer_f = scores(refs, hyps)

    out_path = out_name(args.model, args.beams)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"model: {args.model}\nbeams: {args.beams}\nsamples: {len(ds)}\n")
        f.write(f"WER: {wer_b:.1f}%\nWER (fair): {wer_f:.1f}%\nCER (fair): {cer_f:.1f}%\n")
        f.write(f"avg seconds per sample: {elapsed / len(ds):.2f}\n\n")
        for r, h in zip(refs, hyps):
            f.write(f"REF: {norm_basic(r)}\nHYP: {norm_basic(h)}\n\n")

    for r, h in list(zip(refs, hyps))[:5]:
        print(f"\nREF: {norm_basic(r)}\nHYP: {norm_basic(h)}")
    print(f"\nWER: {wer_b:.1f}%   WER (fair): {wer_f:.1f}%   CER (fair): {cer_f:.1f}%   "
          f"({elapsed / len(ds):.2f}s per sample)")
    print(f"Full results saved to {out_path}")


def run_dir(asr, args):
    folder = args.audio_dir.rstrip("/")
    files = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"No audio files in {folder} (looked for {', '.join(sorted(AUDIO_EXTS))})")

    rows, refs, hyps = [], [], []
    total_audio = total_time = 0.0
    for name in files:
        path = os.path.join(folder, name)
        txt = os.path.join(folder, os.path.splitext(name)[0] + ".txt")
        ref = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else None
        array, sr = load_audio(path)
        t0 = time.time()
        hyp = transcribe(asr, array, sr, args.beams)
        dt = time.time() - t0
        dur = len(array) / sr
        total_audio += dur
        total_time += dt
        file_wer = None
        if ref:
            refs.append(ref)
            hyps.append(hyp)
            file_wer = jiwer.wer(norm_fair(ref), norm_fair(hyp)) * 100
        rows.append((name, dur, dt, ref, hyp, file_wer))
        w = f"  WER {file_wer:.0f}%" if file_wer is not None else "  (no .txt)"
        print(f"\n[{name}] {dur:.1f}s audio, {dt:.2f}s{w}\n  HYP: {hyp}" + (f"\n  REF: {ref}" if ref else ""))

    s = scores(refs, hyps)
    rtf = total_time / total_audio if total_audio else 0
    out_path = out_name(args.model, args.beams, os.path.basename(folder))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"model: {args.model}\nbeams: {args.beams}\nfolder: {folder}\n")
        f.write(f"files: {len(files)} ({len(refs)} with reference .txt)\n")
        if s:
            f.write(f"WER: {s[0]:.1f}%\nWER (fair): {s[1]:.1f}%\nCER (fair): {s[2]:.1f}%\n")
        f.write(f"speed: {total_time:.1f}s for {total_audio:.1f}s of audio (real-time factor {rtf:.2f})\n\n")
        f.write("Fill in MEANING OK for each file: yes = the request would still be understood correctly.\n\n")
        for name, dur, dt, ref, hyp, fw in rows:
            f.write(f"FILE: {name}  ({dur:.1f}s audio, {dt:.2f}s)" + (f"  WER {fw:.0f}%" if fw is not None else "") + "\n")
            if ref:
                f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\nMEANING OK: __\n\n")

    print("\n" + "=" * 60)
    if s:
        print(f"{len(refs)} files with reference: WER {s[0]:.1f}%   WER (fair) {s[1]:.1f}%   CER (fair) {s[2]:.1f}%")
    else:
        print("No reference .txt files found - add 'name.txt' with what was said next to each audio for WER.")
    print(f"Speed: real-time factor {rtf:.2f} ({total_time:.1f}s for {total_audio:.1f}s of audio)")
    print(f"Saved to {out_path} - open it and fill in MEANING OK for each file.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--n", type=int, default=100, help="number of FLEURS test samples")
    p.add_argument("--audio", help="transcribe this one audio file")
    p.add_argument("--audio-dir", help="transcribe (and score) every audio file in this folder")
    p.add_argument("--beams", type=int, default=DEFAULT_BEAMS,
                   help="Whisper beam search width (default 5; 1 = greedy, faster)")
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    if is_gigaam(args.model):
        if args.beams != DEFAULT_BEAMS:
            print("Note: GigaAM uses CTC greedy decoding - --beams is ignored.")
        args.beams = 1
    asr = load_asr(args.model)
    os.makedirs("results", exist_ok=True)

    if args.audio:
        array, sr = load_audio(args.audio)
        print("TEXT:", transcribe(asr, array, sr, args.beams))
    elif args.audio_dir:
        run_dir(asr, args)
    else:
        run_fleurs(asr, args)


if __name__ == "__main__":
    main()
