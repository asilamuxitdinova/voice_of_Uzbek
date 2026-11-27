"""Step 1.3b - TTS test with uzlm/sayro-tts-1.7B (Uzbek fine-tune of Qwen3-TTS).

Run inside the separate TTS environment:
    bash setup_sayro.sh && source .venv-tts/bin/activate
    python tts_sayro.py
    python tts_sayro.py --text "Salom, sizga qanday yordam bera olaman?"
    python tts_sayro.py --ref-audio my_voice.wav --ref-text "exact words spoken in my_voice.wav"

The model is GATED: request access on its Hugging Face page first.

sayro is fine-tuned from the Qwen3-TTS *Base* (voice-clone) model, so it may need a short
reference recording (--ref-audio, 5-15 s, ideally with --ref-text). The script first tries
built-in speakers; if the model has none, it asks for a reference. The model card on the
Hub has the authoritative usage - the script prints where it was downloaded so you can read it.
"""
import argparse
import importlib.util
import os
import time

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

MODEL = "uzlm/sayro-tts-1.7B"

# same sentences as tts_baseline.py (minus the Cyrillic one) so the two voices can be compared
DEFAULT_TEXTS = [
    "Assalomu alaykum, sizga qanday yordam bera olaman?",
    "Buyurtmangiz ertaga soat o'n ikkigacha yetkazib beriladi.",
    "Kechirasiz, savolingizni tushunmadim. Iltimos, qaytadan ayting.",
]


def try_call(fn, default):
    try:
        return fn()
    except Exception:
        return default


def load_normalizer(model_dir: str):
    """Load the model's own uzbek_normalizer.py (only with --normalize; read the file first)."""
    path = os.path.join(model_dir, "uzbek_normalizer.py")
    if not os.path.exists(path):
        print("No uzbek_normalizer.py in the model folder.")
        return None
    spec = importlib.util.spec_from_file_location("uzbek_normalizer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    funcs = [n for n in dir(mod) if "normal" in n.lower() and callable(getattr(mod, n))
             and not isinstance(getattr(mod, n), type)]
    if not funcs:
        print(f"No normalize function found in {path}; check the file and tell Claude its name.")
        return None
    print(f"Using normalizer function: {funcs[0]}()")
    return getattr(mod, funcs[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text")
    p.add_argument("--out", help="output .wav (only with --text)")
    p.add_argument("--language", default="Auto", help="Qwen3-TTS language name, or Auto")
    p.add_argument("--speaker", help="built-in speaker name, if the model has any")
    p.add_argument("--ref-audio", help="reference voice clip (5-15 s .wav) for voice cloning")
    p.add_argument("--ref-text", help="exact transcript of --ref-audio (improves quality)")
    p.add_argument("--normalize", action="store_true", help="apply the model's uzbek_normalizer.py")
    args = p.parse_args()

    print(f"Downloading {MODEL} (~4.5 GB, first run only) ...")
    model_dir = snapshot_download(MODEL)
    print(f"Model files: {model_dir}\n  -> read {os.path.join(model_dir, 'README.md')} for the official usage")

    model = Qwen3TTSModel.from_pretrained(
        model_dir, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    speakers = try_call(model.get_supported_speakers, None) or []
    languages = try_call(model.get_supported_languages, None) or []
    print("Built-in speakers:", speakers or "(none)")
    print("Languages:", languages or "(unknown)")

    normalize = load_normalizer(model_dir) if args.normalize else None

    if args.ref_audio:
        mode = "voice clone"
    elif speakers:
        mode = "built-in speaker"
        speaker = args.speaker or speakers[0]
    else:
        raise SystemExit(
            "\nThis model has no built-in speakers, so it needs a reference voice.\n"
            "Record 5-15 s of clear Uzbek speech and run:\n"
            "  python tts_sayro.py --ref-audio ref.wav --ref-text \"what you said in the clip\"\n"
            f"(also check {os.path.join(model_dir, 'README.md')} - it may describe another way)"
        )
    print(f"Mode: {mode}\n")

    os.makedirs("results", exist_ok=True)
    texts = [args.text] if args.text else DEFAULT_TEXTS
    for i, text in enumerate(texts, 1):
        spoken = normalize(text) if normalize else text
        t0 = time.time()
        if mode == "voice clone":
            kwargs = dict(text=spoken, language=args.language, ref_audio=args.ref_audio)
            if args.ref_text:
                kwargs["ref_text"] = args.ref_text
                wavs, sr = model.generate_voice_clone(**kwargs)
            else:
                # no transcript: use only the speaker embedding of the reference clip
                prompt = model.create_voice_clone_prompt(ref_audio=args.ref_audio, x_vector_only_mode=True)
                wavs, sr = model.generate_voice_clone(
                    text=spoken, language=args.language, voice_clone_prompt=prompt)
        else:
            wavs, sr = model.generate_custom_voice(text=spoken, language=args.language, speaker=speaker)
        dt = time.time() - t0

        out = args.out if (args.text and args.out) else f"results/sayro_{i}.wav"
        sf.write(out, wavs[0], sr)
        print(f"[{i}] {text}\n    spoken as: {spoken}\n    -> {out}  ({len(wavs[0]) / sr:.1f}s audio in {dt:.2f}s)")

    print("\nCompare with results/tts_*.wav from tts_baseline.py (MMS).")


if __name__ == "__main__":
    main()
