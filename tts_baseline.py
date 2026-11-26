"""Step 1.3 - TTS baseline: Meta MMS Uzbek voice.

The MMS Uzbek model reads CYRILLIC script, so Latin text is converted automatically.

Usage:
    python tts_baseline.py
    python tts_baseline.py --text "Salom, sizga qanday yordam bera olaman?"
    python tts_baseline.py --text "..." --out results/my_test.wav
"""
import argparse
import os
import time

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

from translit import needs_conversion, to_cyrillic

MODEL = "facebook/mms-tts-uzb-script_cyrillic"

DEFAULT_TEXTS = [
    "Assalomu alaykum, sizga qanday yordam bera olaman?",
    "Buyurtmangiz ertaga soat o'n ikkigacha yetkazib beriladi.",
    "Kechirasiz, savolingizni tushunmadim. Iltimos, qaytadan ayting.",
    "Ассалому алайкум, менинг исмим Нигора.",
]


class UzbekTTS:
    def __init__(self, model_name: str = MODEL, device: str = "cuda"):
        self.device = device
        self.model = VitsModel.from_pretrained(model_name).to(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.sr = self.model.config.sampling_rate

    def synth(self, text: str):
        if needs_conversion(text):
            text = to_cyrillic(text)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model(**inputs).waveform[0].cpu().numpy(), text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", help="text to speak (Latin or Cyrillic)")
    p.add_argument("--out", default=None, help="output .wav (only with --text)")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    tts = UzbekTTS()
    texts = [args.text] if args.text else DEFAULT_TEXTS

    for i, text in enumerate(texts, 1):
        t0 = time.time()
        wav, spoken = tts.synth(text)
        dt = time.time() - t0
        out = args.out if (args.text and args.out) else f"results/tts_{i}.wav"
        sf.write(out, wav, tts.sr)
        print(f"[{i}] {text}\n    spoken as: {spoken}\n    -> {out}  "
              f"({len(wav) / tts.sr:.1f}s audio in {dt:.2f}s)")

    print("\nListen to the .wav files and note: clear / robotic / hard to understand.")


if __name__ == "__main__":
    main()
