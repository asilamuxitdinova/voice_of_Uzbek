"""Step 1.4 - End-to-end test: audio question -> ASR -> LLM (kronk) -> TTS -> audio answer.

Record a short Uzbek question as a .wav (phone recorder is fine), copy it to the server, then:

    python voice_pipeline.py --audio question.wav --base-url http://localhost:11435/v1

Prints the time each stage takes. Target for a natural conversation: under ~1.5 s total.
"""
import argparse
import os
import time

import librosa
import requests
import soundfile as sf

from asr_baseline import load_asr, transcribe
from llm_baseline import DEFAULT_SYSTEM, from_apst, pick_model, to_apst
from tts_baseline import UzbekTTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="input .wav/.mp3/.ogg with an Uzbek question")
    p.add_argument("--out", default="results/answer.wav")
    p.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:11435/v1"))
    p.add_argument("--model", default=os.getenv("LLM_MODEL"))
    p.add_argument("--apst", action="store_true", help="APST input for uzlm/alloma models")
    args = p.parse_args()
    enc = to_apst if args.apst else (lambda s: s)

    base = args.base_url.rstrip("/")
    model = args.model or pick_model(base)

    print("Loading ASR and TTS models (one-time) ...")
    asr = load_asr(args.asr_model)
    tts = UzbekTTS()
    array, sr = librosa.load(args.audio, sr=16000, mono=True)

    # warm-up so the timings below reflect steady state, not first-call overhead
    transcribe(asr, array[: sr], sr)
    tts.synth("salom")

    t0 = time.time()
    question = transcribe(asr, array, sr)
    t1 = time.time()

    r = requests.post(
        f"{base}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": enc(DEFAULT_SYSTEM)},
                {"role": "user", "content": enc(question)},
            ],
            "max_tokens": 200,
            "temperature": 0.3,
        },
        timeout=300,
    )
    r.raise_for_status()
    answer = from_apst(r.json()["choices"][0]["message"]["content"].strip())
    t2 = time.time()

    wav, spoken = tts.synth(answer)
    t3 = time.time()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sf.write(args.out, wav, tts.sr)

    print(f"\nUSER (ASR):  {question}")
    print(f"AGENT (LLM): {answer}")
    print(f"TTS input:   {spoken}")
    print(f"\nASR {t1 - t0:.2f}s | LLM {t2 - t1:.2f}s | TTS {t3 - t2:.2f}s | TOTAL {t3 - t0:.2f}s")
    print(f"Answer audio saved to {args.out}")


if __name__ == "__main__":
    main()
