# Progress & Results Log

Every result, number, and decision goes here, newest at the bottom of each section.
Raw outputs stay in `results/`. This file is the summary.

---

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA L4, 24 GB VRAM (23034 MiB usable) |
| Driver / CUDA | 595.91.07 / 13.2 |
| Python (server venv) | 3.14 |
| PyTorch | cu128 wheels |
| Inference server | kronk (built on yzma / llama.cpp), OpenAI-compatible API |
| Repo on server | `~/voice_of_Uzbek` |

**GGUF models already on the server (`~/models`):**

| Model | Use for this project |
|---|---|
| gemma-4-12b-it-qat-q4_0 | ⭐ Main LLM candidate |
| gemma-4-E4B_q4_0-it | ⭐ Fast/low-latency candidate |
| gemma-4-26B_q4_0-it | Quality ceiling (~15 GB VRAM, check fit with ASR+TTS) |
| Qwen2.5-7B-Instruct-Q4_K_M | Comparison baseline |
| qwen2.5-3b, SmolLM2-1.7B | Skip (too small / English-only) |
| Qwen2.5-Coder-*, *-gojson-* | Skip (other projects) |

---

## Phase 0: Setup ✅ Done (2026-11-26)

- `setup.sh` completed and the GPU was detected.
- **Issue:** `datasets<4` crashes on Python 3.14 (`TypeError: Pickler._batch_setitems()`).
  **Fix:** `asr_baseline.py` now downloads FLEURS files directly from the Hub and no longer uses the `datasets` library.

---

## Phase 1: Baseline 🔄 In progress

### 1.1 ASR: stock Whisper on FLEURS Uzbek test

| Date | Model | Samples | WER | CER | Speed | Notes |
|---|---|---|---|---|---|---|
| 2026-11-26 | openai/whisper-large-v3-turbo | 100 (100 unique sentences) | **147.3%** | **103.2%** | 0.92 s/sample | Unusable. WER >100% means hallucination/repetition loops or the wrong script/language (Cyrillic/Russian). Root cause still to be confirmed from the HYP lines |

**Conclusion so far:** stock Whisper cannot do Uzbek, so **Phase 2 (ASR fine-tuning) is mandatory.**

**Next:**
- [ ] Inspect the REF/HYP lines to find the failure type (repetition vs. Cyrillic/Russian)
- [ ] Re-score with Cyrillic→Latin conversion and anti-repetition settings
- [ ] Test alternatives: Meta MMS (`uzb-script_latin` adapter), community Uzbek Whisper fine-tunes

### 1.2 LLM: Uzbek prompts via kronk

Score = average of the 1–5 manual scores over 20 prompts in `prompts_uz.txt`.

| Date | Model | Avg score | Avg time/answer | tok/s | Common problems (flags) |
|---|---|---|---|---|---|
| | gemma-4-12b-it-qat-q4_0 | | | | |
| | gemma-4-E4B_q4_0-it | | | | |
| | gemma-4-26B_q4_0-it | | | | |
| | Qwen2.5-7B-Instruct-Q4_K_M | | | | |

### 1.3 TTS: facebook/mms-tts-uzb-script_cyrillic

| Date | Clarity (clear / robotic / hard to understand) | Speed (audio s / gen s) | Notes |
|---|---|---|---|
| | | | |

### 1.4 End-to-end pipeline

| Date | ASR model | LLM | ASR s | LLM s | TTS s | Total s | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | Target: < 1.5 s total |

---

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-26 | Fine-tune existing models instead of training from scratch | One L4 can't pretrain an LLM, and there isn't enough Uzbek text for it |
| 2026-11-26 | Pipeline = ASR → LLM (kronk) → TTS | Each part can be fine-tuned separately on the L4 |
| 2026-11-26 | Store LoRA adapters on the Hugging Face Hub, not GitHub | GitHub's 100 MB file limit and history bloat; saves server disk |
| 2026-11-26 | Use the Latin script internally; convert to Cyrillic only for MMS TTS | MMS Uzbek TTS reads Cyrillic only |
| 2026-11-26 | ASR fine-tuning (Phase 2) is required | Stock Whisper WER 147% |

---

## Open questions

- What is the agent for (phone support, booking, general assistant)? This decides the Phase 3 training data.
- Does kronk/yzma support loading a LoRA adapter on top of a base GGUF, or do we need to merge?
