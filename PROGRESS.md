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
| Inference server | kronk (built on yzma / llama.cpp), runs in `tmux` session `kronk` |
| kronk API | **`http://localhost:11435/v1`** (OpenAI-compatible; the default in the scripts). kronk also listens on 9000 and 11445 (not the API). All bound to 127.0.0.1 |
| kronk model IDs | `google/gemma-4-12b-it-qat-q4_0`, `google/gemma-4-E4B_q4_0-it`, `google/gemma-4-26B_q4_0-it`, Qwen2.5-7B (exact ID: check `/v1/models`) |
| Models registered in kronk | via hard links (`ln`) from `~/models` into `~/.kronk/models/...`, so no extra disk space is used |
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
| 2026-11-26 | openai/whisper-large-v3-turbo | 100 (100 unique sentences) | **147.3%** | **103.2%** | 0.92 s/sample | Raw score, unusable. See the analysis below |
| 2026-11-26 | same output, converted Cyrillic→Latin | 100 | 132.4% | 48.7% | – | Script fixed. Average still inflated by a few hallucination loops |
| 2026-11-26 | same, **median per sentence** | 100 | **89%** | **27%** | – | The "typical sentence" picture |

**What the HYP lines show (analysis of `results/asr_openai_whisper-large-v3-turbo.txt`):**

| Failure type | Count / 100 | Example |
|---|---|---|
| Writes in **Kazakh-style Cyrillic** (ұ, қ, ғ, ң, і, ә) instead of Uzbek Latin | 86 | REF `qayta qurish sa'y harakatlarini` → HYP `кайта құруш сайхаракатларын` |
| **Hallucination loops / wrong alphabet** (e.g. Ethiopic `ጡ ጡ ጡ …`) | ~5 (14 with non-Latin/Cyrillic text) | single sentences with WER 450%, 1564%, 2018% |
| Latin, but wrong spelling and word boundaries | the rest | REF `biroq hali ham qushlar` → HYP `uraq hálxam kusar` |

**Interpretation:**
- Whisper **hears the sounds fairly well**: the median CER after script conversion is 27%, and the hypothesis word count matches the reference (median ratio 1.0).
- But it **doesn't know Uzbek spelling**. It writes Uzbek the way Kazakh is written, so almost every word gets one or two letters wrong, and the WER stays around 90%.
- 5% of the clips trigger runaway hallucination loops that inflate the average.
- This is good news for Phase 2. Fine-tuning mainly has to teach **spelling and script**, not hearing, and that usually improves quickly.

**Conclusion:** stock Whisper cannot do Uzbek, so **Phase 2 (ASR fine-tuning) is mandatory**, and the outlook is promising.

**Next:**
- [x] Inspect the REF/HYP lines to find the failure type → Kazakh-style Cyrillic + ~5% loops
- [x] Re-score with Cyrillic→Latin conversion → median CER 27%
- [x] Test ready-made Uzbek Whisper fine-tunes with `bash asr_compare.sh` → results below
- [x] Added fair scoring, `--beams`, and `--audio-dir` (own recordings + `.txt` references + MEANING OK column)
- [ ] Test on own recordings: `bash asr_compare.sh 0 --audio-dir test_audios` (FLEURS may be in navai-uz's training data)
- [ ] Try `--beams 5` on FLEURS and on own recordings

### 1.1b ASR: ready-made Uzbek Whisper fine-tunes (same 100 FLEURS test clips)

| Date | Model | WER | CER | Median sentence WER | Perfect sentences | Speed | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-27 | **navai-uz/whisper-medium-uzbek** | **14.1%** | **4.8%** | 10% | 27/100 | 2.04 s/sample | 🏆 Best. ⚠️ Trained with FLEURS data, so the score may be flattering |
| 2026-11-27 | navai-uz/whisper-small-uzbek | 17.0% | 5.4% | 14% | 21/100 | 1.11 s/sample | Nearly as good at ~2× the speed |
| 2026-11-27 | Abduqayum/whisper-uzbek-medium-callcenter | 19.5% | 6.1% | 17% | 15/100 | 2.03 s/sample | **Not** trained on FLEURS, so this is an honest out-of-domain score. Trained on phone speech, while FLEURS is clean read speech |
| 2026-11-26 | openai/whisper-large-v3-turbo (stock) | 147.3% | 103.2% | 89% (after translit) | – | 0.92 s/sample | Reference |

**Fair re-scoring (2026-11-27)** of the same outputs with `uz_text.norm_fair`: numbers as words, Cyrillic→Latin, `km`→`kilometr`. FLEURS references write numbers as digits (`1963`), while the models say them as words (`ming to'qqiz yuz oltmish uch`), and basic scoring counted those as errors:

| Model | WER (basic) | **WER (fair)** | CER (fair) |
|---|---|---|---|
| navai-uz/whisper-medium-uzbek | 14.1% | **11.3%** | 2.9% |
| navai-uz/whisper-small-uzbek | 17.0% | **14.3%** | 3.6% |
| Abduqayum/whisper-uzbek-medium-callcenter | 19.5% | **16.3%** | 4.0% |
| openai/whisper-large-v3-turbo | 147.3% | 130.5% | 52.2% |

→ About **3 points of the 14% were fake errors** caused by spelling conventions. **From now on, report WER (fair).**

**What the remaining errors look like:**
- No Cyrillic output and no hallucination loops in any of the three models.
- Most errors are **suffix variations** (`harakatlarini` → `harakatlarni`, `ichimliklar` → `ichimliklari`) and **foreign names** (`jeyms paget` → `james paket`, `greyt yarmutdagi` → `great yarmarkadagi`, `t rex`).
- Very few errors come from apostrophes or word splitting (1–2 and 6–7 per model).
- Some "errors" are mistakes in the FLEURS reference itself: the REF says `januniy`, and navai-medium wrote the correct `janubiy`.
- The call-center model often writes `h` as `x` (`hujayra` → `xo'jayra`), probably a spelling habit from its training transcripts.

**Speed note:** the medium models are ~2× slower than turbo (24 decoder layers vs 4). FLEURS clips are ~10–12 s long, so 2 s is a real-time factor of ~0.2. A typical 3–5 s voice-agent question would take ~0.6–1 s. If that's too slow, convert to CTranslate2 (faster-whisper) later, which is typically 2–4× faster. kronk was also using the GPU during this test.

**Conclusion:** ASR went from unusable (147%) to good (14–20%) **without any training**. Phase 2 changes from "must fine-tune" to **"start from navai-uz/whisper-medium-uzbek (or small) and fine-tune on our own domain recordings"**. The deciding test is now **your own recordings**, especially if the agent will take phone calls, where the call-center model may win.

### 1.2 LLM: Uzbek prompts via kronk

Score = average of the 1–5 manual scores over 20 prompts in `prompts_uz.txt`.

| Date | Model | Avg score | Avg time/answer | tok/s | Common problems (flags) |
|---|---|---|---|---|---|
| 2026-11-27 | gemma-4-12b-it-qat-q4_0 | _not scored yet_ | 11.0 s | 27.3 | |
| 2026-11-27 | gemma-4-E4B_q4_0-it | _not scored yet_ | 4.9 s | 53.1 | |
| 2026-11-27 | gemma-4-26B_q4_0-it | _not scored yet_ | 6.6 s | 45.0 | Faster than the 12B despite its size, probably a mixture-of-experts model (only part of it runs per token) |
| 2026-11-27 | Qwen2.5-7B-Instruct-Q4_K_M | _not scored yet_ | 2.8 s | 35.0 | Short answers, hence the low time |
| | uzlm/alloma-8B-Instruct Q4_K_M (`--apst`) | | | | |
| | uzlm/alloma-8B-Instruct Q4_K_M (no APST) | | | | |

### 1.3 TTS

| Date | Model | Clarity (clear / robotic / hard to understand) | Speed (audio s / gen s) | Notes |
|---|---|---|---|---|
| | facebook/mms-tts-uzb-script_cyrillic | | | |
| | uzlm/sayro-tts-1.7B | | | Gated: access requested? License OK for our use? Built-in speaker or reference voice? |

### 1.4 End-to-end pipeline

| Date | ASR model | LLM | ASR s | LLM s | TTS s | Total s | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | Target: < 1.5 s total |

---

## Ready-made Uzbek models found on Hugging Face (2026-11-27)

| Part | Model | Base | Size | License | Notes |
|---|---|---|---|---|---|
| LLM | `uzlm/alloma-8B-Instruct` | Llama-3.1-8B-Instruct | 16 GB bf16 | Llama 3.1 | Continual pretraining on 3.6B tokens (33% Uzbek) + SFT. Custom Uzbek tokenizer (1.7 tokens/word vs 3.5). Author benchmarks: best of the 8B Uzbek models on Uz→En translation. Recommends replacing `'` with `APST` in input. No official GGUF |
| LLM | `kmamaroziqov/alloma-8b-q4` | alloma-8B | 4.9 GB | – | Third-party Q4_K_M GGUF, unverified. Can be tested in kronk directly |
| LLM | `uzlm/alloma-3B-Instruct`, `-1B-` | Llama-3.2 | small | Llama 3.2 | Low-latency options |
| TTS | `uzlm/sayro-tts-1.7B` | Qwen3-TTS-12Hz-1.7B | 3.8 GB + 0.7 GB | other (**gated**: request access and read the license) | Trained on Uzbek YouTube speech (IT, news, podcasts with Tashkent dialect). Ships `uzbek_normalizer.py`. Likely far better than MMS |
| ASR | `navai-uz/whisper-medium-uzbek` | whisper-medium | ~1.5 GB | Apache-2.0 | Common Voice 22 + FLEURS + FeruzaSpeech. ⚠️ Trained on FLEURS, so the FLEURS test score may look better than real use |
| ASR | `Abduqayum/whisper-uzbek-medium-callcenter` | whisper-medium | ~1.5 GB | Apache-2.0 | 780 h dataset, call-center domain |
| ASR | `hostmepanda/whisper-large-v3-turbo-uzbek-ct2` | turbo | – | – | CTranslate2 format (faster-whisper), not transformers |

**Plan change:** test these in Phase 1, and use the winners as the **starting point** for Phase 2/3 fine-tuning instead of the stock models.

---

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-26 | Fine-tune existing models instead of training from scratch | One L4 can't pretrain an LLM, and there isn't enough Uzbek text for it |
| 2026-11-26 | Pipeline = ASR → LLM (kronk) → TTS | Each part can be fine-tuned separately on the L4 |
| 2026-11-26 | Store LoRA adapters on the Hugging Face Hub, not GitHub | GitHub's 100 MB file limit and history bloat; saves server disk |
| 2026-11-26 | Use the Latin script internally; convert to Cyrillic only for MMS TTS | MMS Uzbek TTS reads Cyrillic only |
| 2026-11-26 | ASR fine-tuning (Phase 2) is required | Stock Whisper WER 147%: writes Uzbek in Kazakh-style Cyrillic, plus ~5% hallucination loops |
| 2026-11-27 | Report **fair WER** (numbers as words, Cyrillic→Latin, unit abbreviations expanded) | Basic WER counted `1963` vs `ming to'qqiz yuz oltmish uch` as errors; ~3 points of the 14% were fake |
| 2026-11-27 | Track **"meaning OK"** on own recordings next to WER | A wrong suffix rarely changes the request; a wrong number, name or negation does. This is what matters for the agent |
| 2026-11-27 | ASR starting point: navai-uz/whisper-medium-uzbek (14.1% WER), with whisper-small (17.0%, 2× faster) as the low-latency option | Best FLEURS scores; final pick after testing on our own recordings |
| 2026-11-27 | Test uzlm alloma (LLM) and sayro (TTS) plus Uzbek Whisper fine-tunes in Phase 1 | Already Uzbek-trained; winners become the starting point for fine-tuning |
| 2026-11-27 | Convert alloma to GGUF ourselves (`convert_alloma.sh`) instead of using the third-party GGUF | Its custom tokenizer needs a llama.cpp patch (unknown vocab hash → treat as Llama-3 `llama-bpe`), and we can't verify how the third-party file was made |
| 2026-11-27 | sayro-tts runs in a separate `.venv-tts` | `qwen-tts` pins transformers==4.57.3; keeps the main venv stable |
| 2026-11-26 | Server pushes results via an SSH deploy key (this repo only) | GitHub rejects passwords; a deploy key limits access if the cloud VM is compromised |

---

## Open questions

- What is the agent for (phone support, booking, general assistant)? This decides the Phase 3 training data.
- Does kronk/yzma support loading a LoRA adapter on top of a base GGUF, or do we need to merge?
