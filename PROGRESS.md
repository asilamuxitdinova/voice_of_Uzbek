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
| Disk (2026-11-27) | 193 GB total, 124 GB used, **70 GB free**. Not enough for the ~66 GB+ Phase 2 training data plus checkpoints, so ask for more or train on a compressed subset |

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
- [x] Added `ai-sage/GigaAM-Multilingual` (`large_ctc` 600M and `@ctc` 220M) to `asr_baseline.py` and `asr_compare.sh` (2026-11-27). Remote code reviewed and pinned (large_ctc `3905cd5`, ctc `2f8a571`). Audio over 25 s is split at quiet points (pyannote not needed)
- [x] Run GigaAM on FLEURS → **winner**, see 1.1d
- [x] Run `Kotib/uzbek_stt_v1` on FLEURS → **6.3% fair WER, best so far**
- [ ] Run all 4 models on own voice messages (`PATTERN='audio_*' SET=my_voice bash run_test_set.sh`), then on `test_set/`
- [ ] Run GigaAM on own recordings: `python asr_baseline.py --model ai-sage/GigaAM-Multilingual --audio-dir test_audios`
- [ ] Add `.txt` references (what was actually said) to `test_audios/` so own-recording WER can be measured

### 1.1d ASR: GigaAM-Multilingual vs Whisper (FLEURS, same 100 clips, 2026-11-27)

| Model | WER (basic) | **WER (fair)** | CER (fair) | Perfect sentences | Speed | License |
|---|---|---|---|---|---|---|
| 🏆 **Kotib/uzbek_stt_v1** (whisper-medium, 769M) | 9.3% | **6.3%** | **1.4%** | **48/100** | 2.01 s/clip (5 beams) | Apache-2.0 |
| ai-sage/GigaAM-Multilingual (large_ctc, 600M) | 11.1% | 8.1% | 1.4% | 35/100 | **0.35 s/clip** | MIT |
| navai-uz/whisper-medium-uzbek | 14.1% | 11.3%* | 2.9% | 30/100 | 2.04 s/clip | Apache-2.0 |

\* navai's fair WER was recomputed from the saved, already-normalized lines, which loses a few ordinals like `15-yanvar`. On the same basis, GigaAM gets 8.6%, so the honest gap is **8.6% vs 11.3%**.

- **Kotib (added 2026-11-27):** on the same basis (recomputed from the saved lines), **Kotib 6.6% vs GigaAM 8.6% vs navai 11.3%**. Median sentence WER 4%, and 48/100 sentences are perfect. Sentence by sentence vs GigaAM: Kotib better on **37**, GigaAM on **19**, and **44** ties. Errors: foreign names (`jeyms paget`→`james paket`), a few real misses (`tarkib topgan`→`tark etayotgan`, `o'liklar`→`o'riklari`). ⚠️ Its training data is "custom" 1,600 h, so we can't rule out that FLEURS train was in it. ⚠️ **~6× slower than GigaAM** (2.0 vs 0.35 s/clip, with 5 beams). Own recordings will decide.
- Sentence by sentence: GigaAM is better on **42**, navai on **20**, and **38** are ties. The worst GigaAM sentence is 36% WER, vs 72% for navai.
- **~6× faster** than navai-medium, though navai ran with 5-beam search (see the beam note below).
- GigaAM wins even though navai was **trained on FLEURS** and GigaAM (as far as we know) wasn't.
- No Cyrillic in the output, even though GigaAM's alphabet includes Cyrillic letters.
- Remaining errors: foreign names (`jeyms`→`jemes`, `t rex`→`terex`, `martelli`→`marteli`), joined/split compounds (`qora qayin`→`qoraqayin`, `o'rta yer`→`o'rtayer`, which are partly spelling conventions), a few suffixes.
- Output is lowercase with no punctuation, while Whisper call-center adds punctuation. For an LLM this doesn't matter much.

⚠️ **Beam-search finding:** transformers 5's ASR pipeline **silently uses `num_beams=5`** when nothing is passed. So *all* Whisper results above (FLEURS and own recordings) already used 5 beams, and the `--beams 5` runs on own recordings were identical to the normal runs. Fixed in `asr_baseline.py`: `num_beams` is now always passed, the default is 5 (keeps old numbers comparable), and **`--beams 1` = real greedy** (2.3× faster in a CPU test). The duplicate `*_beam5_test_audios.txt` files can be deleted.

### 1.1e Own recordings (`test_audios/`, 7 conversational voice messages, 83 s, no references yet)

No WER without `.txt` references. Qualitative comparison of the transcripts:
- **Abduqayum call-center** gives the most coherent output on this conversational speech, with punctuation and capitalization (`Boraman, Xudo xohlasa. Borganda yana mazza qilib ishlaymiz.`).
- **navai-medium** makes meaning-changing errors on casual speech: `Xudo xohlasa` → `xudo xoʻrlashdi`, `sogʻ boʻling` → `sogʻinmiz salomatni salqin`.
- **navai-small** is worse still (`aqlni saqlantirilar`, `kinomalar`).
- This matches the benchmark authors' warning: FLEURS (read speech) flatters models, and conversational speech is much harder.
- Speed (real-time factor): small 0.10, medium 0.18, call-center 0.21.
- GigaAM hasn't been run on these yet.

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
| ASR | `Kotib/uzbek_stt_v1` (Kotibai & Rubai) | whisper-medium | 3.1 GB | **Apache-2.0** | ~1,600 h, 3-stage curriculum (foundation 725 h → robustness 394 h → domain 474 h). Handles Russian loanwords in Latin (`brat`, `davay`). Their own score: 16.7% WER over 8 mixed test sets (clean 6–11%, noisy 12–24%, dialects 16–25%), not comparable to our FLEURS numbers. Added 2026-11-27 |
| ASR | `navai-uz/whisper-medium-uzbek` | whisper-medium | ~1.5 GB | Apache-2.0 | Common Voice 22 + FLEURS + FeruzaSpeech. ⚠️ Trained on FLEURS, so the FLEURS test score may look better than real use |
| ASR | `Abduqayum/whisper-uzbek-medium-callcenter` | whisper-medium | ~1.5 GB | Apache-2.0 | 780 h dataset, call-center domain |
| ASR | `hostmepanda/whisper-large-v3-turbo-uzbek-ct2` | turbo | – | – | CTranslate2 format (faster-whisper), not transformers |

| ASR | **`ai-sage/GigaAM-Multilingual`** (600M) | Conformer CTC, 70+ languages | – | **MIT**, not gated | **7.3% WER on FLEURS Uzbek** out of the box (their numbers: digits excluded, numbers→words), vs 11.3% fair for navai-medium. CTC = very fast. Strong candidate, see below (added 2026-11-27) |
| ASR | `rustam1221/uzbek-asr-gigaam` | GigaAM-Multilingual + Uzbek fine-tune + KenLM | 600M / 220M | other (**gated**) | **8.1% WER on real spontaneous speech** (podcasts, interviews), while whisper-large-v3-turbo-uzbek gets 38.6% on the same test |

**Plan change:** test these in Phase 1, and use the winners as the **starting point** for Phase 2/3 fine-tuning instead of the stock models.

### Open Uzbek speech datasets (found 2026-11-27)

| Dataset | Hours | License | Labels | Use for |
|---|---|---|---|---|
| `openlimi/uzbek-speech-1700h` | **1,634** (train 1,585 / val 23 / test 25) | mixed open (check sources) | crowd + automatic; ⚠️ **converted to the new Uzbek alphabet** (`text_orig` keeps the old one) | Biggest. Deduplicated, no test leakage |
| `uzinfocom-edu-ai/uzbek-asr-curated-701h` | 701 | Apache-2.0 | Common Voice + UzbekVoice + ISSAI USC + islomov news/IT/podcasts | ❌ **Audio not uploaded (checked 2026-11-27):** only `audios/00` exists (~0.2 GB), shards 01–99 are 404, though the commit says "166 GB". Unusable as published, so rebuild it from the original sources below |
| `Abduqayum/Uzbek-STT-Dataset-780h` | 780 | other | ⚠️ **Gemini-generated** transcripts (mostly right) | Audiobooks, podcasts, IT talks |
| `issai/Uzbek_Speech_Corpus` (USC) | 105 | **MIT** | Manual, checked by native speakers | Clean studio speech, 958 speakers |
| Common Voice (e.g. `fsicoli/common_voice_22_0`, `uz`) | ~150+ | **CC0** | Crowd-verified | Read speech, many speakers |
| `DavronSherbaev/uzbekvoice-filtered` | large | Apache-2.0 | Filtered UzbekVoice | Read speech |
| `islomov/{news,it,podcasts_tashkent_dialect}_youtube_uzbek_speech_dataset` | 3 sets | **Apache-2.0** | YouTube | News, IT, Tashkent dialect |
| `k2speech/FeruzaSpeech` | – | other (gated) | – | Single speaker (TTS too) |
| `sarulab-speech/yodas2_sidon` / YODAS2 | – | CC-BY-3.0 | YouTube, automatic | Very large multilingual, contains Uzbek |

**Evaluation sets:**
| Dataset | What | Why it matters |
|---|---|---|
| `rustam1221/uzbek-asr-benchmark-spontaneous` | 311 clips, 2 h of **spontaneous** speech (podcasts, interviews, code-switching), human-verified | "Models score 2–3× better on FLEURS than on real conversation." This is the honest test for a voice agent |
| `OvozifyLabs/asr_evaluate_set` | small eval set | Apache-2.0 |

**Text data for the LLM (Phase 3):** `AsrorAsr/uzbek-customer-support-dialogs`: 990 support dialogs, 27 intents, Apache-2.0.

**TTS data:** `firdavsus/TTS_dataset_single_speaker` (CC-BY-4.0, pre-tokenized audio, not raw wav), `MuzaffarSharofitdinov/uzbek-tts-*` (gated).

**Storage size of the usable sources** (checked 2026-11-27): ISSAI USC 9.5 GB · islomov news 15.9 GB · islomov IT 15.8 GB · islomov podcasts 11.1 GB · uzbekvoice-filtered 14.1 GB (≈ 66 GB, ~540 h together, without Common Voice) · openlimi 1634 h = 118.6 GB.

### Phase 2 plan: fine-tune GigaAM-Multilingual (2026-11-27)
- Official MIT training code: `salute-developers/GigaAM` → `train_utils/train.py`. It continues training a finished CTC model from TSV manifests (`path`, `duration`, `transcription_raw`), with `--raw_text`, `--precision bf16` and `--activation_checkpointing`.
- Prior evidence: `rustam1221/uzbek-asr-gigaam` fine-tuned the same model and got 14.4% → **10.4%** WER on spontaneous speech (→ 8.1% with KenLM).
- Data: rebuild the 701 h mix from the originals (USC, islomov ×3, uzbekvoice-filtered, Common Voice uz). Text must match GigaAM's 71-character alphabet: lowercase, `ʻ`→`'`, numbers as words (`uz_text.py`), no ö/ğ/ş/ç.
- Keep out of training: FLEURS (our test), `test_set/`, and the spontaneous benchmark's source rows. Check YouTube overlap.
- Risks: forgetting Russian (code-switching), so use a low LR or mix in some Russian. GigaAM probably already saw Common Voice uz, so expect the biggest gains on conversational/news speech.
- Cheaper add-on: a KenLM n-gram LM with CTC beam search (pyctcdecode), worth ~1–2 points without GPU training.

⚠️ **New official Uzbek Latin alphabet (law of 21 Oct 2026):** `oʻ`→`ö`, `gʻ`→`ğ`, `sh`→`ş`, `ch`→`ç` (per the openlimi card). Our models, prompts and other datasets use the **old** alphabet. **Decision needed:** which alphabet the agent uses. Never mix both in one training set without converting.

⚠️ **Overlap warning (from the benchmark authors):** the same YouTube recordings appear in several Uzbek datasets. Check for test/train overlap before trusting scores.

---

## Licenses of the models we use (checked on Hugging Face, 2026-11-27)

All are free to download and run. There are no per-use fees; the only cost is the server. But "free" ≠ "any use allowed":

| Model | License | Commercial use? |
|---|---|---|
| ai-sage/GigaAM-Multilingual | MIT | ✅ Yes |
| openai/whisper-* | MIT | ✅ Yes |
| navai-uz/whisper-*, Abduqayum call-center, Kotib/uzbek_stt_v1 | Apache-2.0 | ✅ Yes |
| google/gemma-4-* (12B, E4B, 26B-A4B) | Apache-2.0 | ✅ Yes |
| Qwen/Qwen2.5-7B-Instruct | Apache-2.0 | ✅ Yes |
| uzlm/alloma-* | Llama 3.1 Community License | ✅ Yes, with conditions: "Built with Llama" attribution, acceptable-use policy, special license above 700M monthly users |
| **facebook/mms-tts-uzb** | **CC-BY-NC-4.0** | ❌ **Non-commercial only.** Fine for testing, but it can't ship in a commercial product |
| uzlm/sayro-tts-1.7B | other (gated) | ❓ Read its license before use |
| rustam1221/uzbek-asr-gigaam | other (gated) | ❓ Read its license before use |

Training datasets also have their own licenses (see the dataset table). A model fine-tuned on non-commercial data may inherit restrictions.

---

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-26 | Fine-tune existing models instead of training from scratch | One L4 can't pretrain an LLM, and there isn't enough Uzbek text for it |
| 2026-11-26 | Pipeline = ASR → LLM (kronk) → TTS | Each part can be fine-tuned separately on the L4 |
| 2026-11-26 | Store LoRA adapters on the Hugging Face Hub, not GitHub | GitHub's 100 MB file limit and history bloat; saves server disk |
| 2026-11-26 | Use the Latin script internally; convert to Cyrillic only for MMS TTS | MMS Uzbek TTS reads Cyrillic only |
| 2026-11-26 | ASR fine-tuning (Phase 2) is required | Stock Whisper WER 147%: writes Uzbek in Kazakh-style Cyrillic, plus ~5% hallucination loops |
| 2026-11-27 | ASR front-runners: **Kotib/uzbek_stt_v1** (best FLEURS, 6.3%) and **GigaAM** (8.1%, 6× faster). Final pick after own recordings | Accuracy vs speed trade-off; FLEURS alone isn't decisive |
| 2026-11-27 | ~~GigaAM-Multilingual large_ctc is the ASR front-runner~~ (superseded by the line above) | Best FLEURS: 8.1% fair WER, 1.4% CER, 6× faster than navai-medium, MIT |
| 2026-11-27 | Always pass `num_beams` to Whisper (default 5, `--beams 1` = greedy) | transformers silently defaulted to 5, so the "greedy" vs "beam5" comparison was invalid |
| 2026-11-27 | Add GigaAM-Multilingual as an ASR candidate, with its remote code pinned to reviewed commits | MIT license, reported 7.3% FLEURS WER, CTC is fast. `trust_remote_code` runs repo code on our server, so pinning prevents unreviewed updates |
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
