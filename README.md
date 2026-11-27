# Voice of Uzbek

An Uzbek voice agent built from open models: **ASR (Whisper) → LLM (served by kronk) → TTS (MMS)**.
It runs on a single NVIDIA L4 (24 GB).

This repo currently covers **Phase 0 (setup)** and **Phase 1 (baseline)**. The baseline tests stock models before any training, to find which part is weakest.

| File | What it does |
|---|---|
| `setup.sh` | Installs system packages, Python venv, PyTorch (CUDA), and libraries, then checks the GPU |
| `asr_baseline.py` | Measures Whisper's Word Error Rate on the FLEURS Uzbek test set |
| `llm_baseline.py` | Sends `prompts_uz.txt` to your kronk model and saves the answers for scoring |
| `tts_baseline.py` | Generates Uzbek speech with Meta MMS (Latin text is converted to Cyrillic automatically) |
| `voice_pipeline.py` | End-to-end: your .wav question → text → LLM answer → spoken .wav, with timings |
| `translit.py` | Uzbek Latin ↔ Cyrillic converter (used by TTS and scoring) |
| `uz_text.py` | Text normalization for fair scoring: numbers → words, Cyrillic → Latin, `km` → `kilometr` |
| `prompts_uz.txt` | 20 Uzbek test prompts. Edit them to match your real use case |
| `asr_compare.sh` | Runs `asr_baseline.py` on ready-made Uzbek Whisper fine-tunes and prints a summary |
| `convert_alloma.sh` | Converts `uzlm/alloma-8B-Instruct` to a Q4_K_M GGUF and registers it in kronk |
| `setup_sayro.sh` | Creates a separate `.venv-tts` for `uzlm/sayro-tts-1.7B` (Qwen3-TTS) |
| `tts_sayro.py` | Generates Uzbek speech with sayro-tts, for comparison with MMS |

All outputs go to `results/`. **All numbers, findings and decisions are summarized in [PROGRESS.md](PROGRESS.md).**

---

## Phase 0: Setup (once)

```bash
git clone <your-repo-url> Voice_of_Uzbek
cd Voice_of_Uzbek
bash setup.sh
source .venv/bin/activate
hf auth login
```

`setup.sh` must end with `OK, GPU: NVIDIA L4`.
For `hf auth login`, create a token at https://huggingface.co/settings/tokens (read access is enough).

> Every new terminal: `cd Voice_of_Uzbek && source .venv/bin/activate`

---

## Phase 1: Baseline

### 1.1 ASR (speech → text)
```bash
python asr_baseline.py
```
The first run downloads Whisper (~1.6 GB) and the FLEURS Uzbek test audio (~520 MB). The result is a **WER %** (lower is better) and is saved to `results/asr_*.txt`.
Optionally compare a smaller model: `python asr_baseline.py --model openai/whisper-small`

**1.1b Ready-made Uzbek Whisper models.** Each one is ~1.5 GB. This runs all of them and prints a WER/CER summary:
```bash
bash asr_compare.sh
```
⚠️ The `navai-uz` models were trained with FLEURS data, so their score here may look better than real use.

**1.1c Your own recordings (the honest test).** Put recordings in `test_audios/`. For each one, add a `.txt` file **with the same name** containing exactly what was said:
```
test_audios/
  savol1.m4a
  savol1.txt     <- "Ertaga Toshkentda havo qanday bo'ladi?"
  savol2.m4a
  savol2.txt
```
Then run (the model loads once for the whole folder):
```bash
python asr_baseline.py --model navai-uz/whisper-medium-uzbek --audio-dir test_audios
```
Or compare all models on your recordings:
```bash
bash asr_compare.sh 0 --audio-dir test_audios
```
Open the results file and fill in `MEANING OK: yes/no` for each recording: would the request still be understood correctly? That's the number that matters for the agent.

**Scoring and options:**
- Every run reports **WER (fair)** alongside the original WER. Fair scoring writes numbers as words (`15%` = `o'n besh foiz`), converts Cyrillic to Latin, and expands `km` → `kilometr`, so differences in spelling convention aren't counted as errors. Use the fair score from now on.
- Add `--beams 5` to any command for beam search: usually more accurate, slightly slower. Results are saved under a separate `_beam5` name, so you can compare.
- Recordings longer than 30 s are handled automatically.

### 1.2 LLM (kronk)
1. Make sure kronk is running (`tmux attach -t kronk`). Candidates are already on the server: `google/gemma-4-12b-it-qat-q4_0`, `google/gemma-4-E4B_q4_0-it`, `google/gemma-4-26B_q4_0-it`, and Qwen2.5-7B-Instruct.
2. Run the script once per model:
   ```bash
   python llm_baseline.py --model google/gemma-4-12b-it-qat-q4_0
   ```
   kronk listens on port 11435 (the default). `--model` is required, and running without it prints the model list.
3. Open `results/llm_<model>.md` and fill in a score of 1–5 for each answer.
4. Repeat for each candidate model.

The `flags` column automatically marks answers that slipped into Cyrillic, Russian, English, or markdown.

**1.2b Uzbek-specific LLM: `uzlm/alloma-8B-Instruct`.** It's only published in Hugging Face format, so convert it first. This takes ~20–40 min and needs ~35 GB of free disk temporarily; only the ~5 GB GGUF is kept:
```bash
bash convert_alloma.sh
```
```bash
kronk model index --local && kronk model list --local
```
Restart `kronk server start` in tmux if the model doesn't appear in `python llm_baseline.py`. Then test it **with `--apst`**: alloma's tokenizer spells Uzbek apostrophes as `APST`, and the script converts them back in the answers.
```bash
python llm_baseline.py --model <alloma-id-from-list> --apst
```
Running it once without `--apst` too shows how much the APST step matters.

### 1.3 TTS (text → speech)
```bash
python tts_baseline.py
```
Listen to `results/tts_1.wav` … `tts_4.wav`. You can also try your own text:
```bash
python tts_baseline.py --text "Salom, bugun qanday yordam kerak?"
```

**1.3b Uzbek-specific TTS: `uzlm/sayro-tts-1.7B`.** The model is **gated**: open its Hugging Face page while logged in, request access, and read the license first, especially whether it allows commercial use. It runs in its own environment, because `qwen-tts` pins an older `transformers`:
```bash
bash setup_sayro.sh
```
```bash
source .venv-tts/bin/activate && python tts_sayro.py
```
If it says the model needs a reference voice, record 5–15 s of clear Uzbek speech and run:
```bash
python tts_sayro.py --ref-audio ref.wav --ref-text "exact words you said"
```
Compare `results/sayro_*.wav` with the MMS files. Run `deactivate` afterwards to go back to the main `.venv`.

### 1.4 End-to-end
Record a short Uzbek question (a phone voice memo is fine), copy it to the server, then:
```bash
python voice_pipeline.py --audio question.wav --model <best-model>
```
Options: `--asr-model navai-uz/whisper-medium-uzbek` to use an Uzbek Whisper, and `--apst` when the LLM is alloma.
Listen to `results/answer.wav` and note the timing line (`ASR | LLM | TTS | TOTAL`).

### ✅ Checkpoint 1: report back
- WER % and a few REF/HYP lines from 1.1
- Best LLM from 1.2, with its average score and a couple of example answers
- How the TTS sounds (clear / robotic / hard to understand)
- The timing line from 1.4
- What the agent is for (phone support, booking, assistant…)

---

## Next phases (scripts will be added after Checkpoint 1)
2. Fine-tune Whisper on Uzbek speech (Common Voice, FLEURS train, ISSAI USC, your own recordings)
3. QLoRA fine-tune the best LLM on domain dialogs → GGUF → kronk
4. Fine-tune TTS on a single Uzbek speaker
5. Real-time agent: VAD, streaming, and a phone/Telegram/web channel

## Troubleshooting
- **`CUDA is NOT available`**: run `nvidia-smi`. If it works, reinstall torch: `pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
- **FLEURS fails to download**: check `hf auth login` and internet access. Files are cached in `~/.cache/huggingface`, and the extracted audio goes to `data/`
- **`Connection refused` in llm_baseline.py**: kronk isn't running, or the port in `--base-url` is wrong
- **Out of GPU memory**: stop other processes (`nvidia-smi`), or use a smaller LLM quantization
- **`convert_alloma.sh` says "pre-tokenizer was not recognized"**: the script's patch didn't apply, maybe because llama.cpp changed. Paste the output to Claude
- **`setup_sayro.sh` fails installing `qwen-tts`**: probably Python 3.14. Install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `rm -rf .venv-tts && bash setup_sayro.sh`
- **`tts_sayro.py`: 401 / "gated repo"**: access to sayro-tts isn't approved yet, or `hf auth login` wasn't run inside `.venv-tts`
