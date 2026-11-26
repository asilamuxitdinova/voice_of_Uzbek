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
| `translit.py` | Uzbek Latin → Cyrillic converter (used by TTS) |
| `prompts_uz.txt` | 20 Uzbek test prompts. Edit them to match your real use case |

All outputs go to `results/`.

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

### 1.2 LLM (kronk)
1. Download 2–3 candidate GGUF models (Q4_K_M) and load one in kronk, for example a Qwen3 8B instruct, a Gemma 3 12B instruct, and an Uzbek fine-tune if one exists (search Hugging Face for `behbudiy`).
2. Run the script with **your kronk address** (replace the port):
   ```bash
   python llm_baseline.py --base-url http://localhost:8080/v1
   ```
   If kronk serves several models, add `--model <name>`.
3. Open `results/llm_<model>.md` and fill in a score of 1–5 for each answer.
4. Repeat for each candidate model.

The `flags` column automatically marks answers that slipped into Cyrillic, Russian, English, or markdown.

### 1.3 TTS (text → speech)
```bash
python tts_baseline.py
```
Listen to `results/tts_1.wav` … `tts_4.wav`. You can also try your own text:
```bash
python tts_baseline.py --text "Salom, bugun qanday yordam kerak?"
```

### 1.4 End-to-end
Record a short Uzbek question (a phone voice memo is fine), copy it to the server, then:
```bash
python voice_pipeline.py --audio question.wav --base-url http://localhost:8080/v1
```
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
