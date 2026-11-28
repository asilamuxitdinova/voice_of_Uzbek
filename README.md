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
| `run_test_set.sh` | **Run on the laptop.** Uploads only `test_audios/test*` to the server, runs the ASR models on them, and copies the results back |
| `prepare_data.py` | Phase 2: builds the GigaAM training set (FLAC + TSV manifests) from open Uzbek datasets |
| `setup_gigaam_train.sh` / `train_gigaam.py` | Phase 2: installs the official GigaAM trainer and fine-tunes GigaAM |
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

**1.1b Ready-made Uzbek ASR models.** This runs all of them and prints a WER/CER summary. Models are 0.9–2.4 GB each:
- **`ai-sage/GigaAM-Multilingual`**: a Conformer CTC model, MIT license. Its authors report 7.3% WER on FLEURS Uzbek. Add `@ctc` for the smaller 220M version (the default is `large_ctc`, 600M). It needs three extra packages (once): `pip install hydra-core omegaconf sentencepiece`
- `Kotib/uzbek_stt_v1`, `navai-uz/whisper-medium-uzbek`, `navai-uz/whisper-small-uzbek`, `Abduqayum/whisper-uzbek-medium-callcenter`: Whisper fine-tunes

To test GigaAM alone:
```bash
python asr_baseline.py --model ai-sage/GigaAM-Multilingual
```

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
**Shortcut from the laptop:** this uploads only the `test1…test20` recordings (and their `.txt` files) to a separate `test_set/` folder on the server, runs GigaAM, call-center, navai-medium and Kotib on them, and copies the results back to `results/`. Push your code first.
```bash
bash run_test_set.sh
```
Name models to run only those, for example `bash run_test_set.sh ai-sage/GigaAM-Multilingual`. Use `DRY_RUN=1` to only print what it would do.

Open the results file and fill in `MEANING OK: yes/no` for each recording: would the request still be understood correctly? That's the number that matters for the agent.

**Scoring and options:**
- Every run reports **WER (fair)** alongside the original WER. Fair scoring writes numbers as words (`15%` = `o'n besh foiz`), converts Cyrillic to Latin, and expands `km` → `kilometr`, so differences in spelling convention aren't counted as errors. Use the fair score from now on.
- Whisper models use **5-beam search by default** (that's also what `transformers` does silently). Add `--beams 1` for greedy decoding: faster, sometimes slightly less accurate. Results are saved under a separate `_beam1` name, so you can compare.
- Recordings longer than 30 s are handled automatically. GigaAM's limit is 25 s, so longer audio is split at quiet moments.
- GigaAM runs the model's own code from Hugging Face (`trust_remote_code`). The scripts pin it to the versions reviewed on 2026-11-27 (`GIGAAM_REVISIONS` in `asr_baseline.py`).
- `--beams` only applies to Whisper models. GigaAM always uses greedy CTC decoding.

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
Options: `--asr-model navai-uz/whisper-medium-uzbek` or `--asr-model ai-sage/GigaAM-Multilingual` to choose the ASR model, and `--apst` when the LLM is alloma.
Listen to `results/answer.wav` and note the timing line (`ASR | LLM | TTS | TOTAL`).

### ✅ Checkpoint 1: report back
- WER % and a few REF/HYP lines from 1.1
- Best LLM from 1.2, with its average score and a couple of example answers
- How the TTS sounds (clear / robotic / hard to understand)
- The timing line from 1.4
- What the agent is for (phone support, booking, assistant…)

---

## Phase 2: Fine-tune GigaAM on Uzbek speech

All commands run **on the server**, in `~/voice_of_Uzbek` with `.venv` active. Long steps belong inside `tmux`, so an SSH disconnect doesn't kill them.

### 2.1 Setup (once)
```bash
bash setup_gigaam_train.sh
```
This clones the official GigaAM training code (MIT) at a reviewed commit into `third_party/GigaAM` and installs Lightning, TensorBoard and editdistance. It must end with `OK: gigaam ... | CUDA True | bf16 True`.

### 2.2 Build the dataset (~1–3 hours, ~37 GB)
```bash
tmux new -s prep
```
```bash
python prepare_data.py
```
- Downloads **one parquet file at a time**, converts the clips to 16 kHz mono FLAC (0.059 GB per hour of audio), cleans the transcripts into GigaAM's alphabet (lowercase, `'` apostrophe, numbers as words, no Cyrillic), and deletes each download before fetching the next.
- Sources: ISSAI USC (via `murodbek/uzbek-speech-corpus`), islomov news / IT / podcasts, uzbekvoice-filtered. All have commercial-friendly licenses. Default caps come to about 620 h. Change them with `--max-hours uzbekvoice=100 news=50`.
- **Resumable:** if it stops (disconnect, or the disk guard `--min-free-gb 25`), run the same command again and it continues where it stopped.
- Output: `data/gigaam_uz/train.tsv`, `val.tsv` (~1% held out) and `test_usc.tsv` (USC's official test split).
- FLEURS, `test_audios/` and `test_set/` are **never** used for training.

### 2.3 Trial run first (~10–20 min)
```bash
python train_gigaam.py --trial
```
100 steps, with validation at step 50. It shows **whether batch size 8 × 30 s clips fits in 24 GB** and how fast the training steps are. Send the output to Claude to pick the number of steps for the real run.
If it fails with *CUDA out of memory*: `python train_gigaam.py --trial --batch-size 4 --accumulate 8`.

### 2.4 Real run
```bash
tmux new -s train
```
```bash
python train_gigaam.py --max-steps 6000
```
- Validates and saves a checkpoint every 1000 steps. Only the best 2 are kept, at ~2.4 GB each (weights only).
- Defaults: GigaAM `multilingual_large_ctc` (600M), bf16, activation checkpointing, LR 2e-5 (low, to adapt to Uzbek without forgetting Russian loanwords), effective batch 32.
- Watch progress in the log, or with TensorBoard: `tensorboard --logdir checkpoints`, then on the laptop `ssh -L 6006:localhost:6006 kronk` and open http://localhost:6006.
- Continue a stopped run: `python train_gigaam.py --max-steps 6000 --resume checkpoints/models/<exp>/<file>.ckpt`. Weights only, so the optimizer restarts.

### 2.5 Evaluate the fine-tuned model
```bash
python asr_baseline.py --model checkpoints/models/<exp>/<best>.ckpt
```
```bash
python asr_baseline.py --model checkpoints/models/<exp>/<best>.ckpt --audio-dir test_set
```
Same FLEURS / fair-WER scoring as the baseline, so the result compares directly with GigaAM (8.1%) and Kotib (6.3%).

## Next phases
3. QLoRA fine-tune the best LLM on domain dialogs → GGUF → kronk
4. TTS: a commercially usable Uzbek voice (MMS is non-commercial)
5. Real-time agent: VAD, streaming, and a phone/Telegram/web channel

## Troubleshooting
- **`CUDA is NOT available`**: run `nvidia-smi`. If it works, reinstall torch: `pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
- **FLEURS fails to download**: check `hf auth login` and internet access. Files are cached in `~/.cache/huggingface`, and the extracted audio goes to `data/`
- **`Connection refused` in llm_baseline.py**: kronk isn't running, or the port in `--base-url` is wrong
- **Out of GPU memory**: stop other processes (`nvidia-smi`), or use a smaller LLM quantization
- **`convert_alloma.sh` says "pre-tokenizer was not recognized"**: the script's patch didn't apply, maybe because llama.cpp changed. Paste the output to Claude
- **`setup_sayro.sh` fails installing `qwen-tts`**: probably Python 3.14. Install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `rm -rf .venv-tts && bash setup_sayro.sh`
- **`train_gigaam.py`: "CUDA out of memory"**: halve `--batch-size` and double `--accumulate`, which keeps the same effective batch
- **`prepare_data.py` stopped with "only N GB free"**: free space (see `disk_report.sh`) or lower the caps, then rerun the same command. It resumes
- **`tts_sayro.py`: 401 / "gated repo"**: access to sayro-tts isn't approved yet, or `hf auth login` wasn't run inside `.venv-tts`
