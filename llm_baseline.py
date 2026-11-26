"""Step 1.2 - LLM baseline: send Uzbek test prompts to a model served by kronk.

kronk exposes an OpenAI-compatible API, so this works with any such server.
Run it once per candidate model, then open the results file and score each answer 1-5.

Usage:
    python llm_baseline.py --base-url http://localhost:11435/v1
    python llm_baseline.py --base-url http://localhost:11435/v1 --model google/gemma-4-12b-it-qat-q4_0
(or set LLM_BASE_URL / LLM_MODEL environment variables)
"""
import argparse
import os
import re
import time

import requests

DEFAULT_SYSTEM = (
    "Siz foydali ovozli yordamchisiz. Faqat o'zbek tilida (lotin yozuvida), "
    "qisqa va aniq javob bering. Javoblaringiz ovoz orqali o'qiladi, shuning uchun "
    "ro'yxat, jadval va maxsus belgilardan foydalanmang."
)

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
RUSSIAN_ONLY = re.compile(r"[ыщьЫЩЬ]")  # letters Uzbek Cyrillic basically never uses
COMMON_EN = {"the", "and", "is", "are", "you", "your", "with", "for", "this", "that"}


def flags(text: str) -> str:
    out = []
    if CYRILLIC.search(text):
        out.append("CYRILLIC")
    if RUSSIAN_ONLY.search(text):
        out.append("RUSSIAN?")
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if words and sum(w in COMMON_EN for w in words) / len(words) > 0.05:
        out.append("ENGLISH?")
    if re.search(r"^\s*([-*•]|\d+\.)\s", text, re.M) or "**" in text or "#" in text:
        out.append("MARKDOWN")
    return ", ".join(out) or "-"


def pick_model(base_url: str) -> str:
    r = requests.get(f"{base_url}/models", timeout=30)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data", [])]
    # kronk also serves models from other projects, so never guess - make the user choose.
    listing = "\n  ".join(models) or "(none)"
    raise SystemExit(f"Please choose a model with --model. Models on the server:\n  {listing}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:11435/v1"))
    p.add_argument("--model", default=os.getenv("LLM_MODEL"))
    p.add_argument("--prompts", default="prompts_uz.txt")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--max-tokens", type=int, default=300)
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    model = args.model or pick_model(base)
    with open(args.prompts, encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    os.makedirs("results", exist_ok=True)
    safe = re.sub(r"[^\w.-]", "_", model)
    out_path = f"results/llm_{safe}.md"
    total_time = total_tokens = 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# LLM baseline: {model}\n\n")
        f.write("Score each answer 1-5 (grammar / stays in Uzbek / useful).\n\n")
        for i, prompt in enumerate(prompts, 1):
            t0 = time.time()
            r = requests.post(
                f"{base}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": args.system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.3,
                },
                timeout=300,
            )
            dt = time.time() - t0
            r.raise_for_status()
            data = r.json()
            answer = data["choices"][0]["message"]["content"].strip()
            ntok = (data.get("usage") or {}).get("completion_tokens", 0)
            total_time += dt
            total_tokens += ntok

            print(f"\n[{i}/{len(prompts)}] {prompt}\n-> {answer}\n   ({dt:.1f}s, flags: {flags(answer)})")
            f.write(f"## {i}. {prompt}\n\n{answer}\n\n")
            f.write(f"_time: {dt:.1f}s, tokens: {ntok}, flags: {flags(answer)}_  \n**Score:** __/5\n\n")

        speed = f"{total_tokens / total_time:.1f} tok/s" if total_tokens else "n/a"
        summary = f"Average {total_time / len(prompts):.1f}s per answer, {speed}"
        f.write(f"---\n{summary}\n")

    print(f"\n{summary}\nResults saved to {out_path} - open it and fill in the scores.")


if __name__ == "__main__":
    main()
