"""
run_models.py
-------------
Sends every prompt in data/benchmark_prompts.csv to three models:
  - Gemini (Google)
  - Llama via Groq
  - A free model via OpenRouter

Appends results to data/results/responses.csv so you can re-run
safely without losing progress (skips already-processed prompt_ids).

Run from the repository root:
    python scripts/run_models.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
import google.generativeai as genai
from groq import Groq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    BENCHMARK_PROMPTS_CSV,
    RESPONSES_CSV,
    ensure_output_dirs,
    get_env,
    require_env,
    retry_with_backoff,
)

# ── Model config ──────────────────────────────────────────────────────────────
GEMINI_MODEL      = get_env("GEMINI_MODEL",      "gemini-1.5-flash")
GROQ_MODEL        = get_env("GROQ_MODEL",         "llama3-8b-8192")
OPENROUTER_MODEL  = get_env("OPENROUTER_MODEL",   "mistralai/mistral-7b-instruct:free")

SYSTEM_PROMPT = "You are a helpful assistant."

# ── Provider callables ────────────────────────────────────────────────────────

def call_gemini(model, prompt: str) -> str:
    def _call():
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        # Gemini can block with no candidates
        if not resp.candidates:
            return "[BLOCKED_NO_CANDIDATES]"
        return resp.text.strip()
    return retry_with_backoff(_call, attempts=4, base_delay=2.0, what="Gemini")


def call_groq(client: Groq, prompt: str) -> str:
    def _call():
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return completion.choices[0].message.content.strip()
    return retry_with_backoff(_call, attempts=4, base_delay=2.0, what="Groq")


def call_openrouter(api_key: str, prompt: str) -> str:
    def _call():
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # OpenRouter recommends these headers for free-tier identification
                "HTTP-Referer": "https://github.com/mzansi-codeswitching-safety",
                "X-Title": "Mzansi Code-Switching Safety",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    return retry_with_backoff(_call, attempts=4, base_delay=2.0, what="OpenRouter")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()

    # ── Auth ──────────────────────────────────────────────────────────────────
    gemini_key       = require_env("GEMINI_API_KEY")
    groq_key         = require_env("GROQ_API_KEY")
    openrouter_key   = require_env("OPENROUTER_API_KEY")

    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    groq_client  = Groq(api_key=groq_key)

    print(f"[run_models] Gemini:     {GEMINI_MODEL}")
    print(f"[run_models] Groq:       {GROQ_MODEL}")
    print(f"[run_models] OpenRouter: {OPENROUTER_MODEL}")

    # ── Load benchmark prompts ────────────────────────────────────────────────
    if not BENCHMARK_PROMPTS_CSV.exists():
        print(f"[ERROR] benchmark_prompts.csv not found. Run generate_variants.py first.", file=sys.stderr)
        sys.exit(1)

    prompts_df = pd.read_csv(BENCHMARK_PROMPTS_CSV)
    print(f"[run_models] Loaded {len(prompts_df)} benchmark prompts.")

    # ── Load existing results (resume support) ────────────────────────────────
    if RESPONSES_CSV.exists():
        existing = pd.read_csv(RESPONSES_CSV)
        done_keys = set(zip(existing["prompt_id"], existing["model"]))
        print(f"[run_models] Resuming — {len(existing)} rows already saved, skipping them.")
    else:
        existing = pd.DataFrame()
        done_keys = set()

    # ── Run models ────────────────────────────────────────────────────────────
    models = {
        "gemini":      lambda p: call_gemini(gemini_model, p),
        "groq_llama":  lambda p: call_groq(groq_client, p),
        "openrouter":  lambda p: call_openrouter(openrouter_key, p),
    }

    new_rows = []
    tasks = [
        (row["prompt_id"], row["prompt"], row["language"], row["harm_category"], row["seed_id"], model_name, model_fn)
        for _, row in prompts_df.iterrows()
        for model_name, model_fn in models.items()
    ]

    for prompt_id, prompt, language, category, seed_id, model_name, model_fn in tqdm(tasks, desc="Running models"):
        if (prompt_id, model_name) in done_keys:
            continue

        try:
            response_text = model_fn(prompt)
            error = ""
        except Exception as exc:
            response_text = f"[API_ERROR: {exc}]"
            error = str(exc)

        new_rows.append({
            "prompt_id":  prompt_id,
            "seed_id":    seed_id,
            "language":   language,
            "harm_category": category,
            "model":      model_name,
            "prompt":     prompt,
            "response":   response_text,
            "error":      error,
        })

        # Save every 10 new rows so a crash doesn't lose everything
        if len(new_rows) % 10 == 0:
            _flush(existing, new_rows)

        # Polite delay to stay within free-tier rate limits
        time.sleep(0.3)

    _flush(existing, new_rows)
    print(f"\n[run_models] Done. Results saved to: {RESPONSES_CSV}")


def _flush(existing: pd.DataFrame, new_rows: list):
    if not new_rows:
        return
    combined = pd.concat(
        [existing, pd.DataFrame(new_rows)],
        ignore_index=True,
    ) if not existing.empty else pd.DataFrame(new_rows)
    combined.to_csv(RESPONSES_CSV, index=False)


if __name__ == "__main__":
    main()
