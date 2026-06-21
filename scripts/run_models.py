"""
run_models.py
-------------
Sends every prompt in data/benchmark_prompts.csv to TARGET_MODELS via OpenRouter.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    BENCHMARK_PROMPTS_CSV,
    RESPONSES_CSV,
    ensure_output_dirs,
    call_model,
)

# ── Model config ──────────────────────────────────────────────────────────────
MODELS = {
    "kimi": "moonshotai/kimi-k2.6",
    "llama33": "meta-llama/llama-3.3-70b-instruct",
    "qwen3": "qwen/qwen3-32b",
    "gptoss": "openai/gpt-oss-20b"
}

SYSTEM_PROMPT = "You are a helpful assistant."

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()

    print(f"[run_models] Using models: {list(MODELS.keys())}")

    if not BENCHMARK_PROMPTS_CSV.exists():
        print(f"[ERROR] benchmark_prompts.csv not found. Run generate_variants.py first.", file=sys.stderr)
        sys.exit(1)

    prompts_df = pd.read_csv(BENCHMARK_PROMPTS_CSV)
    print(f"[run_models] Loaded {len(prompts_df)} benchmark prompts.")

    if RESPONSES_CSV.exists():
        existing = pd.read_csv(RESPONSES_CSV)
        done_keys = set(zip(existing["prompt_id"], existing["model"]))
        print(f"[run_models] Resuming — {len(existing)} rows already saved, skipping them.")
    else:
        existing = pd.DataFrame()
        done_keys = set()

    new_rows = []
    tasks = [
        (row["prompt_id"], row["prompt"], row["language"], row["harm_category"], row["seed_id"], model_key, model_id)
        for _, row in prompts_df.iterrows()
        for model_key, model_id in MODELS.items()
    ]

    for prompt_id, prompt, language, category, seed_id, model_key, model_id in tqdm(tasks, desc="Running models"):
        if (prompt_id, model_key) in done_keys:
            continue

        try:
            response_text = call_model(model_id, prompt, system_prompt=SYSTEM_PROMPT)
            error = ""
        except Exception as exc:
            response_text = f"[API_ERROR: {exc}]"
            error = str(exc)

        new_rows.append({
            "prompt_id":  prompt_id,
            "seed_id":    seed_id,
            "language":   language,
            "harm_category": category,
            "model":      model_key,
            "prompt":     prompt,
            "response":   response_text,
            "error":      error,
        })

        if len(new_rows) % 10 == 0:
            _flush(existing, new_rows)

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
