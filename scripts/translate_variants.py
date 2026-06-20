#!/usr/bin/env python3
"""
translate_variants.py
--------------------
Command-line tool to generate translated variants using deep-translator (Google Translate, free).
Reads seed_prompts.csv and produces translated variants for target languages.

Usage:
    python translate_variants.py --input data/seed_prompts.csv --output data/benchmark_prompts.csv --languages zu,af,ss,ts

Languages:
    zu = isiZulu, af = Afrikaans, ss = Swati, ts = Tsonga
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator

# ── Language mapping ─────────────────────────────────────────────────────────
LANG_MAP = {
    "zu": {"name": "isiZulu", "google_code": "zu"},
    "af": {"name": "Afrikaans", "google_code": "af"},
    "xh": {"name": "isiXhosa", "google_code": "ss"},      # isiXhosa
    "ts": {"name": "Tsonga", "google_code": "ts"},     # Xitsonga
}

# ── Translation function ─────────────────────────────────────────────────────
def translate_text(text, dest_lang, max_retries=3):
    """Translate text to destination language with retry logic."""
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source='en', target=dest_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"  Retry {attempt + 1}/{max_retries}: {e}")
            time.sleep(1)
    return f"[TRANSLATION_FAILED: {dest_lang}]"


def main():
    parser = argparse.ArgumentParser(description="Generate translated variants for AfriGuard-SA")
    parser.add_argument("--input", required=True, help="Path to seed_prompts.csv")
    parser.add_argument("--output", default="data/benchmark_prompts.csv", help="Output CSV path")
    parser.add_argument("--languages", default="zu,af,ss,ts", help="Comma-separated language codes")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Parse languages
    languages = [l.strip() for l in args.languages.split(",")]
    invalid = [l for l in languages if l not in LANG_MAP]
    if invalid:
        print(f"ERROR: Invalid language codes: {invalid}", file=sys.stderr)
        print(f"Valid codes: {', '.join(LANG_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    # Load seeds
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    seeds = pd.read_csv(args.input)
    print(f"Loaded {len(seeds)} seed prompts.")

    # Generate variants
    rows = []
    total_variants = len(seeds) * len(languages)
    completed = 0

    for _, seed in seeds.iterrows():
        seed_id = seed["prompt_id"]
        english = seed["english_prompt"]
        category = seed["harm_category"]

        # Always include English baseline
        rows.append({
            "prompt_id": f"{seed_id}_en",
            "seed_id": seed_id,
            "language": "en",
            "language_name": "English",
            "harm_category": category,
            "prompt": english,
        })

        # Generate translated variants
        for lang_code in languages:
            completed += 1
            print(f"[{completed}/{total_variants}] Translating seed {seed_id} to {LANG_MAP[lang_code]['name']}...", end=" ")

            translated = translate_text(english, LANG_MAP[lang_code]["google_code"])
            status = "OK" if not translated.startswith("[TRANSLATION_FAILED") else "FAILED"
            print(status)

            rows.append({
                "prompt_id": f"{seed_id}_{lang_code}",
                "seed_id": seed_id,
                "language": lang_code,
                "language_name": LANG_MAP[lang_code]["name"],
                "harm_category": category,
                "prompt": translated,
            })

            time.sleep(args.delay)

    # Save
    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(df)} prompts.")
    print(f"Saved to: {output_path}")
    print(f"\nLanguage distribution:")
    print(df["language_name"].value_counts())
    print(f"{'='*60}")

    # Report failures
    failures = df[df["prompt"].str.startswith("[TRANSLATION_FAILED", na=False)]
    if len(failures) > 0:
        print(f"\nWARNING: {len(failures)} translations failed. Re-run to retry.")


if __name__ == "__main__":
    main()