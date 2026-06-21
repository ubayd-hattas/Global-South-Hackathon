"""
pipeline_utils.py
------------------
Shared helpers for every script in scripts/. Import from this file instead
of repeating path/dotenv/retry logic in each script.

WHY THIS FILE EXISTS (read this if API keys "aren't loading"):
python-dotenv's load_dotenv() with no arguments searches for a .env file
starting from the current working directory. That means the SAME script
can succeed or fail to find your .env depending on whether you ran it as

    python scripts/run_models.py        (cwd = repo root)
    cd scripts && python run_models.py  (cwd = scripts/)

...or ran it by clicking "Run" in an IDE, which often sets cwd to the
file's own folder. This is the #1 cause of "my API key isn't loading"
bugs and it is NOT an API problem -- it's a path problem.

The fix: compute the project root from THIS file's own location on disk
(which never changes) instead of from the current working directory
(which changes depending on how you launch the script), then load .env
from that exact path, explicitly.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup (robust on Windows, macOS, Linux, any IDE, any working directory)
# ---------------------------------------------------------------------------
# scripts/pipeline_utils.py -> parent (scripts/) -> parent (repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEED_PROMPTS_CSV = DATA_DIR / "seed_prompts.csv"
BENCHMARK_PROMPTS_CSV = DATA_DIR / "benchmark_prompts.csv"
RESPONSES_CSV = RESULTS_DIR / "responses.csv"
EVALUATION_CSV = RESULTS_DIR / "evaluation.csv"

ENV_PATH = PROJECT_ROOT / ".env"

# Load .env explicitly from the project root. override=False means real
# environment variables (e.g. set in CI, or exported in your shell) win
# over the .env file, which is the behavior most people expect.
load_dotenv(dotenv_path=ENV_PATH, override=False)


def ensure_output_dirs() -> None:
    """Create data/results/ and figures/ if they don't exist yet."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_env(var_name: str) -> str:
    """
    Fetch an environment variable or exit with a clear, actionable error.
    Use this for required API keys instead of os.getenv(...) directly.
    """
    value = os.getenv(var_name)
    if value is not None:
        value = value.strip()
    if not value:
        print(
            f"\n[CONFIG ERROR] {var_name} is missing or empty.\n"
            f"Expected to find it in: {ENV_PATH}\n"
            "\n"
            "Checklist:\n"
            f"  1. Does a file named exactly '.env' exist at the repo root\n"
            f"     ({PROJECT_ROOT})? Not inside data/ or scripts/.\n"
            f"  2. Does it contain a line that looks exactly like:\n"
            f"       {var_name}=your_actual_key_here\n"
            "     (no quotes, no spaces around the '=', no trailing spaces)\n"
            "  3. Did you save the file after editing it?\n"
            "  4. On Windows, did Notepad save it as '.env.txt' by accident?\n"
            "     Check 'View > File name extensions' in File Explorer.\n"
            "  5. Is the key still active on the provider's dashboard\n"
            "     (not deleted, not expired, not rate-limited to zero)?\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


def get_env(var_name: str, default: str) -> str:
    """Fetch an optional environment variable, falling back to a default."""
    value = os.getenv(var_name)
    if value is None or not value.strip():
        return default
    return value.strip()


def retry_with_backoff(fn, *, attempts: int = 4, base_delay: float = 2.0, what: str = "API call"):
    """
    Call fn() and retry with exponential backoff (2s, 4s, 8s, ...) on failure.
    Returns fn()'s return value on success, or raises the last exception
    after all attempts are exhausted.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - intentionally broad for a hackathon retry helper
            last_err = exc
            if attempt == attempts:
                break
            wait = base_delay * (2 ** (attempt - 1))
            print(f"    [retry {attempt}/{attempts - 1}] {what} failed: {exc}. Waiting {wait:.0f}s...")
    raise last_err

def call_model(model_name: str, prompt: str, system_prompt: str = None) -> str:
    """Call OpenRouter API with backoff."""
    import requests
    api_key = require_env("OPENROUTER_API_KEY")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    def _call():
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/mzansi-codeswitching-safety",
                "X-Title": "Mzansi Code-Switching Safety",
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data or not data["choices"]:
            raise ValueError(f"No choices in OpenRouter response: {data}")
        return data["choices"][0]["message"]["content"].strip()

    return retry_with_backoff(_call, attempts=4, base_delay=2.0, what=f"OpenRouter ({model_name})")
