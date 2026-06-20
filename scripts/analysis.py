"""
analysis.py
-----------
Reads data/results/evaluation.csv and produces:
  - Summary statistics printed to the terminal
  - figures/asr_heatmap.png       (Attack Success Rate by language × model)
  - figures/asr_by_language.png   (ASR bar chart grouped by language)
  - figures/model_comparison.png  (overall refusal rate per model)

"Attack Success Rate" (ASR) = proportion of prompts labelled "compliance"
among all prompts that received a definitive label (refusal/compliance/partial).

Run from the repository root:
    python scripts/analysis.py
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    EVALUATION_CSV,
    FIGURES_DIR,
    ensure_output_dirs,
)

# ── Language display names ────────────────────────────────────────────────────
LANG_LABELS = {
    "en": "English",
    "zu": "isiZulu",
    "af": "Afrikaans",
    "ss": "Swati",
    "ts": "Tsonga",
}

MODEL_LABELS = {
    "gemini":     "Gemini 1.5 Flash",
    "groq_llama": "Llama 3 (Groq)",
    "openrouter": "Mistral 7B (OR)",
}

PALETTE = sns.color_palette("colorblind")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def load_eval() -> pd.DataFrame:
    if not EVALUATION_CSV.exists():
        print(f"[ERROR] evaluation.csv not found. Run judge.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(EVALUATION_CSV)
    # Drop rows we couldn't classify
    df = df[df["label"].isin(["refusal", "compliance", "partial"])].copy()
    print(f"[analysis] Loaded {len(df)} classified rows.")
    return df


def compute_asr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Attack Success Rate (compliance / total) per language × model."""
    df["is_compliance"] = (df["label"] == "compliance").astype(int)
    asr = (
        df.groupby(["language", "model"])["is_compliance"]
        .mean()
        .reset_index(name="asr")
    )
    asr["asr_pct"] = asr["asr"] * 100
    asr["language_label"] = asr["language"].map(LANG_LABELS).fillna(asr["language"])
    asr["model_label"]    = asr["model"].map(MODEL_LABELS).fillna(asr["model"])
    return asr


def plot_heatmap(asr: pd.DataFrame):
    pivot = asr.pivot(index="language_label", columns="model_label", values="asr_pct")
    # Sort rows: English first, then alphabetical
    order = ["English"] + sorted(r for r in pivot.index if r != "English")
    pivot = pivot.reindex([r for r in order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        linewidths=0.5,
        cbar_kws={"label": "ASR (%)"},
        ax=ax,
        vmin=0,
        vmax=100,
    )
    ax.set_title("Attack Success Rate (%) by Language × Model", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Language")
    plt.tight_layout()
    out = FIGURES_DIR / "asr_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[analysis] Saved: {out}")


def plot_asr_by_language(asr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    lang_order = ["English"] + sorted(l for l in asr["language_label"].unique() if l != "English")
    sns.barplot(
        data=asr,
        x="language_label",
        y="asr_pct",
        hue="model_label",
        order=lang_order,
        palette=PALETTE,
        ax=ax,
    )
    ax.set_title("Attack Success Rate (%) by Language", fontsize=13)
    ax.set_xlabel("Language")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    out = FIGURES_DIR / "asr_by_language.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[analysis] Saved: {out}")


def plot_model_comparison(df: pd.DataFrame):
    """Overall refusal rate per model (higher = safer)."""
    df2 = df.copy()
    df2["is_refusal"] = (df2["label"] == "refusal").astype(int)
    refusal_rates = (
        df2.groupby("model")["is_refusal"]
        .mean()
        .reset_index(name="refusal_rate")
    )
    refusal_rates["refusal_pct"]  = refusal_rates["refusal_rate"] * 100
    refusal_rates["model_label"]  = refusal_rates["model"].map(MODEL_LABELS).fillna(refusal_rates["model"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        refusal_rates["model_label"],
        refusal_rates["refusal_pct"],
        color=PALETTE[:len(refusal_rates)],
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, pct in zip(bars, refusal_rates["refusal_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Overall Refusal Rate by Model\n(higher = safer guardrails)", fontsize=12)
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_xlabel("")
    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[analysis] Saved: {out}")


def print_summary(df: pd.DataFrame, asr: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  MZANSI CODE-SWITCHING SAFETY — RESULTS SUMMARY")
    print("=" * 55)

    total = len(df)
    print(f"\nTotal classified responses : {total}")

    for label in ["refusal", "compliance", "partial"]:
        n = (df["label"] == label).sum()
        print(f"  {label:12s}: {n:4d}  ({100*n/total:.1f}%)")

    print("\n── ASR by Language (all models averaged) ──")
    lang_avg = asr.groupby("language_label")["asr_pct"].mean().sort_values(ascending=False)
    for lang, pct in lang_avg.items():
        print(f"  {lang:12s}: {pct:.1f}%")

    print("\n── ASR by Model (all languages averaged) ──")
    model_avg = asr.groupby("model_label")["asr_pct"].mean().sort_values(ascending=False)
    for model, pct in model_avg.items():
        print(f"  {model:25s}: {pct:.1f}%")

    # English baseline vs code-switched delta
    en_asr   = asr[asr["language"] == "en"]["asr_pct"].mean()
    cs_asr   = asr[asr["language"] != "en"]["asr_pct"].mean()
    delta    = cs_asr - en_asr
    print(f"\n── Code-Switching Effect ──")
    print(f"  English baseline ASR    : {en_asr:.1f}%")
    print(f"  Code-switched avg ASR   : {cs_asr:.1f}%")
    print(f"  Delta (code-switch lift): {delta:+.1f}%")
    print("=" * 55 + "\n")


def main():
    ensure_output_dirs()
    df  = load_eval()
    asr = compute_asr(df)

    print_summary(df, asr)
    plot_heatmap(asr)
    plot_asr_by_language(asr)
    plot_model_comparison(df)

    print("[analysis] All figures saved to figures/")


if __name__ == "__main__":
    main()
