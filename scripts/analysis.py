"""
analysis.py
-----------
Generates analytical figures for publication.
Run from repository root: python scripts/analysis.py
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import EVALUATION_CSV, FIGURES_DIR, ensure_output_dirs

# Mapping constants (mirroring data_loader but we process raw evaluation.csv here)
LANG_MAP = {"en": "English", "af": "Afrikaans", "zu": "isiZulu", "xh": "isiXhosa", "ts": "Tsonga", "nso": "Sepedi", "st": "Sesotho"}
MODEL_MAP = {"kimi": "Kimi k2.6", "llama33": "Llama 3.3 70B", "qwen3": "Qwen 3 32B", "gptoss": "GPT OSS 20B"}
PALETTE = sns.color_palette("colorblind")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

def load_eval() -> pd.DataFrame:
    eval_path = EVALUATION_CSV
    if not eval_path.exists():
        fallback = Path(__file__).resolve().parent.parent / "analytics" / "evaluation.csv"
        if fallback.exists():
            eval_path = fallback
        else:
            print(f"[ERROR] evaluation.csv not found.", file=sys.stderr)
            sys.exit(1)
            
    df = pd.read_csv(eval_path)
    
    # We might have numerical labels depending on the generation step.
    df = df[df["label"].isin(["refusal", "compliance", "partial", 0, 1, 2, "0", "1", "2"])].copy()
    
    label_map = {0: "refusal", 1: "partial", 2: "compliance", "0": "refusal", "1": "partial", "2": "compliance"}
    df["label_str"] = df["label"].map(lambda x: label_map.get(x, x))
    
    df["language_label"] = df["language"].map(lambda x: LANG_MAP.get(x, x))
    df["model_label"] = df["model"].map(lambda x: MODEL_MAP.get(x, x))
    df["is_compliance"] = (df["label_str"] == "compliance").astype(int)
    df["is_refusal"] = (df["label_str"] == "refusal").astype(int)
    return df

def plot_figure_1(df: pd.DataFrame):
    """Fig 1: Attack Success Rate heatmap (Rows: languages, Columns: harm categories)"""
    asr = df.groupby(["language_label", "harm_category"])["is_compliance"].mean().reset_index()
    pivot = asr.pivot(index="language_label", columns="harm_category", values="is_compliance") * 100
    
    order = ["English"] + sorted([r for r in pivot.index if r != "English"])
    pivot = pivot.reindex([r for r in order if r in pivot.index])
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax, vmin=0, vmax=100, cbar_kws={"label": "ASR (%)"})
    ax.set_title("Figure 1: Attack Success Rate (%) by Language and Harm Category", pad=12)
    ax.set_ylabel("Language")
    ax.set_xlabel("Harm Category")
    plt.tight_layout()
    out = FIGURES_DIR / "fig1_asr_heatmap.png"
    fig.savefig(out)
    plt.close(fig)

def plot_figure_2(df: pd.DataFrame):
    """Fig 2: Model comparison bar chart (Safest to least safe = Highest refusal to lowest)"""
    refusal = df.groupby("model_label")["is_refusal"].mean().reset_index()
    refusal["Refusal %"] = refusal["is_refusal"] * 100
    refusal = refusal.sort_values("Refusal %", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(data=refusal, x="model_label", y="Refusal %", hue="model_label", palette="viridis", legend=False, ax=ax)
    for bar in bars.containers[0]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center")
    
    ax.set_title("Figure 2: Model Safety Ranking (Refusal Rate %)", pad=12)
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_model_comparison.png"
    fig.savefig(out)
    plt.close(fig)

def plot_figure_3(df: pd.DataFrame):
    """Fig 3: Language safety gap chart (Percentage points vs English)"""
    asr = df.groupby("language_label")["is_compliance"].mean().reset_index()
    asr["ASR %"] = asr["is_compliance"] * 100
    eng_asr = asr.loc[asr["language_label"] == "English", "ASR %"].values
    if len(eng_asr) == 0:
        return
    eng_asr = eng_asr[0]
    
    asr["Gap (pp)"] = asr["ASR %"] - eng_asr
    gap_df = asr[asr["language_label"] != "English"].sort_values("Gap (pp)")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(data=gap_df, x="Gap (pp)", y="language_label", hue="language_label", palette="RdYlGn_r", legend=False, ax=ax, orient='h')
    for bar in bars.containers[0]:
        ax.text(bar.get_width() + (1 if bar.get_width() > 0 else -3), bar.get_y() + bar.get_height() / 2, f"{bar.get_width():+.1f} pp", va="center")
    
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title(f"Figure 3: Language Safety Gap vs English Baseline ({eng_asr:.1f}%)", pad=12)
    ax.set_xlabel("Gap in ASR (percentage points)")
    ax.set_ylabel("Language")
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_language_safety_gap.png"
    fig.savefig(out)
    plt.close(fig)

def plot_figure_4(df: pd.DataFrame):
    """Fig 4: Jailbreak success by model and language"""
    asr = df.groupby(["language_label", "model_label"])["is_compliance"].mean().reset_index()
    asr["ASR %"] = asr["is_compliance"] * 100
    
    order = ["English"] + sorted([l for l in asr["language_label"].unique() if l != "English"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    n_models = asr["model_label"].nunique()
    sns.barplot(data=asr, x="language_label", y="ASR %", hue="model_label", order=order, palette=PALETTE[:n_models], ax=ax)
    
    ax.set_title("Figure 4: Jailbreak Success Rate (ASR) by Model and Language", pad=12)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Language")
    ax.set_ylim(0, 105)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    out = FIGURES_DIR / "fig4_jailbreak_by_model_lang.png"
    fig.savefig(out)
    plt.close(fig)

def main():
    ensure_output_dirs()
    df = load_eval()
    plot_figure_1(df)
    plot_figure_2(df)
    plot_figure_3(df)
    plot_figure_4(df)
    print("[analysis] All 4 publication figures saved to figures/")

if __name__ == "__main__":
    main()
