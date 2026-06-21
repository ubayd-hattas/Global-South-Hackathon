"""
Mzansi Code-Switching Safety Benchmark — Streamlit Dashboard

Run:
    streamlit run analytics/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mzansi Code-Switching Safety Benchmark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #2D3436; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #636E72; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(filepath: str = "evaluation.csv"):
    import sys
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    from data_loader import load_evaluation, VALID_LANGUAGES, VALID_CATEGORIES, VALID_MODELS
    
    paths = [
        current_dir / filepath,
        current_dir / "evaluation.csv",
        current_dir.parent / "evaluation.csv", 
        current_dir.parent / "data" / "results" / "evaluation.csv"
    ]
    
    df = None
    loaded_path = None
    for p in paths:
        if p.exists():
            try:
                df = load_evaluation(str(p))
                loaded_path = p
                break
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                continue

    if df is not None:
        return df, "Research Mode", loaded_path, VALID_LANGUAGES, VALID_CATEGORIES, VALID_MODELS
    else:
        empty_df = pd.DataFrame({
            "prompt_id": [], "seed_id": [], "language": pd.Series(dtype='category'),
            "harm_category": pd.Series(dtype='category'), "model": pd.Series(dtype='category'),
            "label": [], "judging_method": [], "is_refusal": [], "is_partial": [], 
            "is_jailbreak": [], "is_compliance": [], "label_name": [], "severity_weight": []
        })
        empty_df["language"] = pd.Categorical([], categories=VALID_LANGUAGES, ordered=True)
        empty_df["harm_category"] = pd.Categorical([], categories=VALID_CATEGORIES, ordered=True)
        empty_df["model"] = pd.Categorical([], categories=VALID_MODELS, ordered=True)
        return empty_df, "Demo Mode", None, VALID_LANGUAGES, VALID_CATEGORIES, VALID_MODELS

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
try:
    df, mode, loaded_path, LANG_ORDER, CAT_ORDER, MODEL_ORDER = load_data()
    if mode == "Demo Mode":
        st.warning("⚠️ **Demo Mode Active:** `evaluation.csv` not found. Displaying empty dashboard. Please drop in your data to view visualizations.")
    else:
        st.success(f"✅ **Research Mode Active:** Loaded data from `{loaded_path.name}`")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Mzansi Safety Benchmark")
    st.markdown("---")
    st.markdown("#### Interactive Filters")

    # Default to ALL options (including "Other") to avoid filtering out data
    selected_models = st.multiselect("Models", options=MODEL_ORDER, default=MODEL_ORDER)
    selected_languages = st.multiselect("Languages", options=LANG_ORDER, default=LANG_ORDER)
    selected_categories = st.multiselect("Harm Categories", options=CAT_ORDER, default=CAT_ORDER)

    st.markdown("---")
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

# ---------------------------------------------------------------------------
# Apply Filters
# ---------------------------------------------------------------------------
filtered_df = df[
    df["language"].isin(selected_languages) &
    df["model"].isin(selected_models) &
    df["harm_category"].isin(selected_categories)
].copy()

# Debug information
if mode == "Research Mode":
    with st.expander("🔍 Debug Information", expanded=False):
        st.write(f"**Total rows loaded:** {len(df)}")
        st.write(f"**Rows after filtering:** {len(filtered_df)}")
        st.write(f"**Unique models in data:** {df['model'].unique().tolist()}")
        st.write(f"**Unique languages in data:** {df['language'].unique().tolist()}")
        st.write(f"**Unique categories in data:** {df['harm_category'].unique().tolist()}")
        st.write(f"**Selected models:** {selected_models}")
        st.write(f"**Selected languages:** {selected_languages}")
        st.write(f"**Selected categories:** {selected_categories}")

if len(filtered_df) == 0 and mode != "Demo Mode":
    st.warning("⚠️ No data matches the selected filters. Check the Debug Information above.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">Mzansi Code-Switching Safety Benchmark</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyzing the impact of code-switching on AI safety guardrails</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_models, tab_languages, tab_categories, tab_export = st.tabs([
    "📊 Overview",
    "🤖 Model Analysis",
    "🌍 Language Analysis",
    "⚠️ Harm Categories",
    "📥 Export"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    if len(filtered_df) > 0:
        total_prompts = len(filtered_df)
        total_jailbreaks = filtered_df["is_jailbreak"].sum()
        avg_asr = filtered_df["is_jailbreak"].mean()
        
        model_asr = filtered_df.groupby("model", observed=True)["is_jailbreak"].mean()
        best_model = model_asr.idxmin() if len(model_asr) > 0 else "N/A"  # Best = lowest ASR (safest)
        worst_model = model_asr.idxmax() if len(model_asr) > 0 else "N/A" # Worst = highest ASR
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Prompts", f"{total_prompts:,}")
        col2.metric("Total Jailbreaks", f"{total_jailbreaks:,}")
        col3.metric("Average ASR", f"{avg_asr:.1%}")
        col4.metric("Best Model (Safest)", str(best_model))
        col5.metric("Worst Model (Least Safe)", str(worst_model))
        
        st.markdown("---")
        st.markdown("#### High-Level Attack Success Rate")
        
        # ASR over Models and Languages Heatmap (Interactive)
        pivot = filtered_df.groupby(["language", "model"], observed=True)["is_jailbreak"].mean().unstack()
        if not pivot.empty:
            fig_heat = px.imshow(
                pivot, text_auto=".1%", color_continuous_scale="Reds", aspect="auto",
                labels={"color": "ASR", "x": "Model", "y": "Language"}
            )
            fig_heat.update_layout(height=400, coloraxis_colorbar=dict(tickformat=".0%"))
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data available for Overview.")

# =============================================================================
# TAB 2: MODEL ANALYSIS
# =============================================================================
with tab_models:
    st.subheader("Model Performance")
    if len(filtered_df) > 0:
        model_summary = filtered_df.groupby("model", observed=True).agg(
            ASR=("is_jailbreak", "mean"),
            Refusal_Rate=("is_refusal", "mean")
        ).reset_index()
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**ASR by Model**")
            fig_masr = px.bar(
                model_summary.sort_values("ASR", ascending=False),
                x="model", y="ASR", text=model_summary["ASR"].apply(lambda x: f"{x:.1%}"),
                color="model", labels={"ASR": "Attack Success Rate", "model": ""}
            )
            fig_masr.update_layout(showlegend=False, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_masr, use_container_width=True)

        with col_m2:
            st.markdown("**Refusal Rate by Model**")
            fig_mref = px.bar(
                model_summary.sort_values("Refusal_Rate", ascending=True),
                x="model", y="Refusal_Rate", text=model_summary["Refusal_Rate"].apply(lambda x: f"{x:.1%}"),
                color="model", labels={"Refusal_Rate": "Refusal Rate", "model": ""}
            )
            fig_mref.update_layout(showlegend=False, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_mref, use_container_width=True)
    else:
        st.info("No data available.")

# =============================================================================
# TAB 3: LANGUAGE ANALYSIS
# =============================================================================
with tab_languages:
    st.subheader("Language Vulnerability")
    if len(filtered_df) > 0:
        lang_asr = filtered_df.groupby("language", observed=True)["is_jailbreak"].mean().reset_index()
        
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown("**ASR by Language**")
            fig_lang = px.bar(
                lang_asr.sort_values("is_jailbreak", ascending=False),
                x="language", y="is_jailbreak", text=lang_asr["is_jailbreak"].apply(lambda x: f"{x:.1%}"),
                color="language", labels={"is_jailbreak": "Attack Success Rate", "language": ""}
            )
            fig_lang.update_layout(showlegend=False, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_lang, use_container_width=True)

        with col_l2:
            st.markdown("**Safety Gap vs English (Percentage Points)**")
            en_df = filtered_df[filtered_df["language"] == "English"]
            english_asr = en_df["is_jailbreak"].mean() if len(en_df) > 0 else 0
            
            gap_data = []
            for lang in lang_asr["language"]:
                if lang == "English": continue
                val = lang_asr.loc[lang_asr["language"] == lang, "is_jailbreak"].values[0]
                gap_data.append({"Language": lang, "Gap": (val - english_asr) * 100})
            
            if gap_data:
                gap_df = pd.DataFrame(gap_data)
                fig_gap = px.bar(
                    gap_df.sort_values("Gap"), x="Gap", y="Language", orientation="h",
                    color="Gap", color_continuous_scale="RdYlGn_r",
                    text=gap_df["Gap"].apply(lambda x: f"{x:+.1f}pp"),
                    labels={"Gap": "Gap vs English (pp)"}
                )
                fig_gap.add_vline(x=0, line_dash="dash", line_color="black")
                fig_gap.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_gap, use_container_width=True)
            else:
                st.info("Insufficient data to compute gap against English.")
    else:
        st.info("No data available.")

# =============================================================================
# TAB 4: HARM CATEGORIES
# =============================================================================
with tab_categories:
    st.subheader("Harm Categories Analysis")
    if len(filtered_df) > 0:
        cat_summary = filtered_df.groupby("harm_category", observed=True).agg(
            ASR=("is_jailbreak", "mean")
        ).reset_index().sort_values("ASR", ascending=False)
        
        fig_cat = px.bar(
            cat_summary, x="ASR", y="harm_category", orientation="h",
            text=cat_summary["ASR"].apply(lambda x: f"{x:.1%}"), color="ASR",
            color_continuous_scale="Reds", labels={"ASR": "ASR", "harm_category": ""}
        )
        fig_cat.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No data available.")

# =============================================================================
# TAB 5: EXPORT
# =============================================================================
with tab_export:
    st.subheader("📥 Export Data")
    if len(filtered_df) > 0:
        st.markdown("Download the current filtered dataset or summary statistics.")
        
        col_e1, col_e2 = st.columns(2)
        
        # 1. Filtered CSV
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        col_e1.download_button(
            label="Download Filtered CSV",
            data=csv,
            file_name="mzansi_filtered_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        
        # 2. Summary Statistics JSON
        stats = {
            "total_prompts": int(len(filtered_df)),
            "total_jailbreaks": int(filtered_df["is_jailbreak"].sum()),
            "average_asr": float(filtered_df["is_jailbreak"].mean()),
            "asr_by_model": filtered_df.groupby("model", observed=True)["is_jailbreak"].mean().to_dict(),
            "asr_by_language": filtered_df.groupby("language", observed=True)["is_jailbreak"].mean().to_dict()
        }
        json_str = json.dumps(stats, indent=4)
        col_e2.download_button(
            label="Download Summary Stats (JSON)",
            data=json_str,
            file_name="mzansi_summary_stats.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("No data to export.")
