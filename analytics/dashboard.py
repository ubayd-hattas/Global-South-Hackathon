"""
Mzansi Code-Switching Safety Benchmark — Streamlit Dashboard

Professional interactive dashboard for hackathon demonstration.

Run:
    streamlit run dashboard.py

The dashboard auto-loads evaluation.csv from the project root.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Mzansi Code-Switching Safety Benchmark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2D3436;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #636E72;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 4px solid #534AB7;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2D3436;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #636E72;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-highlight {
        color: #D85A30;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANG_ORDER = ["English", "Afrikaans", "isiZulu", "isiXhosa", "Tsonga"]
CAT_ORDER = [
    "Financial Fraud",
    "Xenophobic Incitement",
    "Political Disinformation",
    "Gang / Criminal Facilitation",
]
MODEL_ORDER = ["Gemini", "Llama", "OpenRouter model"]

LABEL_NAMES = {0: "Refusal", 1: "Partial Compliance", 2: "Harmful Compliance"}
LABEL_COLORS = {"Refusal": "#1D9E75", "Partial Compliance": "#EF9F27", "Harmful Compliance": "#D85A30"}
MODEL_COLORS = {"Gemini": "#534AB7", "Llama": "#0F6E56", "OpenRouter model": "#D85A30"}

# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(filepath: str = "evaluation.csv"):
    """Load and prepare evaluation data."""
    import sys
    # Ensure we can import data_loader when running from root
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    from data_loader import load_evaluation
    
    # Try multiple paths robustness (__file__ based)
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
        return df, "Research Mode", loaded_path
    else:
        # Return an empty DataFrame with the correct schema
        empty_df = pd.DataFrame({
            "prompt_id": [], "seed_id": [], "language": pd.Series(dtype='category'),
            "harm_category": pd.Series(dtype='category'), "model": pd.Series(dtype='category'),
            "label": [], "judging_method": [], "is_refusal": [], "is_partial": [], 
            "is_jailbreak": [], "is_compliance": [], "label_name": [], "severity_weight": []
        })
        empty_df["language"] = pd.Categorical([], categories=LANG_ORDER, ordered=True)
        empty_df["harm_category"] = pd.Categorical([], categories=CAT_ORDER, ordered=True)
        empty_df["model"] = pd.Categorical([], categories=MODEL_ORDER, ordered=True)
        return empty_df, "Demo Mode", None

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

try:
    df, mode, loaded_path = load_data()
    if mode == "Demo Mode":
        st.warning("⚠️ **Demo Mode Active:** `evaluation.csv` not found. Displaying empty dashboard. Please generate data to view visualizations.")
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

    st.markdown("#### Filters")

    selected_languages = st.multiselect(
        "Languages",
        options=LANG_ORDER,
        default=LANG_ORDER,
    )

    selected_models = st.multiselect(
        "Models",
        options=MODEL_ORDER,
        default=MODEL_ORDER,
    )

    selected_categories = st.multiselect(
        "Harm Categories",
        options=CAT_ORDER,
        default=CAT_ORDER,
    )

    st.markdown("---")
    st.markdown("#### Label Filter")
    label_filter = st.radio(
        "Response Type",
        options=["All", "Refusal Only", "Partial Only", "Jailbreak Only"],
        index=0,
    )

    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("*Dashboard auto-refreshes on filter change.*")

# ---------------------------------------------------------------------------
# Apply Filters
# ---------------------------------------------------------------------------

filtered_df = df[
    df["language"].isin(selected_languages) &
    df["model"].isin(selected_models) &
    df["harm_category"].isin(selected_categories)
].copy()

if label_filter == "Refusal Only":
    filtered_df = filtered_df[filtered_df["label"] == 0]
elif label_filter == "Partial Only":
    filtered_df = filtered_df[filtered_df["label"] == 1]
elif label_filter == "Jailbreak Only":
    filtered_df = filtered_df[filtered_df["label"] == 2]

if len(filtered_df) == 0:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selection or ensure data is loaded.")

# ---------------------------------------------------------------------------
# KPI Calculation
# ---------------------------------------------------------------------------

total_evals = len(filtered_df)
asr = filtered_df["is_jailbreak"].mean()
refusal_rate = filtered_df["is_refusal"].mean()
partial_rate = filtered_df["is_partial"].mean()

# Most vulnerable language
lang_asr = filtered_df.groupby("language", observed=True)["is_jailbreak"].mean()
most_vuln_lang = lang_asr.idxmax() if len(lang_asr) > 0 else "N/A"
most_vuln_lang_val = lang_asr.max() if len(lang_asr) > 0 else 0

# Most vulnerable category
cat_asr = filtered_df.groupby("harm_category", observed=True)["is_jailbreak"].mean()
most_vuln_cat = cat_asr.idxmax() if len(cat_asr) > 0 else "N/A"
most_vuln_cat_val = cat_asr.max() if len(cat_asr) > 0 else 0

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">Mzansi Code-Switching Safety Benchmark</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Evaluating whether code-switching into South African languages weakens LLM safety guardrails'
    '</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_languages, tab_models, tab_categories, tab_explorer = st.tabs([
    "📊 Overview",
    "🌍 Languages",
    "🤖 Models",
    "⚠️ Harm Categories",
    "🔍 Jailbreak Explorer",
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================

with tab_overview:
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Evaluations", f"{total_evals:,}")
    with col2:
        st.metric("ASR", f"{asr:.1%}", help="Attack Success Rate: proportion of label==2")
    with col3:
        st.metric("Refusal Rate", f"{refusal_rate:.1%}")
    with col4:
        st.metric(
            "Most Vulnerable Language",
            str(most_vuln_lang),
            f"{most_vuln_lang_val:.1%} ASR",
            delta_color="inverse",
        )
    with col5:
        st.metric(
            "Most Vulnerable Category",
            str(most_vuln_cat),
            f"{most_vuln_cat_val:.1%} ASR",
            delta_color="inverse",
        )

    st.markdown("---")

    # Charts Row 1
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Attack Success Rate Heatmap")

        pivot = filtered_df.groupby(["language", "harm_category"], observed=True)["is_jailbreak"].mean().unstack()
        pivot = pivot.reindex([l for l in LANG_ORDER if l in pivot.index])

        fig_heat = px.imshow(
            pivot,
            text_auto=".1%",
            color_continuous_scale="RdYlGn_r",
            zmin=0,
            zmax=1,
            aspect="auto",
            labels={"color": "ASR", "x": "Harm Category", "y": "Language"},
        )
        fig_heat.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=350,
            coloraxis_colorbar=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.subheader("Response Distribution")

        dist = filtered_df.groupby("language", observed=True)[["is_refusal", "is_partial", "is_jailbreak"]].mean()
        dist = dist.reindex([l for l in LANG_ORDER if l in dist.index]) * 100

        fig_dist = go.Figure()
        for col, name, color in [
            ("is_refusal", "Refusal", LABEL_COLORS["Refusal"]),
            ("is_partial", "Partial Compliance", LABEL_COLORS["Partial Compliance"]),
            ("is_jailbreak", "Harmful Compliance", LABEL_COLORS["Harmful Compliance"]),
        ]:
            fig_dist.add_trace(go.Bar(
                name=name,
                x=[str(l) for l in dist.index],
                y=dist[col].round(1),
                marker_color=color,
            ))

        fig_dist.update_layout(
            barmode="stack",
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(ticksuffix="%", title="Percentage"),
            xaxis_title="",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Charts Row 2
    st.markdown("---")
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("ASR by Language")

        lang_asr_df = filtered_df.groupby("language", observed=True)["is_jailbreak"].mean().reset_index()
        lang_asr_df["asr_pct"] = (lang_asr_df["is_jailbreak"] * 100).round(1)

        fig_lang = px.bar(
            lang_asr_df,
            x="language",
            y="is_jailbreak",
            text=lang_asr_df["asr_pct"].apply(lambda x: f"{x:.1f}%"),
            color="language",
            color_discrete_map={
                "English": "#534AB7",
                "Afrikaans": "#0F6E56",
                "isiZulu": "#D85A30",
                "isiXhosa": "#EF9F27",
                "Tsonga": "#6C5CE7",
            },
            labels={"is_jailbreak": "ASR", "language": ""},
        )
        fig_lang.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    with col_right2:
        st.subheader("ASR by Model")

        model_asr_df = filtered_df.groupby("model", observed=True)["is_jailbreak"].mean().reset_index()
        model_asr_df["asr_pct"] = (model_asr_df["is_jailbreak"] * 100).round(1)

        fig_model = px.bar(
            model_asr_df,
            x="model",
            y="is_jailbreak",
            text=model_asr_df["asr_pct"].apply(lambda x: f"{x:.1f}%"),
            color="model",
            color_discrete_map=MODEL_COLORS,
            labels={"is_jailbreak": "ASR", "model": ""},
        )
        fig_model.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_model, use_container_width=True)

    # Download Section
    st.markdown("---")
    st.subheader("📥 Export Data")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_evaluation.csv",
            mime="text/csv",
            use_container_width=True,
        )

# =============================================================================
# TAB 2: LANGUAGES
# =============================================================================

with tab_languages:
    st.subheader("Language Safety Analysis")

    # Safety Gap
    st.markdown("#### Safety Gap vs English Baseline")

    english_asr = filtered_df[filtered_df["language"] == "English"]["is_jailbreak"].mean()

    gap_data = []
    for lang in LANG_ORDER:
        if lang == "English":
            continue
        lang_df = filtered_df[filtered_df["language"] == lang]
        if len(lang_df) == 0:
            continue
        lang_asr = lang_df["is_jailbreak"].mean()
        gap_data.append({
            "Language": lang,
            "ASR": lang_asr,
            "ASR %": round(lang_asr * 100, 1),
            "English Baseline": english_asr,
            "Gap (pp)": round((lang_asr - english_asr) * 100, 1),
        })

    gap_df = pd.DataFrame(gap_data)

    if len(gap_df) > 0:
        fig_gap = px.bar(
            gap_df.sort_values("Gap (pp)"),
            x="Gap (pp)",
            y="Language",
            orientation="h",
            color="Gap (pp)",
            color_continuous_scale="RdYlGn_r",
            text=gap_df["Gap (pp)"].apply(lambda x: f"{x:+.1f}pp"),
            labels={"Gap (pp)": "Gap vs English (percentage points)"},
        )
        fig_gap.add_vline(
            x=0,
            line_dash="dash",
            line_color="#2D3436",
            annotation_text=f"English baseline: {english_asr:.1%}",
        )
        fig_gap.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    # Language Detail Table
    st.markdown("#### Language Summary Table")

    lang_summary = filtered_df.groupby("language", observed=True).agg(
        Total=("label", "size"),
        Refusals=("is_refusal", "sum"),
        Partials=("is_partial", "sum"),
        Jailbreaks=("is_jailbreak", "sum"),
        Refusal_Rate=("is_refusal", "mean"),
        Partial_Rate=("is_partial", "mean"),
        ASR=("is_jailbreak", "mean"),
    ).reset_index()

    lang_summary["Refusal %"] = (lang_summary["Refusal_Rate"] * 100).round(1)
    lang_summary["Partial %"] = (lang_summary["Partial_Rate"] * 100).round(1)
    lang_summary["ASR %"] = (lang_summary["ASR"] * 100).round(1)

    display_df = lang_summary[["language", "Total", "Refusals", "Partials", "Jailbreaks", "Refusal %", "Partial %", "ASR %"]]
    display_df.columns = ["Language", "Total", "Refusals", "Partial", "Jailbreaks", "Refusal %", "Partial %", "ASR %"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: MODELS
# =============================================================================

with tab_models:
    st.subheader("Model Robustness Analysis")

    # Model ASR comparison
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Attack Success Rate by Model")

        model_summary = filtered_df.groupby("model", observed=True).agg(
            Total=("label", "size"),
            Jailbreaks=("is_jailbreak", "sum"),
            ASR=("is_jailbreak", "mean"),
        ).reset_index()
        model_summary["ASR %"] = (model_summary["ASR"] * 100).round(1)
        model_summary["Robustness"] = (100 - model_summary["ASR %"]).round(1)

        fig_masr = px.bar(
            model_summary.sort_values("ASR", ascending=False),
            x="model",
            y="ASR",
            text=model_summary["ASR %"].apply(lambda x: f"{x:.1f}%"),
            color="model",
            color_discrete_map=MODEL_COLORS,
            labels={"ASR": "Attack Success Rate", "model": ""},
        )
        fig_masr.update_layout(
            showlegend=False,
            height=300,
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_masr, use_container_width=True)

    with col_m2:
        st.markdown("#### Robustness Score (100 - ASR%)")

        fig_rob = px.bar(
            model_summary.sort_values("Robustness", ascending=True),
            x="model",
            y="Robustness",
            text=model_summary["Robustness"].apply(lambda x: f"{x:.1f}"),
            color="model",
            color_discrete_map=MODEL_COLORS,
            labels={"Robustness": "Robustness Score", "model": ""},
        )
        fig_rob.update_layout(
            showlegend=False,
            height=300,
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig_rob, use_container_width=True)

    # Model × Language Grid
    st.markdown("---")
    st.markdown("#### ASR by Model × Language")

    model_lang = filtered_df.groupby(["model", "language"], observed=True)["is_jailbreak"].mean().unstack()
    model_lang = model_lang.reindex(columns=[l for l in LANG_ORDER if l in model_lang.columns])

    fig_ml = px.imshow(
        model_lang,
        text_auto=".1%",
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=1,
        aspect="auto",
        labels={"color": "ASR", "x": "Language", "y": "Model"},
    )
    fig_ml.update_layout(
        height=300,
        coloraxis_colorbar=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig_ml, use_container_width=True)

    # Model summary table
    st.markdown("#### Model Summary Table")

    display_model = model_summary[["model", "Total", "Jailbreaks", "ASR %", "Robustness"]]
    display_model.columns = ["Model", "Total", "Jailbreaks", "ASR %", "Robustness"]

    st.dataframe(display_model, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: HARM CATEGORIES
# =============================================================================

with tab_categories:
    st.subheader("Harm Category Risk Analysis")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### Risk Score by Category")

        cat_summary = filtered_df.groupby("harm_category", observed=True).agg(
            Total=("label", "size"),
            Jailbreaks=("is_jailbreak", "sum"),
            ASR=("is_jailbreak", "mean"),
        ).reset_index()
        cat_summary["ASR %"] = (cat_summary["ASR"] * 100).round(1)
        cat_summary = cat_summary.sort_values("ASR", ascending=True)

        fig_cat = px.bar(
            cat_summary,
            x="ASR",
            y="harm_category",
            orientation="h",
            text=cat_summary["ASR %"].apply(lambda x: f"{x:.1f}%"),
            color="ASR",
            color_continuous_scale="RdYlGn_r",
            labels={"ASR": "Attack Success Rate", "harm_category": ""},
        )
        fig_cat.update_layout(
            height=300,
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_c2:
        st.markdown("#### Category × Language Heatmap")

        cat_lang = filtered_df.groupby(["harm_category", "language"], observed=True)["is_jailbreak"].mean().unstack()
        cat_lang = cat_lang.reindex(columns=[l for l in LANG_ORDER if l in cat_lang.columns])

        fig_cl = px.imshow(
            cat_lang,
            text_auto=".1%",
            color_continuous_scale="RdYlGn_r",
            zmin=0,
            zmax=1,
            aspect="auto",
            labels={"color": "ASR", "x": "Language", "y": "Harm Category"},
        )
        fig_cl.update_layout(
            height=300,
            coloraxis_colorbar=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_cl, use_container_width=True)

    # Category table
    st.markdown("#### Category Summary Table")

    display_cat = cat_summary.sort_values("ASR", ascending=False)
    display_cat = display_cat[["harm_category", "Total", "Jailbreaks", "ASR %"]]
    display_cat.columns = ["Harm Category", "Total", "Jailbreaks", "ASR %"]

    st.dataframe(display_cat, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: JAILBREAK EXPLORER
# =============================================================================

with tab_explorer:
    st.subheader("🔍 Jailbreak Case Explorer")
    st.caption("Drill down into individual evaluation cases. Filter and inspect specific prompts.")

    # Explorer filters
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        exp_lang = st.multiselect("Language", LANG_ORDER, default=LANG_ORDER, key="exp_lang")
    with col_e2:
        exp_model = st.multiselect("Model", MODEL_ORDER, default=MODEL_ORDER, key="exp_model")
    with col_e3:
        exp_label = st.multiselect(
            "Label",
            options=[0, 1, 2],
            default=[0, 1, 2],
            format_func=lambda x: LABEL_NAMES[x],
            key="exp_label",
        )

    explorer_df = filtered_df[
        filtered_df["language"].isin(exp_lang) &
        filtered_df["model"].isin(exp_model) &
        filtered_df["label"].isin(exp_label)
    ].copy()

    st.markdown(f"**Showing {len(explorer_df)} of {len(filtered_df)} filtered cases**")

    # Display table
    display_cols = ["prompt_id", "seed_id", "language", "harm_category", "model", "label", "label_name", "judging_method"]
    display_cols = [c for c in display_cols if c in explorer_df.columns]

    st.dataframe(
        explorer_df[display_cols].head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "prompt_id": st.column_config.NumberColumn("Prompt ID"),
            "seed_id": st.column_config.NumberColumn("Seed ID"),
            "language": st.column_config.TextColumn("Language"),
            "harm_category": st.column_config.TextColumn("Harm Category"),
            "model": st.column_config.TextColumn("Model"),
            "label": st.column_config.NumberColumn("Label"),
            "label_name": st.column_config.TextColumn("Outcome"),
            "judging_method": st.column_config.TextColumn("Judge"),
        },
    )

    # Label distribution pie chart for explorer
    st.markdown("---")
    st.markdown("#### Label Distribution in Selection")

    label_counts = explorer_df["label_name"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]

    fig_pie = px.pie(
        label_counts,
        values="Count",
        names="Label",
        color="Label",
        color_discrete_map=LABEL_COLORS,
        hole=0.4,
    )
    fig_pie.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "🛡️ **Mzansi Code-Switching Safety Benchmark** | "
    "Global South AI Safety Hackathon | "
    "Built with Streamlit"
)
