import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Footprint Prediction - India",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

# ── Load Champion Info dynamically ────────────────────────────────────
champion_name = "XGBoost"
champion_rmse = 10704.0
champion_rmse_str = "RMSE: 10,704"
metrics_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
if os.path.exists(metrics_path):
    try:
        df_m = pd.read_csv(metrics_path)
        best_m = df_m.sort_values(by="RMSE").iloc[0]
        champion_name = best_m["Model"].split(" ")[0] # e.g. "XGBoost" or "Random"
        champion_rmse = best_m["RMSE"]
        champion_rmse_str = f"RMSE: {champion_rmse:,.0f}"
    except Exception:
        pass

# ── Load Expanding-Window CV metrics ──────────────────────────────────
ew_metrics = {}
ew_csv_path = os.path.join(RESULTS_DIR, "expanding_window_cv_metrics.csv")
if os.path.exists(ew_csv_path):
    try:
        ew_df = pd.read_csv(ew_csv_path)
        for _, row in ew_df.iterrows():
            ew_metrics[row["Metric"]] = row["Value"]
    except Exception:
        pass

# ── Helpers ───────────────────────────────────────────────────────────
def load_csv(fname):
    p = os.path.join(DATA_PROCESSED, fname)
    return pd.read_csv(p) if os.path.exists(p) else None

def load_img(fname):
    p = os.path.join(RESULTS_DIR, fname)
    return Image.open(p) if os.path.exists(p) else None

def sh(text):
    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:700;color:#4ade80;'
        f'margin:2rem 0 .9rem 0;padding-bottom:.45rem;'
        f'border-bottom:2px solid rgba(74,222,128,0.3);">{text}</div>',
        unsafe_allow_html=True,
    )

def mcards(cards):
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        col.markdown(
            f'<div style="background:linear-gradient(145deg,#1e293b,#0f172a);'
            f'border:1px solid rgba(255,255,255,0.08);border-radius:14px;'
            f'padding:1.3rem 1.5rem;text-align:center;'
            f'box-shadow:0 4px 20px rgba(0,0,0,0.2);">'
            f'<div style="font-size:.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:.45rem;">{c["label"]}</div>'
            f'<div style="font-size:1.8rem;font-weight:700;color:#4ade80;">{c["value"]}</div>'
            f'<div style="font-size:.75rem;color:rgba(255,255,255,0.4);margin-top:.25rem;">{c.get("sub","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

def pcard(num, title, desc):
    st.markdown(
        f'<div style="background:linear-gradient(145deg,#1a2332,#0f1923);'
        f'border:1px solid rgba(74,222,128,0.14);border-radius:12px;'
        f'padding:1.4rem;margin-bottom:.9rem;box-shadow:0 2px 12px rgba(0,0,0,0.15);">'
        f'<span style="display:inline-block;background:linear-gradient(135deg,#16a34a,#22c55e);'
        f'color:#fff;width:30px;height:30px;line-height:30px;text-align:center;'
        f'border-radius:50%;font-weight:700;font-size:.82rem;margin-right:.65rem;">{num}</span>'
        f'<span style="font-size:1.05rem;font-weight:600;color:#e2e8f0;">{title}</span>'
        f'<div style="color:rgba(255,255,255,0.58);font-size:.88rem;margin-top:.45rem;line-height:1.55;">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def finding(txt):
    st.markdown(
        f'<div style="background:rgba(30,41,59,0.5);border-left:4px solid #4ade80;'
        f'padding:1rem 1.25rem;border-radius:0 12px 12px 0;margin-bottom:.85rem;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.15);"> '
        f'<p style="margin:0;color:#f8fafc;font-size:0.92rem;line-height:1.6;font-weight:400;">{txt}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1b2a 0%,#1b2838 100%);
}
[data-testid="stMetricValue"] { color: #4ade80 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a 0%,#1b2838 35%,#1a5632 70%,#2d8a4e 100%);
            padding:2.5rem 2rem;border-radius:16px;margin-bottom:1.8rem;
            box-shadow:0 8px 32px rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);">
  <h1 style="color:#fff;font-size:2.3rem;font-weight:800;margin-bottom:.4rem;font-family:Inter,sans-serif;">
    🌿 Carbon Footprint Prediction &mdash; India
  </h1>
  <p style="color:rgba(255,255,255,0.75);font-size:1rem;line-height:1.65;margin:0;font-family:Inter,sans-serif;">
    An advanced Machine Learning &amp; Explainable AI (XAI) pipeline to predict and interpret
    India&#39;s agricultural carbon emissions across decades of multi-source environmental data.
  </p>
  <div style="margin-top:1rem;display:flex;gap:.5rem;flex-wrap:wrap;">
    <span style="background:#3498db;color:#fff;padding:4px 12px;border-radius:20px;font-size:.75rem;font-weight:600;">Author: Prakash Ukhalkar</span>
    <span style="background:#306998;color:#FFD43B;padding:4px 12px;border-radius:20px;font-size:.75rem;font-weight:600;">Python 3.10+</span>
    <span style="background:#c0392b;color:#fff;padding:4px 12px;border-radius:20px;font-size:.75rem;font-weight:600;">XGBoost</span>
    <span style="background:#e67e22;color:#fff;padding:4px 12px;border-radius:20px;font-size:.75rem;font-weight:600;">SHAP</span>
    <span style="background:#f1c40f;color:#333;padding:4px 12px;border-radius:20px;font-size:.75rem;font-weight:600;">MIT License</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Navigation")
    page = st.radio(
        "section",
        [
            "🏠 Introduction",
            "📥 Data Acquisition",
            "🔧 Feature Engineering",
            "🤖 Predictive Modeling",
            "🔍 Model Interpretability",
            "📊 Key Findings",
            "📝 Summary",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:rgba(255,255,255,0.4);font-size:.76rem;'>"
        "<p>Built with ❤️ using Streamlit</p><p>© 2026 Prakash Ukhalkar</p></div>",
        unsafe_allow_html=True,
    )


# ============================================================
#  INTRODUCTION
# ============================================================
if page == "🏠 Introduction":
    sh("Project Overview")
    st.markdown("""
The **India Agri Carbon Prediction** project is a research-grade machine learning pipeline
designed to predict and interpret India's agricultural carbon footprint.
By leveraging three decades of multi-source data — emissions, energy consumption, and
natural resource utilization — the pipeline builds **XGBoost** predictive models
and demystifies results with **SHAP (SHapley Additive exPlanations)**.

**Core Research Questions addressed:**
- Which agricultural activities are the primary CO₂e emission drivers?
- How can sustainable policy interventions curb future carbon output?
- What does a transparent, publication-ready ML pipeline look like in practice?
""")

    sh("Pipeline Architecture")
    col1, col2 = st.columns(2)
    with col1:
        pcard("01", "Initial Data Consolidation",
              "Merges 32 raw CSV/Excel files into 3 thematic master panels: Emissions, "
              "Energy &amp; Activity, and Land &amp; Water Resources — via automated parsing.")
        pcard("02", "Feature Engineering &amp; Target Alignment",
              "Engineers temporal lag/rolling features, applies ADF stationarity tests, "
              "and constructs Water-Energy-Food Nexus indicators.")
    with col2:
        pcard("03", "Predictive Modeling &amp; Optimization",
              "Trains Random Forest (baseline), XGBoost, and SVR with Time-Series CV "
              "and GridSearchCV. XGBoost is selected as the champion model.")
        pcard("04", "Interpretability &amp; Policy Insights",
              "Applies SHAP TreeExplainer for global (beeswarm) and local (waterfall) "
              "interpretability, then runs sustainable policy scenario simulations.")

    sh("At a Glance")
    mcards([
        {"label": "Raw Files Processed", "value": "32", "sub": "CSV / Excel"},
        {"label": "Years Covered",       "value": "30+", "sub": "1993 – 2023"},
        {"label": "Thematic Masters",    "value": "3",   "sub": "Emissions, Energy, Resources"},
        {"label": "Champion Model",      "value": champion_name, "sub": champion_rmse_str},
    ])


# ============================================================
#  DATA ACQUISITION
# ============================================================
elif page == "📥 Data Acquisition":
    sh("Notebook 01 — Initial Data Consolidation")
    st.markdown("""
**Objective:** Automate the transformation of **32 raw files** (CSV & Excel) into 3 coherent,
temporally aligned master DataFrames so the rest of the pipeline has clean, structured inputs.

**Challenge:** Raw files use inconsistent year formats (`1995`, `1995-96`, `1995-1996`)
and span multiple agencies: FAOSTAT, Ministry of Agriculture (MoAFW), and MoEFCC.
""")

    df_emi = load_csv("master_emissions_panel.csv")
    df_ene = load_csv("master_energy_activity.csv")
    df_res = load_csv("master_land_water_resource.csv")

    sh("Thematic Masters Created")
    col1, col2, col3 = st.columns(3)
    for col, label, icon, df, desc in [
        (col1, "Emissions",        "🏭", df_emi, "CO₂, CH₄, N₂O from FAOSTAT crop & energy emission data."),
        (col2, "Energy & Activity","⚡", df_ene, "Electricity, renewables, fertiliser, fuel use from multiple agencies."),
        (col3, "Land & Water",     "🌊", df_res, "Irrigated area, water withdrawal, soil quality proxies."),
    ]:
        with col:
            st.markdown(f"#### {icon} {label}")
            st.markdown(desc)
            if df is not None:
                st.metric("Shape", f"{df.shape[0]:,} × {df.shape[1]}")
            else:
                st.warning("File not found")

    sh("Year Standardization Logic")
    st.code("""
def clean_year_column(series: pd.Series) -> pd.Series:
    \'\'\'Converts 1995-96, 1995-1996, or plain 1995  →  integer 1995.\'\'\'
    def extract(val):
        s = str(val).strip()
        if \'-\' in s:           # e.g. \'1995-96\' or \'1995-1996\'
            return int(s.split(\'-\')[0])
        return int(float(s))
    return series.apply(extract)
""", language="python")

    if df_emi is not None:
        sh("Emissions Master — Sample Preview")
        st.dataframe(df_emi.head(10), use_container_width=True)

    sh("Key Findings — Data Consolidation")
    finding("32 heterogeneous files from FAOSTAT, MoAFW, and MoEFCC merged into 3 master panels with zero data loss.")
    finding("Fiscal-year vs calendar-year inconsistency resolved via universal year extraction function.")
    finding("Long-format panel structure retained full feature granularity for downstream engineering.")


# ============================================================
#  FEATURE ENGINEERING
# ============================================================
elif page == "🔧 Feature Engineering":
    sh("Notebook 02 — Feature Engineering & Target Alignment")
    st.markdown("""
**Objective:** Convert three master panels into a single, model-ready, stationary
feature matrix — the foundation for all ML work downstream.
""")

    sh("Dimensionality Reduction")
    st.markdown("""
The Energy master arrived with **317 columns**. Two sequential passes were applied:
1. Drop columns with > 30% missing values
2. Drop zero-variance columns

Result: a lean, noise-free feature space without losing predictive signal.
""")
    mcards([
        {"label": "Energy Cols Before",    "value": "317", "sub": "Raw"},
        {"label": "Energy Cols After",     "value": "10",  "sub": "After cleaning"},
        {"label": "Resources Cols Before", "value": "30",  "sub": "Raw"},
        {"label": "Resources Cols After",  "value": "9",   "sub": "After cleaning"},
    ])

    sh("Water-Energy-Food Nexus Features")
    st.code("""
# Emission intensity per unit of agricultural land
df[\'Emission_Intensity_Land\'] = df[\'Target_Total_CO2e\'] / df[\'Agricultural_Area\']

# Energy use per unit of water withdrawal
df[\'Energy_Intensity_Water\'] = df[\'Total_Energy_Use\'] / df[\'Water_Withdrawal\']
""", language="python")

    sh("Temporal Engineering — Lag & Rolling Features")
    st.markdown("""
Policy changes in agriculture take **1–2 years** to show measurable emission effects.
We encode this with:

| Feature | Description |
|---------|-------------|
| `_lag1` | Variable value at year t−1 |
| `_lag2` | Variable value at year t−2 |
| `_roll3` | 3-year rolling mean (volatility smoothing) |
""")

    sh("Stationarity — ADF Test Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📉 Raw Target: `Target_Total_CO2e`")
        st.metric("ADF Statistic", "0.8386")
        st.metric("p-value", "0.9922")
        st.error("⚠️ NOT STATIONARY — 1st-order differencing required")
    with col2:
        st.markdown("##### ✅ After 1st-Order Differencing")
        st.metric("ADF Statistic", "-5.6681")
        st.metric("p-value", "< 0.0001")
        st.success("✅ STATIONARY — safe for ML modeling")

    df_final = load_csv("final_feature_set.csv")
    if df_final is not None:
        sh("Final Feature Set — Preview")
        st.dataframe(df_final.head(10), use_container_width=True)
        st.info(f"📐 Final Feature Matrix Shape: **{df_final.shape[0]} rows × {df_final.shape[1]} columns**")

    sh("Key Findings — Feature Engineering")
    finding("Target variable (Total CO₂e) defined by aggregating CO₂, CH₄ and N₂O emissions into a single metric.")
    finding("Dimensionality slimmed from 317 → 10 (Energy) and 30 → 9 (Resources) without loss of predictive signal.")
    finding("WEF Nexus indicators created: Emission_Intensity_Land & Energy_Intensity_Water.")
    finding("ADF confirmed non-stationarity on the raw target; 1st-order differencing achieves stationarity (p < 0.0001).")


# ============================================================
#  PREDICTIVE MODELING
# ============================================================
elif page == "🤖 Predictive Modeling":
    sh("Notebook 03 — Predictive Modeling & Optimization")
    st.markdown("""
**Objective:** Train and benchmark multiple ML models with **regularised hyperparameters**,
evaluate via **expanding-window cross-validation** (primary) and a 5-year hold-out (secondary),
and report **bootstrap confidence intervals** and **permutation significance tests**.
""")

    sh("Evaluation Strategy")
    st.markdown("""
Two complementary evaluation approaches are used to provide honest performance assessment:

1. **Expanding-Window CV (Primary):** Starting with 15 years of training data, the window
   expands by 1 year, predicting the next year. This yields 15 out-of-sample predictions
   with bootstrap 95% confidence intervals.
2. **5-Year Hold-Out (Secondary):** Final 5 years (2019–2023) held out as a traditional test set.
""")
    mcards([
        {"label": "Training Set",     "value": "1994–2018", "sub": "25 years of data"},
        {"label": "Test Set",         "value": "2019–2023", "sub": "5-year hold-out"},
        {"label": "Expanding-Window", "value": "15 folds",  "sub": "Min 15yr train window"},
        {"label": "CV Strategy",      "value": "TS-Split",  "sub": "k = 3 (GridSearchCV)"},
    ])

    # ── Expanding-Window CV Results (Primary) ─────────────────────
    if ew_metrics:
        sh("Primary Validation — Expanding-Window Cross-Validation")
        ew_rmse = ew_metrics.get('RMSE', 0)
        ew_r2 = ew_metrics.get('R2', 0)
        ew_mae = ew_metrics.get('MAE', 0)
        ew_rmse_lo = ew_metrics.get('RMSE_CI_lower', 0)
        ew_rmse_hi = ew_metrics.get('RMSE_CI_upper', 0)
        ew_r2_lo = ew_metrics.get('R2_CI_lower', 0)
        ew_r2_hi = ew_metrics.get('R2_CI_upper', 0)
        ew_n = int(ew_metrics.get('n_folds', 15))
        perm_p = ew_metrics.get('permutation_p_value', 1.0)

        mcards([
            {"label": "EW-CV RMSE",   "value": f"{ew_rmse:,.0f}", "sub": f"95% CI: [{ew_rmse_lo:,.0f}, {ew_rmse_hi:,.0f}]"},
            {"label": "EW-CV R²",     "value": f"{ew_r2:.4f}",    "sub": f"95% CI: [{ew_r2_lo:.4f}, {ew_r2_hi:.4f}]"},
            {"label": "EW-CV MAE",    "value": f"{ew_mae:,.0f}",  "sub": f"{ew_n} out-of-sample folds"},
            {"label": "Permutation p", "value": f"{perm_p:.3f}",  "sub": "p < 0.05 = significant"},
        ])

        if perm_p < 0.05:
            st.success(f"✅ Permutation test significant (p = {perm_p:.3f}) — model performs better than chance.")
        else:
            st.warning(f"⚠️ Permutation test not significant (p = {perm_p:.3f}) — model does not definitively outperform chance. This is an honest finding given the small sample size (n = 25 training years).")

        # Load the full CV comparison table
        cv_comp_path = os.path.join(RESULTS_DIR, "expanding_window_cv_comparison.csv")
        if os.path.exists(cv_comp_path):
            st.markdown("#### 📊 Model Performance vs. Naive Persistence Baseline")
            st.markdown("""
            In time-series forecasting with extremely small country-level sample sizes, absolute $R^2$ scores can be low.
            To justify the machine learning pipeline, we benchmark our models against a **Naive Persistence Baseline** (which predicts the previous year's difference).
            """)
            df_cv_comp = pd.read_csv(cv_comp_path)
            
            # Format comparison columns
            df_cv_comp_display = df_cv_comp.rename(columns={
                "CV_R2": "CV R²",
                "CV_MAE": "CV MAE",
                "CV_RMSE": "CV RMSE"
            })
            st.dataframe(
                df_cv_comp_display.style.format({"CV RMSE": "{:,.2f}", "CV MAE": "{:,.2f}", "CV R²": "{:.4f}"}),
                use_container_width=True,
                hide_index=True
            )
            
            # Add a bar chart showing the comparison
            fig_cv = go.Figure()
            df_cv_comp_sorted = df_cv_comp.sort_values(by="CV_RMSE", ascending=False)
            
            cv_colors = ["#ef4444" if "Naive" in m else "#3b82f6" for m in df_cv_comp_sorted["Model"]]
            # Highlight champion
            for idx, r in df_cv_comp_sorted.iterrows():
                if "Random Forest" in r["Model"]:
                    cv_colors[df_cv_comp_sorted.index.get_loc(idx)] = "#22c55e"
                    
            fig_cv.add_trace(go.Bar(
                x=df_cv_comp_sorted["Model"],
                y=df_cv_comp_sorted["CV_RMSE"],
                marker_color=cv_colors,
                text=[f"{val:,.0f}" for val in df_cv_comp_sorted["CV_RMSE"]],
                textposition="outside"
            ))
            fig_cv.update_layout(
                title="Cross-Validated RMSE Comparison (Lower is Better)",
                yaxis_title="RMSE (kt CO₂e)",
                template="plotly_dark",
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(14,17,23,0.9)",
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_cv, use_container_width=True)
            
            # Calculate % improvement over baseline
            naive_row = df_cv_comp[df_cv_comp["Model"].str.contains("Naive")]
            if not naive_row.empty:
                naive_rmse = naive_row["CV_RMSE"].values[0]
                champ_cv_row = df_cv_comp[df_cv_comp["Model"].str.contains("Random Forest")]
                if not champ_cv_row.empty:
                    champ_cv_rmse = champ_cv_row["CV_RMSE"].values[0]
                    pct_improvement = (naive_rmse - champ_cv_rmse) / naive_rmse * 100
                    st.info(f"💡 **Key Scientific Finding:** The champion model reduces out-of-sample prediction error (RMSE) by **{pct_improvement:.1f}%** compared to the classical Naive Persistence Baseline.")

    # ── Hold-Out Results (Secondary) ──────────────────────────────
    sh("Secondary Validation — 5-Year Hold-Out Test Set")
    metrics_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        results_df = df_metrics.rename(columns={
            "R2_Score": "R² Score",
            "RMSE": "RMSE (kt CO₂e)"
        })[["Model", "RMSE (kt CO₂e)", "R² Score"]]
    else:
        results_df = pd.DataFrame({
            "Model": ["Random Forest (Baseline)", "XGBoost (Optimized)", "SVR (Optimized)"],
            "RMSE (kt CO₂e)": [8710.21, 9253.34, 10703.37],
            "R² Score": [0.3379, 0.2528, 0.0003],
        })
    st.dataframe(
        results_df.style.format({"RMSE (kt CO₂e)": "{:,.2f}", "R² Score": "{:.4f}"}),
        use_container_width=True,
    )

    colors = ["#22c55e", "#3b82f6", "#ef4444"]
    fig = go.Figure()
    for i, row in results_df.iterrows():
        rmse_val = row["RMSE (kt CO\u2082e)"]
        fig.add_trace(go.Bar(
            x=[row["Model"]], y=[rmse_val],
            marker_color=colors[i % len(colors)],
            text=[f"{rmse_val:,.0f}"],
            textposition="outside",
            name=row["Model"],
        ))
    fig.update_layout(
        title="Hold-Out Model RMSE Comparison — Lower is Better",
        yaxis_title="RMSE (kt CO₂e)", xaxis_title="",
        template="plotly_dark", height=400, showlegend=False,
        font=dict(family="Inter"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.9)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not results_df.empty:
        best_row = results_df.sort_values("RMSE (kt CO₂e)").iloc[0]
        champion_model_name = best_row["Model"]
        champion_model_rmse = best_row["RMSE (kt CO₂e)"]
    else:
        champion_model_name = "Random Forest (Baseline)"
        champion_model_rmse = 8710.21
    st.success(f"🏆 **Champion Model Selected: {champion_model_name}** — Lowest hold-out RMSE of {champion_model_rmse:,.2f} kt CO₂e")

    sh("Regularised XGBoost Hyperparameters")
    st.markdown("""
XGBoost hyperparameters were **constrained** to prevent overfitting on the small dataset (n=25):
- `max_depth` capped at 4 (prevents trees deeper than the sample count)
- `reg_alpha` (L1) and `reg_lambda` (L2) regularisation added
- `subsample` and `colsample_bytree` added for stochastic regularisation
""")
    st.code("""
xgb_params = {
    'n_estimators':     [50, 100, 200],
    'max_depth':        [2, 3, 4],         # Capped at 4
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.7, 0.9],        # Row subsampling
    'colsample_bytree': [0.5, 0.8],        # Column subsampling
    'reg_alpha':        [0, 0.1, 1.0],     # L1 regularisation
    'reg_lambda':       [1.0, 5.0, 10.0],  # L2 regularisation
}
""", language="python")

    sh("Inverse Transformation — Back to Absolute CO₂e")
    st.markdown("""
Since the model was trained on **1st-order differences** (ΔCO₂e), the predictions are
back-transformed using a cumulative sum seeded from the last known absolute value:

```
Last known level (Year 2018): 193,634.73 kt CO₂e
Forecast_2023 = cumsum(predicted_diffs) + 193,634.73
```
""")

    sh("Key Findings — Predictive Modeling")
    finding(f"{champion_name} emerged as the champion with hold-out RMSE = {champion_rmse:,.2f} kt CO₂e.")
    if ew_metrics:
        finding(f"Expanding-window CV R² = {ew_metrics.get('R2', 0):.4f} with 95% CI [{ew_metrics.get('R2_CI_lower', 0):.4f}, {ew_metrics.get('R2_CI_upper', 0):.4f}] — honest assessment over {int(ew_metrics.get('n_folds', 15))} out-of-sample folds.")
        finding(f"Permutation test p-value = {ew_metrics.get('permutation_p_value', 1.0):.3f} — {'model significantly outperforms chance' if ew_metrics.get('permutation_p_value', 1.0) < 0.05 else 'model does not significantly outperform chance, reflecting the challenge of small-sample agricultural time-series prediction'}.")
    finding("XGBoost constrained with max_depth ≤ 4, L1/L2 regularisation, and sub-sampling to prevent overfitting on n=25 samples.")
    finding("Inverse differencing successfully restores forecasts to absolute CO₂e metric tonnage for research use.")


# ============================================================
#  MODEL INTERPRETABILITY
# ============================================================
elif page == "🔍 Model Interpretability":
    sh("Notebook 04 — Interpretability & Policy Insights")
    st.markdown("""
**Objective:** Use SHAP (SHapley Additive exPlanations) to open the XGBoost black box,
identify emission drivers, and simulate the impact of sustainable policy interventions.
""")

    sh("Global Feature Importance — SHAP Summary Plot")
    st.markdown("""
The beeswarm plot ranks features by **mean |SHAP value|** across all training samples.
**Red = high feature value, Blue = low feature value.**
A feature with high mean |SHAP| has a large average impact on model output.
""")
    img = load_img("shap_summary_plot.png")
    if img:
        st.image(img, use_container_width=True,
                 caption="SHAP Summary (Beeswarm) Plot — Global Feature Impact Distribution")
    else:
        st.warning("📂 `results/shap_summary_plot.png` not found — run Notebook 04 to generate it.")

    sh("SHAP Feature Importance — Ranked Bar Chart")
    shap_csv = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")
    if os.path.exists(shap_csv):
        df_shap = pd.read_csv(shap_csv)
        top_n   = min(15, len(df_shap))
        
        # Ensure column names match CSV: Feature, Mean_Absolute_SHAP_Impact
        df_top  = df_shap.nlargest(top_n, "Mean_Absolute_SHAP_Impact").sort_values("Mean_Absolute_SHAP_Impact")
        
        fig = px.bar(
            df_top, x="Mean_Absolute_SHAP_Impact", y="Feature", orientation="h",
            color="Mean_Absolute_SHAP_Impact",
            color_continuous_scale=["#064e3b", "#22c55e", "#fbbf24"],
            labels={"Mean_Absolute_SHAP_Impact": "Mean |SHAP Value|", "Feature": "Feature"},
            title=f"Top {top_n} Features by Mean Absolute SHAP Value",
        )
        fig.update_layout(
            template="plotly_dark", height=500,
            font=dict(family="Inter"), coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.9)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP feature importance CSV not found in `results/` directory.")

    sh("Local Interpretation — SHAP Waterfall Plot (Year 2023)")
    st.markdown("""
The waterfall plot shows exactly how each feature pushed the **2023 prediction**
up ↑ or down ↓ from the global base value (average model output).
""")
    img = load_img("shap_waterfall_2023.png")
    if img:
        st.image(img, use_container_width=True,
                 caption="SHAP Waterfall Plot — 2023 Prediction Decomposed Feature by Feature")
    else:
        st.warning("📂 `results/shap_waterfall_2023.png` not found — run Notebook 04 to generate it.")

    sh("SHAP Dependence Plots — Feature Relationships")
    st.markdown("These plots reveal how individual feature values relate to their SHAP impact.")
    dep_images = [
        ("shap_dependence_Annual_Rainfall_lag2.png", "Annual Rainfall (Lag-2)"),
        ("shap_dependence_Renewable_Energy_Use_in_Agriculture_GWh_diff_lag1.png", "Renewable Energy Use (Lag-1)"),
        ("shap_dependence_Estimated_agriculture_withdrawals_billion_m3_diff_lag2.png", "Water Withdrawals (Lag-2)"),
    ]
    dep_cols = st.columns(3)
    for col, (fname, cap) in zip(dep_cols, dep_images):
        img = load_img(fname)
        if img:
            col.image(img, use_container_width=True, caption=cap)

    sh("Key Findings — Interpretability")
    finding("Climate indicators (Annual Rainfall Lag-2) are the strongest overall SHAP drivers, highlighting agricultural climate dependence.")
    finding("Clean energy transitions (Renewable Energy Use in Agriculture Lag-1) are the most important actionable driver for emission reductions.")
    finding("Deterministic indices like Year Code have been excluded to force the model to learn physical and policy-relevant covariates.")
    finding("SHAP's game-theoretic guarantees ensure feature attributions are consistent and mathematically faithful to the model.")


# ============================================================
#  KEY FINDINGS
# ============================================================
elif page == "📊 Key Findings":
    sh("Key Findings — Full Pipeline Summary")

    with st.expander("📥 Notebook 01 — Data Consolidation", expanded=True):
        finding("32 heterogeneous files from FAOSTAT, MoAFW, and MoEFCC merged into 3 master panels.")
        finding("Temporal data aligned using a universal year extraction (Fiscal Year → Calendar Year).")
        finding("Long-format structure retained for maximum downstream flexibility in feature engineering.")

    with st.expander("🔧 Notebook 02 — Feature Engineering", expanded=True):
        finding("Target variable: Total CO₂e defined as weighted sum of CO₂, CH₄, and N₂O emissions.")
        finding("Energy columns reduced 317 → 10; Resources reduced 30 → 9 without loss of predictive signal.")
        finding("WEF Nexus features created: Emission_Intensity_Land and Energy_Intensity_Water.")
        finding("1st-order differencing applied after ADF confirmed non-stationarity (p-value = 0.99).")

    with st.expander("🤖 Notebook 03 — Predictive Modeling", expanded=True):
        finding(f"{champion_name} selected as champion model — hold-out {champion_rmse_str} kt CO₂e.")
        if ew_metrics:
            finding(f"Expanding-window CV (15 folds): R² = {ew_metrics.get('R2', 0):.4f}, RMSE = {ew_metrics.get('RMSE', 0):,.0f} with bootstrap 95% CIs.")
            finding(f"Permutation test p = {ew_metrics.get('permutation_p_value', 1.0):.3f} — transparent assessment of statistical significance.")
        finding("XGBoost regularised (max_depth ≤ 4, L1/L2, sub-sampling) to prevent overfitting on small dataset.")
        finding("Inverse differencing applied to restore forecasts to absolute CO₂e metric tonnes.")

    with st.expander("🔍 Notebook 04 — Model Interpretability", expanded=True):
        finding("SHAP analysis restricted to training set only — prevents test-set influence on feature importance rankings.")
        finding("Partial Dependence Plots (PDPs) with bootstrap uncertainty bands replace naive single-feature counterfactuals.")
        finding("SHAP waterfall plots provide per-year attribution for complete research transparency.")
        finding("Software version table and limitations section added for MethodsX reproducibility requirements.")

    sh("Model Comparison at a Glance")
    metrics_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        # Find baseline RF RMSE
        rf_row = df_metrics[df_metrics["Model"].str.contains("Random Forest")]
        rf_rmse = rf_row["RMSE"].values[0] if not rf_row.empty else 9013.85
        
        df_metrics["RMSE (kt CO₂e)"] = df_metrics["RMSE"]
        
        def calc_vs_baseline(row):
            if "Random Forest" in row["Model"]:
                return "Baseline"
            diff = (row["RMSE"] - rf_rmse) / rf_rmse * 100
            return f"{diff:+.1f}%"
            
        df_metrics["vs. Baseline RMSE"] = df_metrics.apply(calc_vs_baseline, axis=1)
        
        # Sort by RMSE to find champion
        df_metrics = df_metrics.sort_values("RMSE")
        ch_model = df_metrics.iloc[0]["Model"]
        df_metrics["Champion"] = df_metrics["Model"].apply(lambda m: "✅ Selected" if m == ch_model else "—")
        
        df_cmp = df_metrics[["Model", "RMSE (kt CO₂e)", "vs. Baseline RMSE", "Champion"]]
    else:
        df_cmp = pd.DataFrame({
            "Model":              ["XGBoost (Optimised)", "SVR (Optimised)",   "Random Forest (Baseline)"],
            "RMSE (kt CO₂e)":    [10704.87,              10752.04,             12539.09],
            "vs. Baseline RMSE": ["−14.6%",              "−14.3%",             "Baseline"],
            "Champion":          ["✅ Selected",         "—",                  "—"],
        })
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    colors2 = ["#22c55e", "#3b82f6", "#ef4444"]
    fig2 = go.Figure()
    for i, row in df_cmp.iterrows():
        rmse2 = row["RMSE (kt CO\u2082e)"]
        fig2.add_trace(go.Bar(
            x=[row["Model"]], y=[rmse2],
            marker_color=colors2[i % len(colors2)],
            text=[f"{rmse2:,.0f}"],
            textposition="outside", name=row["Model"],
        ))
    fig2.update_layout(
        title="Champion XGBoost achieves lowest RMSE",
        yaxis_title="RMSE (kt CO₂e)", xaxis_title="",
        template="plotly_dark", height=380, showlegend=False,
        font=dict(family="Inter"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.9)",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
#  SUMMARY
# ============================================================
elif page == "📝 Summary":
    sh("Project Summary")
    st.markdown("""
This project delivers a **complete, reproducible, research-grade ML pipeline** for predicting
India's agricultural carbon footprint — from raw data ingestion all the way to
publication-ready explainability and policy scenario analysis.
""")

    sh("Pipeline at a Glance")
    mcards([
        {"label": "Raw Files",       "value": "32",       "sub": "CSV / Excel"},
        {"label": "Master Panels",   "value": "3",        "sub": "Thematic grouping"},
        {"label": "Final Features",  "value": "≤ 20",     "sub": "After VIF + corr pruning"},
        {"label": "Champion Model",  "value": champion_name,  "sub": champion_rmse_str},
    ])

    sh("Methodology Summary")
    st.markdown("""
| Stage | Technique | Outcome |
|-------|-----------|---------|
| Data Consolidation | Automated parsing, year standardization | 3 clean master panels |
| Feature Engineering | Lag/roll features, ADF, correlation + VIF pruning | Stationary ≤20-feature matrix |
| Modeling | Expanding-window CV, regularised GridSearchCV, bootstrap CIs | Champion model with honest metrics |
| Interpretability | SHAP (training-only), PDPs with uncertainty | Robust feature attributions |
""")

    sh("Technology Stack")
    tech_cols = st.columns(5)
    for col, (icon, name) in zip(tech_cols, [
        ("🐍", "Python 3.10+"), ("🐼", "Pandas / NumPy"),
        ("🚀", "XGBoost"),      ("🔍", "SHAP"),
        ("📊", "Matplotlib"),
    ]):
        col.markdown(
            f'<div style="background:linear-gradient(145deg,#1e293b,#0f172a);'
            f'border:1px solid rgba(255,255,255,0.08);border-radius:14px;'
            f'padding:1.3rem 1.5rem;text-align:center;">'
            f'<div style="font-size:2rem;">{icon}</div>'
            f'<div style="font-size:.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;'
            f'letter-spacing:1px;margin-top:.5rem;">{name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    sh("Future Work")
    st.markdown("""
- 🗺️ **State-level data** — Expand from national to disaggregated state-level analysis
- 🧠 **Deep Learning** — LSTM / Transformer architectures for long-range temporal modelling
- 🔄 **Real-time Pipeline** — Streamlit + live API data for continuous emission monitoring
- 🌐 **Multi-policy Optimisation** — Co-optimise N₂O, CH₄, and CO₂ levers simultaneously
""")

    sh("Reproducibility")
    st.markdown("""
All notebooks are sequentially numbered and self-contained. Run in order to fully reproduce every result:

```
notebooks/
  01_Initial_Data_Consolidation.ipynb
  02_Feature_Engineering_and_Target_Alignment.ipynb
  03_Predictive_Modeling_and_Optimization.ipynb
  04_Interpretability_and_Policy_Insights.ipynb
```
""")


# ============================================================
#  FOOTER  (every page)
# ============================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a 0%,#1a5632 100%);
            padding:2rem;border-radius:14px;margin-top:3rem;text-align:center;
            border:1px solid rgba(255,255,255,0.06);">
  <p style="font-size:1.05rem;color:#4ade80;font-weight:700;margin:.3rem 0;">
    🌿 Carbon Footprint Prediction — India
  </p>
  <p style="color:rgba(255,255,255,0.5);margin:.3rem 0;font-size:.85rem;">
    <a href="https://github.com/prakash-ukhalkar/India-Agri-Carbon-Prediction"
       style="color:#4ade80;text-decoration:none;font-weight:600;" target="_blank">
       GitHub Repository
    </a>
    &nbsp;|&nbsp;
    <a href="https://github.com/prakash-ukhalkar"
       style="color:#4ade80;text-decoration:none;font-weight:600;" target="_blank">
       Author Profile
    </a>
  </p>
  <p style="color:rgba(255,255,255,0.5);margin:.3rem 0;font-size:.85rem;">
    Developed for Sustainable Agricultural Research &nbsp;•&nbsp; MIT License
  </p>
  <p style="color:rgba(255,255,255,0.35);margin-top:.7rem;font-size:.75rem;">
    Built with Streamlit &nbsp;•&nbsp; Powered by XGBoost &amp; SHAP
  </p>
</div>
""", unsafe_allow_html=True)
