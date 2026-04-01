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
        {"label": "Champion Model",      "value": "XGBoost", "sub": "RMSE: 10,704"},
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
**Objective:** Train and benchmark multiple ML models on a temporally correct train/test
split and select the best model for downstream interpretability.
""")

    sh("Chronological Train–Test Split")
    st.markdown("""
To respect time-ordering, a **chronological hold-out strategy** is used — data is never
shuffled. Cross-validation uses **TimeSeriesSplit (k = 3)** inside GridSearchCV.
""")
    mcards([
        {"label": "Training Set",   "value": "1993–2018", "sub": "~26 years of data"},
        {"label": "Test Set",       "value": "2019–2023", "sub": "5-year hold-out"},
        {"label": "CV Strategy",    "value": "TS-Split",  "sub": "k = 3 folds, no lookahead"},
    ])

    sh("Model Performance Comparison")
    results_df = pd.DataFrame({
        "Model": ["XGBoost (Optimised)", "SVR (Optimised)", "Random Forest (Baseline)"],
        "RMSE (kt CO₂e)": [10704.87, 10752.04, 12539.09],
        "R² Score": [-0.000002, -0.008834, -0.372050],
    })
    st.dataframe(
        results_df.style.format({"RMSE (kt CO₂e)": "{:,.2f}", "R² Score": "{:.6f}"}),
        use_container_width=True,
    )

    colors = ["#22c55e", "#3b82f6", "#ef4444"]
    fig = go.Figure()
    for i, row in results_df.iterrows():
        rmse_val = row["RMSE (kt CO\u2082e)"]
        fig.add_trace(go.Bar(
            x=[row["Model"]], y=[rmse_val],
            marker_color=colors[i],
            text=[f"{rmse_val:,.0f}"],
            textposition="outside",
            name=row["Model"],
        ))
    fig.update_layout(
        title="Model RMSE Comparison — Lower is Better",
        yaxis_title="RMSE (kt CO₂e)", xaxis_title="",
        template="plotly_dark", height=400, showlegend=False,
        font=dict(family="Inter"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.9)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("🏆 **Champion Model Selected: XGBoost (Optimised)** — Lowest RMSE of 10,704.87 kt CO₂e")

    sh("Best XGBoost Hyperparameters")
    st.code("""
XGBRegressor(
    learning_rate    = 0.01,
    max_depth        = 3,
    n_estimators     = 50,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
)
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
    finding("XGBoost emerged as the champion with RMSE = 10,704.87 kt CO₂e on the 5-year hold-out test set.")
    finding("Negative R² across all models is expected for differenced environmental time-series with only ~31 samples.")
    finding("TimeSeriesSplit CV with k=3 folds strictly prevents look-ahead bias during hyperparameter tuning.")
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
        ("shap_dependence_Target_Total_CO2e_lag1.png", "CO₂e Lag-1"),
        ("shap_dependence_Value_lag2.png",               "Value Lag-2"),
        ("shap_dependence_Year Code.png",                "Year Code"),
    ]
    dep_cols = st.columns(3)
    for col, (fname, cap) in zip(dep_cols, dep_images):
        img = load_img(fname)
        if img:
            col.image(img, use_container_width=True, caption=cap)

    sh("Key Findings — Interpretability")
    finding("Historical emission lags (lag-1, lag-2) are the strongest SHAP drivers — confirming strongly autoregressive behaviour.")
    finding("Fertiliser usage (Value feature) is the most actionable controllable driver — a prime policy intervention target.")
    finding("Year Code captures the secular upward trend in India's agricultural carbon emissions over the decades.")
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
        finding("XGBoost (Optimised) selected as champion model — RMSE: 10,704.87 kt CO₂e.")
        finding("TimeSeriesSplit Cross-Validation (k=3) strictly prevents look-ahead data leakage.")
        finding("Inverse differencing applied to restore forecasts to absolute CO₂e metric tonnes.")

    with st.expander("🔍 Notebook 04 — Model Interpretability", expanded=True):
        finding("Emission lag-1 is the single largest SHAP driver — emissions are strongly autoregressive.")
        finding("Nitrogen fertiliser use is the most actionable policy lever for emissions reduction.")
        finding("Scenario analysis: 20% renewable energy shift projected to reduce forecast emissions by ~8%.")
        finding("SHAP waterfall plots provide per-year attribution for complete research transparency.")

    sh("Model Comparison at a Glance")
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
            marker_color=colors2[i],
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
        {"label": "Final Features",  "value": "22",       "sub": "Post-engineering"},
        {"label": "Champion Model",  "value": "XGBoost",  "sub": "RMSE 10,704.87"},
    ])

    sh("Methodology Summary")
    st.markdown("""
| Stage | Technique | Outcome |
|-------|-----------|---------|
| Data Consolidation | Automated parsing, year standardization | 3 clean master panels |
| Feature Engineering | Lag/roll features, ADF, WEF-Nexus | Stationary 22-feature matrix |
| Modeling | TimeSeriesSplit CV, GridSearchCV | XGBoost champion selected |
| Interpretability | SHAP TreeExplainer | Policy-ready feature attributions |
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
