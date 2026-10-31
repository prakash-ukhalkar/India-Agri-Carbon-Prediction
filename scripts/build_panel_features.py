"""
Feature engineering for the state x year panel, producing the final ML-ready
feature set for predicting Target_Agri_GHG_ktCO2e.

CIRCULARITY GUARD (the key design decision here): Target_Agri_GHG_ktCO2e is
built ARITHMETICALLY from a subset of the panel's own columns (see
build_tier1_ghg_target.py) - Rice_Area, Wheat/Coarse_Cereals Production,
Fertiliser_Consumption_NPK, Livestock headcounts, Agri_Electricity_Consumption.
Using those same columns (or anything that can reconstruct them, e.g.
Area+Yield implying Production) as CURRENT-YEAR features would let a model
trivially back out the target formula rather than learn anything predictive.
EXCLUDE_FORMULA_INPUTS below removes every column that is a direct input to
the target or algebraically reconstructs one (e.g. all of Rice_Area/
Production/Yield are excluded together, since any two imply the third).

The remaining, genuinely exogenous predictors are: climate (rainfall,
temperature, heat/cold wave days), groundwater, aggregate land-use (gross/net
sown & irrigated area, cropping intensity), economic activity (GSVA/NSVA
agriculture), livestock OUTPUT (meat/milk/egg/fish/wool production - distinct
from the excluded headcounts), and non-formula crops (cotton, sugarcane,
jute, oilseeds, pulses) with their own area/production/yield.

Also adds per-state lag1/lag2/roll3-mean features (including of the target
itself - autoregressive terms using PAST values are not circular) and a
two-stage correlation + VIF pruning pass, following the same methodology
already validated in notebook 02 for the (much smaller) national series.
"""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel_with_target.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "panel_final_feature_set.csv")

TARGET_COL = "Target_Agri_GHG_ktCO2e"

# Direct inputs to the target formula, and anything that algebraically
# reconstructs one of them (Area+Yield => Production; Total_Foodgrains sums
# in the excluded crops' production).
EXCLUDE_FORMULA_INPUTS = [
    "Rice_Area", "Rice_Production", "Rice_Yield",
    "Wheat_Area", "Wheat_Production", "Wheat_Yield",
    "Coarse_Cereals_Area", "Coarse_Cereals_Production", "Coarse_Cereals_Yield",
    "Total_Foodgrains_Area", "Total_Foodgrains_Production", "Total_Foodgrains_Yield",
    "Fertiliser_Consumption_NPK",
    "Agri_Electricity_Consumption_GWh",
    "Livestock_Cattle_Head", "Livestock_Buffalo_Head", "Livestock_Sheep_Head",
    "Livestock_Goat_Head", "Livestock_Pigs_Head", "Livestock_Horse_and_Ponies_Head",
    "Livestock_Mules_Head", "Livestock_Donkeys_Head", "Livestock_Camel_Head",
    "Livestock_Yaks_Head", "Livestock_Mithun_Head", "Livestock_Total_Livestock_Head",
]
# The GHG_* components literally sum to the target; the electricity-year flag
# and the redundant NASA rainfall (RBI's Annual_Rainfall_mm is the primary,
# meteorologically authoritative source) are metadata, not predictors.
EXCLUDE_METADATA = ["AgriElectricity_component_is_2019_only", "NASA_Rainfall_mm"]

LAG_PERIODS = [1, 2]
ROLL_WINDOW = 3
MAX_VIF = 10.0
CORR_THRESHOLD = 0.90


def main():
    df = pd.read_csv(PANEL_PATH)
    df = df.sort_values(["State", "Year_Standardized"]).reset_index(drop=True)

    ghg_component_cols = [c for c in df.columns if c.startswith("GHG_")]
    drop_cols = EXCLUDE_FORMULA_INPUTS + EXCLUDE_METADATA + ghg_component_cols
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    base_feature_cols = [c for c in df.columns if c not in ("State", "Year_Standardized", TARGET_COL)
                         and df[c].dtype in ("float64", "int64")]
    print(f"Exogenous base features (post circularity exclusion): {len(base_feature_cols)}")
    print(base_feature_cols)

    # Lag / rolling features, computed per state to avoid cross-state leakage.
    # Includes the target itself (autoregressive: PAST target values only).
    cols_to_lag = base_feature_cols + [TARGET_COL]
    g = df.groupby("State")
    for col in cols_to_lag:
        for lag in LAG_PERIODS:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        df[f"{col}_roll{ROLL_WINDOW}_mean"] = g[col].shift(1).rolling(ROLL_WINDOW).mean().reset_index(level=0, drop=True)

    # Drop rows without enough history per state for the deepest lag/roll feature.
    required_history_cols = [f"{TARGET_COL}_lag{max(LAG_PERIODS)}", f"{TARGET_COL}_roll{ROLL_WINDOW}_mean"]
    before = len(df)
    df = df.dropna(subset=required_history_cols)
    print(f"\nDropped {before - len(df)} rows lacking sufficient per-state history; {len(df)} rows remain")

    # Feature set for selection = lagged/rolled versions only (never current-year
    # base features for the direct target components - already excluded above;
    # current-year values of the remaining exogenous variables, e.g. this year's
    # rainfall, ARE legitimate contemporaneous predictors and are kept alongside
    # the lags).
    all_feature_cols = base_feature_cols + [f"{c}_lag{lag}" for c in cols_to_lag for lag in LAG_PERIODS] + \
        [f"{c}_roll{ROLL_WINDOW}_mean" for c in cols_to_lag]
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    # Drop feature columns that are entirely NaN or constant (uninformative / break VIF)
    feat_df = df[all_feature_cols]
    non_degenerate = [c for c in all_feature_cols if feat_df[c].notna().any() and feat_df[c].std(skipna=True) > 0]
    dropped_degenerate = set(all_feature_cols) - set(non_degenerate)
    if dropped_degenerate:
        print(f"\nDropped {len(dropped_degenerate)} all-NaN/constant feature columns: {sorted(dropped_degenerate)}")
    all_feature_cols = non_degenerate

    # Any remaining NaNs (a few sparse variables, e.g. groundwater snapshot years)
    # are median-imputed per column - documented, not silently dropped further.
    for c in all_feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Stage 1: correlation pruning (|r| > 0.90), drop the member less correlated with target
    corr_matrix = df[all_feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    target_corr = df[all_feature_cols].corrwith(df[TARGET_COL]).abs()
    cols_to_drop_corr = set()
    for col in upper_tri.columns:
        partners = upper_tri.index[upper_tri[col] > CORR_THRESHOLD].tolist()
        for partner in partners:
            if col in cols_to_drop_corr or partner in cols_to_drop_corr:
                continue
            if target_corr.get(col, 0) >= target_corr.get(partner, 0):
                cols_to_drop_corr.add(partner)
            else:
                cols_to_drop_corr.add(col)
    feat_cols_pruned = [c for c in all_feature_cols if c not in cols_to_drop_corr]
    print(f"\nAfter correlation pruning (|r|>{CORR_THRESHOLD}): {len(feat_cols_pruned)} features ({len(cols_to_drop_corr)} dropped)")

    # Stage 2: iterative VIF removal
    vif_cols = feat_cols_pruned.copy()
    for _ in range(100):
        if len(vif_cols) <= 2:
            break
        X = df[vif_cols].values
        X_c = np.column_stack([np.ones(len(X)), X])
        vifs = [variance_inflation_factor(X_c, i) for i in range(1, X_c.shape[1])]
        max_vif = max(vifs)
        if max_vif <= MAX_VIF:
            break
        vif_cols.pop(vifs.index(max_vif))
    print(f"After VIF removal (VIF<{MAX_VIF}): {len(vif_cols)} features")

    selected_features = vif_cols
    final_cols = ["State", "Year_Standardized", TARGET_COL] + selected_features
    final_df = df[final_cols]

    final_df.to_csv(OUT_PATH, index=False)
    print(f"\nFinal selected features ({len(selected_features)}):")
    for f in selected_features:
        print(f"  - {f}")
    print(f"\nSaved: {OUT_PATH} | shape {final_df.shape}")


if __name__ == "__main__":
    main()
