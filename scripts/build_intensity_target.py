"""
Reframe the target as emission INTENSITY (tCO2e per hectare of gross sown
area) instead of the absolute level, to address a real limitation: the level
target is dominated by state agricultural SCALE (bigger states have more of
everything), so SHAP's top drivers were mostly scale proxies (pulses
production, sugarcane area, GSVA) rather than genuine efficiency/practice
differences. Dividing by Gross_Sown_Area removes the scale effect
mathematically, leaving "how much does this state emit per unit of
agricultural activity" - a more policy-relevant and less tautological
question.

Units: Target_Agri_GHG_ktCO2e (kilotonnes) / Gross_Sown_Area (thousand
hectares) simplifies exactly to tonnes CO2e per hectare (both numerator and
denominator carry a factor of 1000 that cancels).

CIRCULARITY GUARD (same principle as build_panel_features.py): since the
denominator IS Gross_Sown_Area, that column (and the near-identical
Net_Sown_Area/Gross_Irrigated_Area/Net_Irrigated_Area, all highly correlated
with it) must be excluded from the feature set here, on top of the existing
formula-input exclusions - otherwise the model could trivially reconstruct
the denominator's scale effect.
"""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel_with_target.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "panel_intensity_feature_set.csv")

TARGET_COL = "Target_Agri_GHG_ktCO2e"
INTENSITY_COL = "Target_Agri_GHG_Intensity_tCO2e_per_ha"

EXCLUDE_FORMULA_INPUTS = [
    "Rice_Area", "Rice_Production", "Rice_Yield",
    "Wheat_Area", "Wheat_Production", "Wheat_Yield",
    "Coarse_Cereals_Area", "Coarse_Cereals_Production", "Coarse_Cereals_Yield",
    "Total_Foodgrains_Area", "Total_Foodgrains_Production", "Total_Foodgrains_Yield",
    "Fertiliser_Consumption_NPK", "Agri_Electricity_Consumption_GWh",
    "Livestock_Cattle_Head", "Livestock_Buffalo_Head", "Livestock_Sheep_Head",
    "Livestock_Goat_Head", "Livestock_Pigs_Head", "Livestock_Horse_and_Ponies_Head",
    "Livestock_Mules_Head", "Livestock_Donkeys_Head", "Livestock_Camel_Head",
    "Livestock_Yaks_Head", "Livestock_Mithun_Head", "Livestock_Total_Livestock_Head",
]
# Additional exclusion specific to the intensity target: the denominator and
# its close correlates (area measures are highly collinear with each other).
EXCLUDE_DENOMINATOR_FAMILY = ["Gross_Sown_Area", "Net_Sown_Area", "Gross_Irrigated_Area", "Net_Irrigated_Area", "Cropping_Intensity"]
EXCLUDE_METADATA = ["AgriElectricity_component_is_2019_only", "NASA_Rainfall_mm"]

LAG_PERIODS = [1, 2]
ROLL_WINDOW = 3
MAX_VIF = 10.0
CORR_THRESHOLD = 0.90


def main():
    df = pd.read_csv(PANEL_PATH)
    df = df.sort_values(["State", "Year_Standardized"]).reset_index(drop=True)

    valid_area = df["Gross_Sown_Area"] > 0
    df[INTENSITY_COL] = np.where(valid_area, df[TARGET_COL] / df["Gross_Sown_Area"], np.nan)
    print(f"Intensity target populated for {df[INTENSITY_COL].notna().sum()} of {len(df)} rows")
    print(f"Intensity range: {df[INTENSITY_COL].min():.3f} - {df[INTENSITY_COL].max():.3f} tCO2e/ha")
    print(f"Median: {df[INTENSITY_COL].median():.3f} tCO2e/ha")

    ghg_component_cols = [c for c in df.columns if c.startswith("GHG_")]
    drop_cols = EXCLUDE_FORMULA_INPUTS + EXCLUDE_DENOMINATOR_FAMILY + EXCLUDE_METADATA + ghg_component_cols + [TARGET_COL]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=[INTENSITY_COL])

    base_feature_cols = [c for c in df.columns if c not in ("State", "Year_Standardized", INTENSITY_COL)
                         and df[c].dtype in ("float64", "int64")]
    print(f"\nExogenous base features (post circularity + denominator exclusion): {len(base_feature_cols)}")

    cols_to_lag = base_feature_cols + [INTENSITY_COL]
    g = df.groupby("State")
    for col in cols_to_lag:
        for lag in LAG_PERIODS:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        df[f"{col}_roll{ROLL_WINDOW}_mean"] = g[col].shift(1).rolling(ROLL_WINDOW).mean().reset_index(level=0, drop=True)

    required_history_cols = [f"{INTENSITY_COL}_lag{max(LAG_PERIODS)}", f"{INTENSITY_COL}_roll{ROLL_WINDOW}_mean"]
    before = len(df)
    df = df.dropna(subset=required_history_cols)
    print(f"Dropped {before - len(df)} rows lacking sufficient per-state history; {len(df)} rows remain")

    all_feature_cols = base_feature_cols + [f"{c}_lag{lag}" for c in cols_to_lag for lag in LAG_PERIODS] + \
        [f"{c}_roll{ROLL_WINDOW}_mean" for c in cols_to_lag]
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    feat_df = df[all_feature_cols]
    non_degenerate = [c for c in all_feature_cols if feat_df[c].notna().any() and feat_df[c].std(skipna=True) > 0]
    all_feature_cols = non_degenerate
    df = df.copy()
    for c in all_feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    corr_matrix = df[all_feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    target_corr = df[all_feature_cols].corrwith(df[INTENSITY_COL]).abs()
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
    print(f"After correlation pruning (|r|>{CORR_THRESHOLD}): {len(feat_cols_pruned)} features")

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

    final_cols = ["State", "Year_Standardized", INTENSITY_COL] + vif_cols
    final_df = df[final_cols]
    final_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} | shape {final_df.shape}")
    print(f"Selected features: {vif_cols}")


if __name__ == "__main__":
    main()
