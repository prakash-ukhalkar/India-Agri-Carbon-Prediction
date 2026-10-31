"""
Build a "dynamic-only" sub-target: Rice CH4 + Residue Burning CH4/N2O +
Fertilizer Soil N2O, EXCLUDING Livestock and Agri-Electricity CO2.

Why: the full target's temporal signal is diluted by the livestock and
electricity components, which are smoothly interpolated/held-constant per
state (only 1-2 real census/report years each) and therefore carry almost no
genuine year-to-year variation. The remaining three components DO have real
annual variation driven by weather and crop-cycle decisions (rainfall
affecting rice area/yield, temperature affecting residue burning timing,
fertilizer application varying with input costs and monsoon outlook). If
there is any genuine temporal signal to find in this panel, it's more likely
to surface here than in a target ~40%+ dominated by artificially smooth
components.

Same circularity guard as build_panel_features.py.
"""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel_with_target.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "panel_dynamic_feature_set.csv")

DYNAMIC_COMPONENTS = ["GHG_Rice_CH4_ktCO2e", "GHG_ResidueBurning_CH4_ktCO2e",
                      "GHG_ResidueBurning_N2O_ktCO2e", "GHG_FertilizerSoil_N2O_ktCO2e"]
DYNAMIC_COL = "Target_Dynamic_GHG_ktCO2e"

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
EXCLUDE_METADATA = ["AgriElectricity_component_is_2019_only", "NASA_Rainfall_mm"]
LAG_PERIODS = [1, 2]
ROLL_WINDOW = 3
MAX_VIF = 10.0
CORR_THRESHOLD = 0.90


def main():
    df = pd.read_csv(PANEL_PATH).sort_values(["State", "Year_Standardized"]).reset_index(drop=True)
    df[DYNAMIC_COL] = df[DYNAMIC_COMPONENTS].sum(axis=1, min_count=1)
    print(f"Dynamic-only target populated for {df[DYNAMIC_COL].notna().sum()} of {len(df)} rows")

    ghg_component_cols = [c for c in df.columns if c.startswith("GHG_")]
    drop_cols = EXCLUDE_FORMULA_INPUTS + EXCLUDE_METADATA + ghg_component_cols + ["Target_Agri_GHG_ktCO2e"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols).dropna(subset=[DYNAMIC_COL]).reset_index(drop=True)

    base_feature_cols = [c for c in df.columns if c not in ("State", "Year_Standardized", DYNAMIC_COL)
                         and df[c].dtype in ("float64", "int64")]

    cols_to_lag = base_feature_cols + [DYNAMIC_COL]
    g = df.groupby("State")
    for col in cols_to_lag:
        for lag in LAG_PERIODS:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        df[f"{col}_roll{ROLL_WINDOW}_mean"] = g[col].shift(1).rolling(ROLL_WINDOW).mean().reset_index(level=0, drop=True)

    required = [f"{DYNAMIC_COL}_lag{max(LAG_PERIODS)}", f"{DYNAMIC_COL}_roll{ROLL_WINDOW}_mean"]
    before = len(df)
    df = df.dropna(subset=required)
    print(f"Dropped {before - len(df)} rows lacking history; {len(df)} rows remain")

    all_feature_cols = base_feature_cols + [f"{c}_lag{lag}" for c in cols_to_lag for lag in LAG_PERIODS] + \
        [f"{c}_roll{ROLL_WINDOW}_mean" for c in cols_to_lag]
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]
    non_degenerate = [c for c in all_feature_cols if df[c].notna().any() and df[c].std(skipna=True) > 0]
    all_feature_cols = non_degenerate
    df = df.copy()
    for c in all_feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    corr_matrix = df[all_feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    target_corr = df[all_feature_cols].corrwith(df[DYNAMIC_COL]).abs()
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
    pruned = [c for c in all_feature_cols if c not in cols_to_drop_corr]
    print(f"After correlation pruning: {len(pruned)} features")

    vif_cols = pruned.copy()
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
    print(f"After VIF removal: {len(vif_cols)} features")

    final_df = df[["State", "Year_Standardized", DYNAMIC_COL] + vif_cols]
    final_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} | shape {final_df.shape}")
    print(f"Features: {vif_cols}")


if __name__ == "__main__":
    main()
