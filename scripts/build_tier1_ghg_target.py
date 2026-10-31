"""
Build a bottom-up, IPCC Tier-1 state x year agricultural GHG emissions target
from state_year_panel.csv, replacing the national top-down FAOSTAT target used
in the original (pre-redesign) pipeline.

Components (each converted to CO2e via IPCC AR5 GWP-100: CH4 x28, N2O x265):
  1. Rice cultivation CH4        - from Rice_Area
  2. Crop residue burning CH4+N2O - from Rice/Wheat/Coarse_Cereals Production
  3. Livestock enteric fermentation + manure management CH4/N2O - from 2019
     Livestock Census headcounts (single year - see LIMITATIONS below)
  4. Synthetic fertilizer soil N2O (direct + indirect) - from Fertiliser_Consumption_NPK
  5. Agricultural electricity CO2 - from 2019 CEA agri electricity consumption
     (single year - see LIMITATIONS below)

All emission factors are standard IPCC 2006 Guidelines (Vol. 4) Tier-1
defaults for the Indian subcontinent / South Asia region, chosen because
finer (state-specific, breed-specific, water-regime-specific) factors are not
available from our current data sources. This is explicitly a Tier-1
approximation, appropriate for a first defensible target - not a substitute
for India's official Tier-2/3 national inventory methodology.

LIMITATIONS (state these plainly wherever the data are used):
  - Livestock headcounts and agri-electricity consumption are single-year
    (2019) snapshots. Applying them to every year in the panel means the
    livestock and agri-energy emission COMPONENTS are held CONSTANT across
    years for a given state - i.e. only the rice/residue/fertilizer
    components carry real year-to-year variation. This is flagged via the
    *_is_2019_only columns; getting additional Livestock Census editions
    (1997/2003/2007/2012) and CEA General Review editions for other years
    would let these vary properly and should be the next data-acquisition
    priority if this turns out to matter for the model (e.g. if year and
    state effects become confounded).
  - Fertiliser_Consumption_NPK is combined N+P+K weight, not nitrogen alone.
    An assumed N-share of 0.64 (approx. India's recent all-India N:P:K
    consumption ratio) is applied - a real approximation, not a measurement.
  - Rice/wheat/residue burning fractions use fixed national defaults, not
    state-specific values, even though burning practice varies hugely by
    state (e.g. Punjab/Haryana rice burning rates are far higher than the
    national average used here).
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel_with_target.csv")

# GWP set. India's national inventories (BUR) have historically used IPCC SAR
# GWP-100 (CH4 21, N2O 310); set GWP_SET="AR5" for CH4 28 / N2O 265. The
# default is SAR so that totals are directly comparable with the BUR.
GWP_SET = os.environ.get("GWP_SET", "SAR")
GWP_CH4, GWP_N2O = {"SAR": (21.0, 310.0), "AR5": (28.0, 265.0)}[GWP_SET]  # SAR = values in the factor file

# --- 1. Rice cultivation CH4 ---
# National-average blended emission factor (kg CH4/ha/season) approximating
# India's mixed water regimes (continuously/intermittently flooded, rainfed),
# in the range IPCC 2006 Vol.4 Ch.5 Table 5.11 baseline EFs (1.30 kg/ha/day)
# would imply over a ~100-day season once water-regime scaling factors are
# applied. State-specific water-regime shares are not available.
# v3: India-specific factors are read from data/processed/state_panel/india_specific_emission_factors.csv, written by
# scripts/derive_india_emission_factors.py with provenance (area-weighted over water regimes for rice; census-composition-
# weighted for cattle and buffalo). v2 used IPCC Tier-1 defaults and overstated livestock (x1.36) and fertiliser soil N2O
# (x2.08) against the GHG Platform India state inventory. v1 used unscaled defaults and included electricity.
FACTORS = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "state_panel", "india_specific_emission_factors.csv"))
FACTORS = dict(zip(FACTORS["parameter"], FACTORS["value"]))
EF_RICE_CH4_KG_PER_HA = FACTORS["rice_ch4_effective"]

# --- 2. Crop residue burning (IPCC 2006 Vol.4 Ch.2 Table 2.5 defaults) ---
# (Residue:Product Ratio, Dry Matter Fraction, assumed Fraction Burnt)
RESIDUE_PARAMS = {
    "Rice_Production": {"rpr": 1.5, "dmf": 0.85, "frac_burnt": 0.25},
    "Wheat_Production": {"rpr": 1.3, "dmf": 0.83, "frac_burnt": 0.10},
    "Coarse_Cereals_Production": {"rpr": 2.0, "dmf": 0.78, "frac_burnt": 0.25},
}
EF_RESIDUE_CH4_G_PER_KG_DM = 2.7
EF_RESIDUE_N2O_G_PER_KG_DM = 0.07

# --- 3. Livestock (kg/head/year): (enteric CH4, manure CH4, manure N2O) ---
# IPCC 2006 Vol.4 Ch.10 Tables 10.10-10.14, Indian-subcontinent defaults,
# blended across sub-categories (e.g. dairy/non-dairy cattle) not
# distinguished in the Livestock Census headcount columns we have.
def _lf(prefix):
    return (FACTORS[f"{prefix}_enteric"], FACTORS[f"{prefix}_manure_ch4"], FACTORS[f"{prefix}_manure_n2o"])


# census column -> (enteric kg CH4/head/yr, manure CH4 kg/head/yr, manure N2O kg/head/yr). Mules take the donkey factor
# and yaks/mithun the cattle factor (assumptions; <0.5% of livestock emissions). Poultry is not in the panel
# (manure N2O ~0.6 Mt CO2e nationally, <0.2% of the target).
LIVESTOCK_FACTORS = {
    "Livestock_Cattle_Head": _lf("cattle"), "Livestock_Buffalo_Head": _lf("buffalo"),
    "Livestock_Sheep_Head": _lf("sheep"), "Livestock_Goat_Head": _lf("goat"), "Livestock_Pigs_Head": _lf("pig"),
    "Livestock_Horse_and_Ponies_Head": _lf("horse"), "Livestock_Mules_Head": _lf("donkey"),
    "Livestock_Donkeys_Head": _lf("donkey"), "Livestock_Camel_Head": _lf("camel"),
    "Livestock_Yaks_Head": _lf("cattle"), "Livestock_Mithun_Head": _lf("cattle"),
}

# --- 4. Fertilizer soil N2O (IPCC 2006 Vol.4 Ch.11 Tier-1 defaults) ---
ASSUMED_N_SHARE_OF_NPK = 0.64   # approx. India's N share of N+P+K consumption (no source found; an assumption)
EF1_DIRECT = FACTORS["fert_direct_ef"]            # kg N2O-N / kg N applied (India-specific)
FRAC_GASF = FACTORS["fert_frac_volatilised"]      # fraction volatilised
EF4_VOLATILIZATION = FACTORS["fert_ef_deposition"]
FRAC_LEACH = FACTORS["fert_frac_leached"]         # fraction leached
EF5_LEACHING = FACTORS["fert_ef_leaching"]

# --- 5. Agricultural electricity CO2 ---
# CEA's approximate all-India grid average CO2 emission factor, FY2019-20
# (CEA CO2 Baseline Database order of magnitude). A single national factor is
# used since state-wise grid emission factors are not available.
GRID_EF_KG_CO2_PER_KWH = 0.82


def main():
    df = pd.read_csv(PANEL_PATH)

    # 1. Rice CH4 (Rice_Area in thousand hectares)
    rice_ch4_kt = (df["Rice_Area"] * 1000 * EF_RICE_CH4_KG_PER_HA) / 1e6
    df["GHG_Rice_CH4_ktCO2e"] = rice_ch4_kt * GWP_CH4

    # 2. Crop residue burning (Production in thousand tonnes).
    # v3.1: a crop a state NEVER reports (e.g. wheat in Kerala/Tamil Nadu/Goa) is a structural zero, not a gap. v3.0 let
    # that NaN wipe out the whole component, discarding rice-straw burning for those states (understated 13-29%).
    # A state that reports none of the three crops (small UTs) has no data -> component missing, not zero.
    # A crop that the state reports in other years but not this year is a real gap -> component missing.
    never = pd.DataFrame({c: df.groupby("State")[c].transform(lambda x: x.isna().all()) for c in RESIDUE_PARAMS})
    no_crop_data = never.all(axis=1)
    gap = pd.DataFrame({c: df[c].isna() & ~never[c] for c in RESIDUE_PARAMS}).any(axis=1)
    residue_missing = no_crop_data | gap
    residue_ch4_kt = pd.Series(0.0, index=df.index)
    residue_n2o_kt = pd.Series(0.0, index=df.index)
    for col, p in RESIDUE_PARAMS.items():
        production_t = df[col].where(~never[col], 0.0) * 1000  # thousand tonnes -> tonnes
        dm_burnt_t = production_t * p["rpr"] * p["dmf"] * p["frac_burnt"]
        residue_ch4_kt += (dm_burnt_t * EF_RESIDUE_CH4_G_PER_KG_DM) / 1e6  # kg -> kt
        residue_n2o_kt += (dm_burnt_t * EF_RESIDUE_N2O_G_PER_KG_DM) / 1e6
    df["GHG_ResidueBurning_CH4_ktCO2e"] = (residue_ch4_kt * GWP_CH4).where(~residue_missing)
    df["GHG_ResidueBurning_N2O_ktCO2e"] = (residue_n2o_kt * GWP_N2O).where(~residue_missing)
    df["Flag_Residue"] = np.where(residue_missing, "missing", np.where(never.any(axis=1), "structural_zero_assumed", "observed"))
    df["Flag_Rice"] = np.where(df["Rice_Area"].notna(), "observed", "missing")

    # 3. Livestock (headcounts: interpolated 2012-2019 from two real Census
    # points via scripts/interpolate_livestock.py, held constant outside that
    # range - NOT zero-filled, which previously created a fake spike/dip at
    # the single-year boundary. NaN only remains for states with no census
    # match at all, which should genuinely drop out of the sum (min_count=1
    # below), not be treated as zero emissions.
    livestock_ch4_kt = pd.Series(0.0, index=df.index)
    livestock_n2o_kt = pd.Series(0.0, index=df.index)
    any_livestock_data = pd.Series(False, index=df.index)
    for col, (enteric, manure_ch4, manure_n2o) in LIVESTOCK_FACTORS.items():
        if col not in df.columns:
            continue
        heads = df[col]
        any_livestock_data |= heads.notna()
        heads = heads.fillna(0)
        livestock_ch4_kt += (heads * (enteric + manure_ch4)) / 1e6  # kg -> kt
        livestock_n2o_kt += (heads * manure_n2o) / 1e6
    df["GHG_Livestock_CH4_ktCO2e"] = livestock_ch4_kt.where(any_livestock_data) * GWP_CH4
    df["GHG_Livestock_N2O_ktCO2e"] = livestock_n2o_kt.where(any_livestock_data) * GWP_N2O

    # 4. Fertilizer soil N2O (Fertiliser_Consumption_NPK in thousand tonnes).
    # RBI Table 89 only reports 2013-2021 by the source's own design (not a
    # parsing gap). Held constant outside that window (2013 value backward to
    # 2004, 2021 value forward to 2024) for the same reason as livestock/
    # electricity above - avoids a level-shift artifact at the window's edges.
    df = df.sort_values(["State", "Year_Standardized"])
    # v3.2: nitrogen applied (kt N). Observed NPK (RBI, 2013-2021) x a STATE-SPECIFIC N share calibrated on the 2014-2018
    # overlap with the compiled N series (replaces the unsourced national 0.64; national weighted share is ~0.66);
    # 2004-2012 taken from the compiled N series (flag compiled_secondary: secondary compilation, agreement with GHGPI
    # there is not independent); anything left (2022-2024) held constant at the state's last value.
    FERT_PATTERN = os.environ.get("FERT_PATTERN", "primary")   # "primary" (released) or "compiled" (sensitivity only)
    official = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "state_panel", "fertiliser_national_official.csv")).set_index("Year_Standardized")
    if FERT_PATTERN == "primary":
        # PRIMARY SOURCES ONLY: nitrogen applied = state N+P+K (RBI Handbook, 2013-2021) x the official All-India N/NPK ratio of
        # that year. Outside the RBI window the state's N+P+K of the nearest observed year gives the state pattern (2013
        # backwards, 2021 forwards), scaled to the official national N of each year (2004-2023); 2024 keeps the 2023 level.
        npk = df["Fertiliser_Consumption_NPK"]
        yrs = range(int(df["Year_Standardized"].min()), int(df["Year_Standardized"].max()) + 1)
        n_nat = official["N_ktN"].reindex(yrs).ffill()
        pattern = npk.groupby(df["State"]).transform(lambda x: x.ffill().bfill())           # state NPK of nearest observed year
        pat_sum = pattern.groupby(df["Year_Standardized"]).transform("sum")
        n_applied_kt = pattern / pat_sum * df["Year_Standardized"].map(n_nat)
        ratio_nat = (official["N_ktN"] / official["NPK_kt"]).reindex(yrs).ffill().bfill()
        df["N_share_of_NPK"] = df["Year_Standardized"].map(ratio_nat)                          # national N/NPK of that year
        df["N_scale_to_national"] = 1.0
        df["Flag_Fertiliser"] = np.where(npk.notna(), "observed",
                                         np.where(df["Year_Standardized"] <= int(official.index.max()), "national_downscaled", "held_constant"))
        df.loc[n_applied_kt.isna(), "Flag_Fertiliser"] = "missing"
        df["Fertiliser_N_applied_ktN"] = n_applied_kt
    else:
        comp = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "state_panel", "fertiliser_N_compiled_2004_2018.csv"))
        df = df.merge(comp, on=["State", "Year_Standardized"], how="left")
        npk = df["Fertiliser_Consumption_NPK"]
        ov = df[df["Year_Standardized"].between(2014, 2018) & npk.notna() & df["N_compiled_ktN"].notna() & (npk > 0)]
        nat_share = ov["N_compiled_ktN"].sum() / ov["Fertiliser_Consumption_NPK"].sum()
        st_share = (ov.groupby("State")["N_compiled_ktN"].sum() / ov.groupby("State")["Fertiliser_Consumption_NPK"].sum()).clip(0.3, 1.0)
        df["N_share_of_NPK"] = df["State"].map(st_share).fillna(nat_share)
        n_obs = npk * df["N_share_of_NPK"]
        n_applied_kt = n_obs.copy()
        fill = n_obs.isna() & df["N_compiled_ktN"].notna()
        n_applied_kt[fill] = df.loc[fill, "N_compiled_ktN"]
        df["Flag_Fertiliser"] = np.where(n_obs.notna(), "observed", np.where(fill, "compiled_secondary", "held_constant"))
        n_applied_kt = n_applied_kt.groupby(df["State"]).transform(lambda x: x.ffill().bfill())
        df.loc[n_applied_kt.isna(), "Flag_Fertiliser"] = "missing"
        # National control: rescale the state values each year so they sum to the official All-India N (table in data/raw,
        # parsed by parse_fertiliser_national_official.py; matches FAOSTAT within 1.5%). Corrects the compiled series'
        # overstatement in 2004-2006 (4-19%), a 6% dip in 2013, and extends the national level to 2022-2023. 2024 has no
        # official figure and keeps the 2023 national level. State shares for 2022-2024 are those of the last observed year.
        nat = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "state_panel", "fertiliser_national_official.csv")).set_index("Year_Standardized")["N_ktN"]
        nat = nat.reindex(range(int(df["Year_Standardized"].min()), int(df["Year_Standardized"].max()) + 1)).ffill()
        state_sum = n_applied_kt.groupby(df["Year_Standardized"]).transform("sum")
        df["N_scale_to_national"] = df["Year_Standardized"].map(nat) / state_sum
        n_applied_kt = n_applied_kt * df["N_scale_to_national"]
        official_years = set(pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "state_panel", "fertiliser_national_official.csv"))["Year_Standardized"])
        down = (df["Flag_Fertiliser"] == "held_constant") & df["Year_Standardized"].isin(official_years)
        df.loc[down, "Flag_Fertiliser"] = "national_downscaled"
        df["Fertiliser_N_applied_ktN"] = n_applied_kt
    direct_n2o_n_kt = n_applied_kt * EF1_DIRECT
    volat_n2o_n_kt = n_applied_kt * FRAC_GASF * EF4_VOLATILIZATION
    leach_n2o_n_kt = n_applied_kt * FRAC_LEACH * EF5_LEACHING
    total_n2o_n_kt = direct_n2o_n_kt + volat_n2o_n_kt + leach_n2o_n_kt
    fertilizer_n2o_kt = total_n2o_n_kt * (44.0 / 28.0)
    df["GHG_FertilizerSoil_N2O_ktCO2e"] = fertilizer_n2o_kt * GWP_N2O

    # 5. Agricultural electricity CO2 (GWh). Only one year (2019) of data was
    # obtainable - held constant across all years within each state (ffill/
    # bfill) rather than left at 0 elsewhere, which previously created a fake
    # spike/dip at the single-year boundary. A held-constant approximation
    # is still a real limitation (state agri-electricity use does trend over
    # two decades) - flagged via AgriElectricity_component_is_2019_only.
    df = df.sort_values(["State", "Year_Standardized"])
    df["Agri_Electricity_Consumption_GWh"] = df.groupby("State")["Agri_Electricity_Consumption_GWh"].transform(lambda s: s.ffill().bfill())
    df["GHG_AgriElectricity_CO2_ktCO2e"] = df["Agri_Electricity_Consumption_GWh"] * GRID_EF_KG_CO2_PER_KWH
    df["AgriElectricity_component_is_2019_only"] = True

    # v2: grid electricity CO2 is an Energy-sector (IPCC 1.A.4.c / 1A) source in
    # national inventories, not Agriculture, so it is excluded from the target and
    # kept as a separate column. v1 included it (~17% of the v1 target).
    df = df.rename(columns={"GHG_AgriElectricity_CO2_ktCO2e": "Memo_AgriElectricity_CO2_ktCO2e"})
    component_cols = [c for c in df.columns if c.startswith("GHG_")]
    df["Target_Agri_GHG_ktCO2e"] = df[component_cols].sum(axis=1, min_count=1)

    # Data-quality flags. Livestock: real values exist only in census years (2012, 2019); between them linear
    # interpolation, outside them held constant. Source CSVs list which states each census covered.
    cdir = os.path.join(BASE_DIR, "data", "processed", "state_panel")
    anchor_years = {}   # state -> sorted list of census years for which the state has a value (cattle used as the marker)
    for ayear, fname in ((2007, "livestock_census_2007.csv"), (2012, "livestock_census_2012.csv"), (2019, "livestock_census_2019.csv")):
        path = os.path.join(cdir, fname)
        if os.path.exists(path):
            for st_name in pd.read_csv(path)["State"]:
                anchor_years.setdefault(st_name, []).append(ayear)
    # Andhra Pradesh / Telangana: the 2007 and 2012 censuses are undivided, so for the years the RBI data splits the two
    # states (from Telangana's first row) only 2019 is observed; Telangana before 2019 is derived (flag: interpolated).
    tel_first = int(df.loc[df["State"] == "Telangana", "Year_Standardized"].min())
    if "Andhra Pradesh" in anchor_years:
        anchor_years["Andhra Pradesh"] = [y for y in anchor_years["Andhra Pradesh"] if y < tel_first or y >= 2019]
    if "Telangana" in anchor_years:
        anchor_years["Telangana"] = [2019]
    first = df["State"].map(lambda x: min(anchor_years[x]) if x in anchor_years else np.nan)
    first = first.where(df["State"] != "Telangana", tel_first - 1)       # Telangana is derived (interpolated) from tel_first
    last = df["State"].map(lambda x: max(anchor_years[x]) if x in anchor_years else np.nan)
    is_anchor = pd.Series([yr_ in anchor_years.get(st_, []) for yr_, st_ in zip(df["Year_Standardized"], df["State"])], index=df.index)
    yr = df["Year_Standardized"]
    df["Flag_Livestock"] = np.select([is_anchor, (yr > first) & (yr < last), first.notna()],
                                     ["observed", "interpolated", "held_constant"], "missing")
    comp_ok = {"GHG_Rice_CH4_ktCO2e": df["Flag_Rice"] == "observed",
               "GHG_ResidueBurning_CH4_ktCO2e": df["Flag_Residue"] != "missing",
               "GHG_Livestock_CH4_ktCO2e": df["Flag_Livestock"] != "missing",
               "GHG_FertilizerSoil_N2O_ktCO2e": df["Flag_Fertiliser"] != "missing"}
    df["Data_Completeness"] = np.where(np.logical_and.reduce(list(comp_ok.values())), "complete", "partial")
    observed_kt = (df["GHG_Rice_CH4_ktCO2e"].where(df["Flag_Rice"] == "observed", 0).fillna(0)
                   + (df["GHG_ResidueBurning_CH4_ktCO2e"].fillna(0) + df["GHG_ResidueBurning_N2O_ktCO2e"].fillna(0)).where(df["Flag_Residue"] != "missing", 0)
                   + (df["GHG_Livestock_CH4_ktCO2e"].fillna(0) + df["GHG_Livestock_N2O_ktCO2e"].fillna(0)).where(df["Flag_Livestock"] == "observed", 0)
                   + df["GHG_FertilizerSoil_N2O_ktCO2e"].where(df["Flag_Fertiliser"] == "observed", 0).fillna(0))
    df["Target_Observed_Share"] = (observed_kt / df["Target_Agri_GHG_ktCO2e"]).where(df["Target_Agri_GHG_ktCO2e"] > 0)

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print("\nFlag counts:", {c: df[c].value_counts().to_dict() for c in ["Flag_Rice", "Flag_Residue", "Flag_Livestock", "Flag_Fertiliser", "Data_Completeness"]})
    print(f"\nTarget populated for {df['Target_Agri_GHG_ktCO2e'].notna().sum()} of {len(df)} state-year rows")
    print("\nComponent columns and their non-null coverage:")
    for c in component_cols:
        print(f"  {c}: {df[c].notna().mean():.1%}")

    print("\nSample (Punjab, all years):")
    sample = df[df["State"] == "Punjab"][["Year_Standardized", "Target_Agri_GHG_ktCO2e"] + component_cols]
    print(sample.sort_values("Year_Standardized").to_string(index=False))


if __name__ == "__main__":
    main()
