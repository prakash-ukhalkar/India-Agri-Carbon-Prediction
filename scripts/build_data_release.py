"""
Assemble the public data release in data_release/ from the processed panel, Monte Carlo output and validation tables:
  agri_ghg_state_year.csv      component emissions, total, 90% interval, activity data and per-component quality flags
  emission_factors.csv         every emission factor with provenance
  data_dictionary.csv          column definitions and units
  validation_summary.csv       agreement with GHG Platform India and BUR4
  (README, citation file, licence notice, checksums and Zenodo metadata are written by prepare_zenodo_deposit.py)
The licence file is NOT written: the author must choose it after checking the terms of the sources (RBI Handbook,
Livestock Censuses, CEA, NASA POWER, GHG Platform India usage policy).
Run after: build_tier1_ghg_target, uncertainty_mc, compare_with_ghgpi, reconcile_with_bur.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(BASE, "data/processed/state_panel")
RES = os.path.join(BASE, "results/state_panel")
OUT = os.path.join(BASE, "data_release")
VERSION = "0.1.0-draft"
os.makedirs(OUT, exist_ok=True)

p = pd.read_csv(os.path.join(SP, "state_year_panel_with_target.csv"))
mc = pd.read_csv(os.path.join(RES, "uncertainty_mc_state_year.csv"))
df = p.merge(mc[["State", "Year_Standardized", "p05", "p50", "p95"]], on=["State", "Year_Standardized"], how="left")

cols = {
    "State": ("State or union territory (canonical name)", "-"),
    "Year_Standardized": ("Start calendar year of the fiscal year (e.g. 2013 = FY2013-14)", "year"),
    "GHG_Rice_CH4_ktCO2e": ("Rice cultivation CH4", "kt CO2e (SAR GWP)"),
    "GHG_Livestock_CH4_ktCO2e": ("Livestock enteric fermentation + manure management CH4", "kt CO2e (SAR GWP)"),
    "GHG_Livestock_N2O_ktCO2e": ("Livestock manure management N2O (poultry not included)", "kt CO2e (SAR GWP)"),
    "GHG_FertilizerSoil_N2O_ktCO2e": ("Synthetic-fertiliser soil N2O, direct + indirect", "kt CO2e (SAR GWP)"),
    "GHG_ResidueBurning_CH4_ktCO2e": ("Crop-residue burning CH4 (rice, wheat, coarse cereals)", "kt CO2e (SAR GWP)"),
    "GHG_ResidueBurning_N2O_ktCO2e": ("Crop-residue burning N2O (rice, wheat, coarse cereals)", "kt CO2e (SAR GWP)"),
    "Target_Agri_GHG_ktCO2e": ("Sum of the six components above (missing components are not counted; see Data_Completeness)", "kt CO2e (SAR GWP)"),
    "Total_p05_ktCO2e": ("5th percentile of Monte Carlo total (emission-factor uncertainty only)", "kt CO2e"),
    "Total_p50_ktCO2e": ("Median of Monte Carlo total", "kt CO2e"),
    "Total_p95_ktCO2e": ("95th percentile of Monte Carlo total", "kt CO2e"),
    "Flag_Rice": ("observed | missing (state never reports rice area)", "-"),
    "Flag_Residue": ("observed | structural_zero_assumed (a crop never reported by the state is taken as zero) | missing", "-"),
    "Flag_Livestock": ("observed (census years 2007, 2012, 2019; 2007 is a secondary compilation and Telangana 2012 is derived) | interpolated (between census years) | held_constant (before 2007 or after 2019) | missing", "-"),
    "Flag_Fertiliser": ("observed (RBI state N+P+K 2013-2021 x the official All-India N/NPK ratio of the year) | national_downscaled (other years to 2023: official All-India N distributed with the state's N+P+K pattern of the nearest observed year, 2013 backwards and 2021 forwards) | held_constant (2024: 2023 national level) | missing", "-"),
    "Data_Completeness": ("complete if no component is missing, else partial (total is then a lower bound)", "-"),
    "Target_Observed_Share": ("Share of the total from components that are observed (not interpolated/held constant)", "fraction"),
    "Rice_Area": ("Rice area", "thousand ha"),
    "Fertiliser_Consumption_NPK": ("N+P+K fertiliser consumption as reported (empty outside 2013-2021)", "thousand t"),
    "Fertiliser_N_applied_ktN": ("Nitrogen applied used for N2O (see Flag_Fertiliser)", "kt N"),
    "N_share_of_NPK": ("National N/(N+P2O5+K2O) ratio of the year from the official All-India table", "fraction"),
    "N_scale_to_national": ("Reserved (1 for all rows in the released version)", "ratio"),
    "Livestock_Cattle_Head": ("Cattle headcount (census years real, otherwise see Flag_Livestock)", "head"),
    "Livestock_Buffalo_Head": ("Buffalo headcount", "head"),
    "Livestock_Sheep_Head": ("Sheep headcount", "head"),
    "Livestock_Goat_Head": ("Goat headcount", "head"),
    "Livestock_Pigs_Head": ("Pig headcount", "head"),
}
df = df.rename(columns={"p05": "Total_p05_ktCO2e", "p50": "Total_p50_ktCO2e", "p95": "Total_p95_ktCO2e"})
release = df[list(cols)].sort_values(["State", "Year_Standardized"])
release.to_csv(os.path.join(OUT, "agri_ghg_state_year.csv"), index=False, float_format="%.6g")
pd.DataFrame([{"column": c, "description": d, "unit": u} for c, (d, u) in cols.items()]).to_csv(
    os.path.join(OUT, "data_dictionary.csv"), index=False)
(pd.read_csv(os.path.join(SP, "india_specific_emission_factors.csv"))
 .merge(pd.read_csv(os.path.join(SP, "factor_verification.csv")), on="parameter", how="left")
 .to_csv(os.path.join(OUT, "emission_factors.csv"), index=False))

g = pd.read_csv(os.path.join(RES, "ghgpi_comparison_v4.csv")).assign(benchmark="GHG Platform India 2005-2018, state-year")
b = pd.read_csv(os.path.join(RES, "bur_reconciliation.csv")).assign(benchmark="BUR4 2020, national")
pd.concat([g, b], ignore_index=True).to_csv(os.path.join(OUT, "validation_summary.csv"), index=False)

n = len(release)
flags = {c: release[c].value_counts().to_dict() for c in ["Flag_Livestock", "Flag_Fertiliser", "Flag_Rice", "Flag_Residue", "Data_Completeness"]}
print(f"data_release written: {n} rows, {release.shape[1]} columns")
print("flags:", flags)
