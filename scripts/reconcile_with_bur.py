"""
Compare the national sum of the state panel with India's official agriculture inventory, BUR-4 (UNFCCC, Dec 2024),
Table 2.36, for 2011-2020 (data/processed/state_panel/bur4_agriculture_2011_2020.csv, transcribed from the report and
checked to sum to the stated totals). Both use SAR GWPs.
Component mapping: rice <-> Rice cultivation; livestock <-> Enteric + Manure; fertiliser soil N2O <-> Agricultural soils
(BUR soils also include organic, residue and indirect sources, so a ratio below 1 is expected); residue <-> field burning.
NOTE: BUR-4 manure management falls from 27.5 Mt (2019) to 13.1 Mt (2020), so 2020 is an atypical comparison year.
Outputs: results/state_panel/bur_reconciliation.csv (2020, kept for compatibility), bur_reconciliation_2011_2020.csv
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv"))
bur = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/bur4_agriculture_2011_2020.csv"))
bur = bur.pivot(index="Year", columns="Category", values="Gg_CO2e") / 1000.0  # Mt

mine = df.groupby("Year_Standardized").agg(
    rice=("GHG_Rice_CH4_ktCO2e", "sum"), fert=("GHG_FertilizerSoil_N2O_ktCO2e", "sum"),
    lv_ch4=("GHG_Livestock_CH4_ktCO2e", "sum"), lv_n2o=("GHG_Livestock_N2O_ktCO2e", "sum"),
    res_ch4=("GHG_ResidueBurning_CH4_ktCO2e", "sum"), res_n2o=("GHG_ResidueBurning_N2O_ktCO2e", "sum"),
    total=("Target_Agri_GHG_ktCO2e", "sum")) / 1000.0
mine["livestock"] = mine.lv_ch4 + mine.lv_n2o
mine["residue"] = mine.res_ch4 + mine.res_n2o

rows = []
for y in range(2011, 2021):
    ref = {"Rice cultivation": bur.loc[y, "Rice_cultivation"],
           "Livestock (enteric + manure)": bur.loc[y, "Enteric_fermentation"] + bur.loc[y, "Manure_management"],
           "Agricultural soils (panel: fertiliser N2O only)": bur.loc[y, "Agricultural_soils"],
           "Residue burning": bur.loc[y, "Field_burning_of_residues"], "TOTAL": bur.loc[y, "Agriculture_total"]}
    val = {"Rice cultivation": mine.loc[y, "rice"], "Livestock (enteric + manure)": mine.loc[y, "livestock"],
           "Agricultural soils (panel: fertiliser N2O only)": mine.loc[y, "fert"], "Residue burning": mine.loc[y, "residue"],
           "TOTAL": mine.loc[y, "total"]}
    for k in ref:
        rows.append({"Year": y, "Component": k, "Panel_Mt": val[k], "BUR4_Mt": ref[k], "Ratio": val[k] / ref[k]})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, "results/state_panel/bur_reconciliation_2011_2020.csv"), index=False)
out[out.Year == 2020].drop(columns="Year").rename(columns={"Panel_Mt": "Panel_2020_Mt", "BUR4_Mt": "BUR4_2020_Mt"}).to_csv(
    os.path.join(BASE, "results/state_panel/bur_reconciliation.csv"), index=False)
piv = out.pivot(index="Year", columns="Component", values="Ratio").round(2)
print(piv.to_string())
t = out[out.Component == "TOTAL"].set_index("Year")
print("\nTOTAL panel vs BUR4 (Mt):"); print(t[["Panel_Mt", "BUR4_Mt", "Ratio"]].round(2).to_string())
