"""
Shared loader for comparisons with the GHG Platform India (GHGPI) benchmark.

Andhra Pradesh and Telangana are compared as ONE unit in every year. GHGPI keeps Andhra Pradesh undivided through 2013 and
splits it from 2014 (the legal bifurcation), whereas the RBI crop series used here splits the two states from 2010, so a
state-by-state comparison would be an artefact of different boundaries in 2010-2013 (it showed Andhra Pradesh at -40%).
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINE_COLS = ["GHG_Rice_CH4_ktCO2e", "GHG_Livestock_CH4_ktCO2e", "GHG_Livestock_N2O_ktCO2e", "GHG_FertilizerSoil_N2O_ktCO2e",
             "GHG_ResidueBurning_CH4_ktCO2e", "GHG_ResidueBurning_N2O_ktCO2e", "Target_Agri_GHG_ktCO2e"]
REF_COLS = ["rice_ktCO2e", "enteric_ktCO2e", "manure_ktCO2e", "soils_ktCO2e", "cropland_burning_ktCO2e"]
MERGED_NAME = "Andhra Pradesh (incl. Telangana)"


def load_merged(panel_path=None):
    panel_path = panel_path or os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv")
    p = pd.read_csv(panel_path)
    g = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/ghgpi_state_agri_2005_2018.csv"))
    complete = p.set_index(["State", "Year_Standardized"]).Data_Completeness if "Data_Completeness" in p.columns else None
    m = p[["State", "Year_Standardized", "Data_Completeness"] + MINE_COLS].merge(
        g[["State", "Year_Standardized"] + REF_COLS], on=["State", "Year_Standardized"], how="inner")
    ap = m.State.isin(["Andhra Pradesh", "Telangana"])
    merged = m[ap].groupby("Year_Standardized").agg(
        {**{c: (lambda s: s.sum(min_count=1)) for c in MINE_COLS + REF_COLS},
         "Data_Completeness": lambda s: "complete" if (s == "complete").all() else "partial"}).reset_index()
    merged["State"] = MERGED_NAME
    out = pd.concat([m[~ap], merged[m.columns]], ignore_index=True).sort_values(["State", "Year_Standardized"]).reset_index(drop=True)
    out["ref_total"] = out[REF_COLS].sum(axis=1, min_count=1)
    out["ref_livestock"] = out.enteric_ktCO2e + out.manure_ktCO2e
    out["mine_livestock"] = out.GHG_Livestock_CH4_ktCO2e + out.GHG_Livestock_N2O_ktCO2e
    out["mine_residue"] = out.GHG_ResidueBurning_CH4_ktCO2e + out.GHG_ResidueBurning_N2O_ktCO2e
    return out
