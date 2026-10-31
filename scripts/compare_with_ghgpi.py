"""
State-by-state comparison of the bottom-up panel target with the GHG Platform India (GHGPI) estimates, 2005-2018.
Both use IPCC SAR/AR2 GWPs (CH4 21, N2O 310). Component mapping:
  rice               <-> Rice Cultivation
  livestock          <-> Enteric Fermentation + Manure Management
  fertiliser soil N2O <-> Agriculture Soils   (GHGPI soils include organic, residue and indirect sources, so a
                         ratio below 1 is expected for a fertiliser-only estimate; far above 1 means overstatement)
  residue burning    <-> Biomass Burning in Cropland
  total              <-> sum of the above
Metrics per component: national ratio (mean over years), pooled raw R2 and log R2 of panel vs GHGPI over state-years,
Spearman rank correlation across state-years, median absolute percentage error.
Output: results/state_panel/ghgpi_comparison.csv  (set TAG env var to label a target version)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = os.environ.get("TAG", "current")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _compare_utils import load_merged   # Andhra Pradesh + Telangana compared as one unit (see that file)

m = load_merged()
mine = pd.DataFrame({"State": m.State, "Year": m.Year_Standardized, "rice": m.GHG_Rice_CH4_ktCO2e, "livestock": m.mine_livestock,
                     "soils": m.GHG_FertilizerSoil_N2O_ktCO2e, "residue": m.mine_residue, "total": m.Target_Agri_GHG_ktCO2e})
ref = pd.DataFrame({"rice": m.rice_ktCO2e, "livestock": m.ref_livestock, "soils": m.soils_ktCO2e,
                    "residue": m.cropland_burning_ktCO2e, "total": m.ref_total})

rows = []
for c in ["rice", "livestock", "soils", "residue", "total"]:
    ok = mine[c].notna() & ref[c].notna() & (ref[c] > 0) & (mine[c] > 0)
    a, b = mine.loc[ok, c].values, ref.loc[ok, c].values
    yrs = mine.loc[ok, "Year"]
    nat_ratio = (mine.loc[ok].groupby("Year")[c].sum() / ref.loc[ok].groupby(yrs)[c].sum()).mean()
    rows.append({"target_version": TAG, "component": c, "n_state_years": int(ok.sum()),
                 "national_ratio_panel_over_GHGPI": nat_ratio, "R2_raw": r2_score(b, a),
                 "R2_log": r2_score(np.log(b), np.log(a)), "spearman": spearmanr(a, b)[0],
                 "median_APE_pct": float(np.median(np.abs(a - b) / b) * 100)})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, f"results/state_panel/ghgpi_comparison_{TAG}.csv"), index=False)
print(out.round(3).to_string(index=False))
