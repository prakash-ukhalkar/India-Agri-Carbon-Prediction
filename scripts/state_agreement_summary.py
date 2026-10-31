"""
State-level agreement with GHGPI from one script: completeness split, share of state-years within
10% / 20%, most under- and over-estimated units, and within-state temporal correlations (levels and annual changes).
Andhra Pradesh and Telangana are one unit (see _compare_utils.py). Output: results/state_panel/state_agreement_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _compare_utils import load_merged, BASE

m = load_merged()
m = m[m.Target_Agri_GHG_ktCO2e.notna() & (m.ref_total > 0)].copy()
m["err"] = (m.Target_Agri_GHG_ktCO2e - m.ref_total) / m.ref_total
rows = [{"item": "state_years_compared", "value": len(m)}, {"item": "units", "value": m.State.nunique()},
        {"item": "median_abs_error_all", "value": m.err.abs().median()},
        {"item": "share_within_10pct", "value": (m.err.abs() < 0.10).mean()},
        {"item": "share_within_20pct", "value": (m.err.abs() < 0.20).mean()}]
for lab, d in m.groupby("Data_Completeness"):
    rows += [{"item": f"n_{lab}", "value": len(d)}, {"item": f"median_abs_error_{lab}", "value": d.err.abs().median()},
             {"item": f"mean_error_{lab}", "value": d.err.mean()}]
s = m.groupby("State").err.mean().sort_values()
for st, v in list(s.head(5).items()) + list(s.tail(5).items()):
    rows.append({"item": f"mean_error_{st}", "value": v})


def within(a, b, diff=False):
    out = []
    for _, d in m.sort_values("Year_Standardized").groupby("State"):
        d = d.dropna(subset=[a, b])
        if len(d) < 8:
            continue
        x, y = d[a].values, d[b].values
        if diff:
            x, y = np.diff(x), np.diff(y)
        if x.std() > 0 and y.std() > 0:
            out.append(np.corrcoef(x, y)[0, 1])
    return np.array(out)


for nm, (a, b) in {"total": ("Target_Agri_GHG_ktCO2e", "ref_total"), "livestock": ("mine_livestock", "ref_livestock"),
                   "rice": ("GHG_Rice_CH4_ktCO2e", "rice_ktCO2e"), "fertiliser": ("GHG_FertilizerSoil_N2O_ktCO2e", "soils_ktCO2e")}.items():
    lv, ch = within(a, b), within(a, b, True)
    rows += [{"item": f"within_state_corr_levels_{nm}", "value": np.median(lv)}, {"item": f"within_state_corr_changes_{nm}", "value": np.median(ch)},
             {"item": f"share_states_changes_corr_gt_0.5_{nm}", "value": float(np.mean(ch > 0.5))}, {"item": f"n_units_{nm}", "value": len(ch)}]
nat = m.groupby("Year_Standardized")[["Target_Agri_GHG_ktCO2e", "ref_total"]].sum()
rows.append({"item": "national_change_2005_2018_mine_pct", "value": (nat.iloc[-1, 0] / nat.iloc[0, 0] - 1) * 100})
rows.append({"item": "national_change_2005_2018_ghgpi_pct", "value": (nat.iloc[-1, 1] / nat.iloc[0, 1] - 1) * 100})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, "results/state_panel/state_agreement_summary.csv"), index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
