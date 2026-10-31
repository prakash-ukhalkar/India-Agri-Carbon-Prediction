"""
Does the "size explains spatial skill" result depend on WHICH size variable is used?
For each alternative scale proxy, fit the scale-only regression (log target ~ log proxy) in leave-one-state-out fashion and
compare with the ML models' leave-one-state-out predictions (from validation_suite) on exactly the same rows.
Proxies: Gross_Sown_Area (primary), Net_Sown_Area, Gross_Irrigated_Area, GSVA_Agriculture_Constant (smaller sample).
Output: results/state_panel/scale_sensitivity.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results/state_panel")
panel = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv"))
vp = pd.read_csv(os.path.join(RES, "validation_predictions.csv"))
vp = vp[vp.scheme == "LOSO"]
PROXIES = ["Gross_Sown_Area", "Net_Sown_Area", "Gross_Irrigated_Area", "GSVA_Agriculture_Constant"]

rows = []
for target, label in [("Target_Agri_GHG_ktCO2e", "level"), (None, "dynamic_only")]:
    d = vp[vp.target == label].merge(panel, on=["State", "Year_Standardized"], how="left")
    for prox in PROXIES:
        s = d.dropna(subset=[prox]).copy()
        s = s[(s[prox] > 0) & (s.actual > 0)]
        pred = np.zeros(len(s))
        for st in s.State.unique():
            te = (s.State == st).values
            lr = LinearRegression().fit(np.log(s.loc[~te, [prox]]), np.log(s.loc[~te, "actual"]))
            pred[te] = np.exp(lr.predict(np.log(s.loc[te, [prox]])))
        y = s.actual.values
        r = {"target": label, "scale_proxy": prox, "n": len(s), "states": s.State.nunique(),
             "scale_only_R2": r2_score(y, pred), "scale_only_R2_log": r2_score(np.log(y), np.log(pred))}
        for m in ["RF", "XGB"]:
            r[f"{m}_R2"] = r2_score(y, s[m])
            r[f"{m}_R2_log"] = r2_score(np.log(y), np.log(np.clip(s[m], y.min(), None)))
        rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "scale_sensitivity.csv"), index=False)
print(out.round(3).to_string(index=False))
