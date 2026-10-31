"""
SHAP interpretation for the dynamic-only target (the one framing where ML beats
the scale-only baseline in raw R2). Interpretation-only: features are selected
on the full sample and the model refit on all rows, so no accuracy is claimed.
Features are exogenous only (see validation_suite.build). A scale-only
dependence check (SHAP of Gross_Sown_Area-correlated features) is left to the
reader via the CSV.
Output: results/state_panel/shap_dynamic_v2.csv, shap_summary_dynamic_v2.png
"""
import os
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validation_suite as vs

df, cols = vs.build()
target = "Target_Dynamic"
d = df.dropna(subset=[target]).reset_index(drop=True)
sel = vs.select(d, cols, target)
X = d[sel].fillna(d[sel].median())
model = vs.MODELS["XGB"]().fit(X.values, d[target].values)
# shap 0.49 cannot parse xgboost>=3.1 models (base_score format), so use XGBoost's own
# exact TreeSHAP; the last column is the bias term.
import xgboost as xgb
sv = model.get_booster().predict(xgb.DMatrix(X.values), pred_contribs=True)[:, :-1]
imp = pd.DataFrame({"Feature": [c.replace("F__", "") for c in sel],
                    "Mean_Abs_SHAP": np.abs(sv).mean(axis=0)}).sort_values("Mean_Abs_SHAP", ascending=False)
out = os.path.join(vs.BASE, "results/state_panel")
imp.to_csv(os.path.join(out, "shap_dynamic_v2.csv"), index=False)
print(imp.head(12).to_string(index=False))
Xn = X.copy()
Xn.columns = [c.replace("F__", "") for c in sel]
shap.summary_plot(sv, Xn, max_display=12, show=False)
plt.tight_layout()
plt.savefig(os.path.join(out, "shap_summary_dynamic_v2.png"), dpi=200)
