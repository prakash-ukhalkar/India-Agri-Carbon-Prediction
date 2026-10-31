"""
State-cluster bootstrap (2000 draws, resampling held-out states) for the difference in pooled R2
between each ML model and the matched baseline, from validation_predictions.csv.
Output: results/state_panel/bootstrap_r2_differences.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results/state_panel")
vp = pd.read_csv(os.path.join(RES, "validation_predictions.csv"))
rng = np.random.default_rng(42)
PAIRS = {"LOSO": ["base_scale_only"], "Expanding": ["base_own_history_mean", "base_own_lag1"]}
rows = []
for (target, scheme), d in vp.groupby(["target", "scheme"]):
    if scheme == "LOSO" and target == "yoy_change":
        bases = ["base_other_states_mean"]
    elif scheme == "Expanding" and target == "yoy_change":
        bases = ["base_zero_change", "base_own_mean_change"]
    else:
        bases = PAIRS[scheme]
    states = d.State.unique()
    groups = {s: g for s, g in d.groupby("State")}
    for ml in ["RF", "XGB"]:
        for b in bases:
            def diff(sample):
                g = pd.concat([groups[s] for s in sample])
                return r2_score(g.actual, g[ml]) - r2_score(g.actual, g[b])
            point = diff(states)
            boots = [diff(rng.choice(states, len(states))) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"target": target, "scheme": scheme, "model": ml, "baseline": b,
                         "R2_diff_ML_minus_baseline": point, "CI_low": lo, "CI_high": hi})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "bootstrap_r2_differences.csv"), index=False)
print(out.round(3).to_string(index=False))
