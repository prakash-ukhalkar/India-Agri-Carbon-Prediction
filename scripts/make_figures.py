"""
Figures generated from results/state_panel/*.csv only.
  fig1_spatial_vs_baselines.png  LOSO R2: ML models vs scale-only and other-states mean
  fig2_temporal_vs_baselines.png expanding-window R2: ML models vs own-history / own-lag1 baselines
  fig3_per_state_error.png       per-state XGBoost LOSO error vs state size
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results/state_panel")
OUT = os.path.join(BASE, "results/figures")
os.makedirs(OUT, exist_ok=True)
v = pd.read_csv(os.path.join(RES, "validation_suite.csv"))
COL = {"RF": "#0072B2", "XGB": "#D55E00", "base_scale_only": "#009E73", "base_other_states_mean": "#999999",
       "base_own_history_mean": "#999999", "base_own_lag1": "#009E73", "base_zero_change": "#999999",
       "base_own_mean_change": "#009E73"}
NAME = {"RF": "Random forest", "XGB": "XGBoost", "base_scale_only": "Scale only (gross sown area)",
        "base_other_states_mean": "Other-states mean", "base_own_history_mean": "Own historical mean",
        "base_own_lag1": "Own lag-1", "base_zero_change": "Zero change", "base_own_mean_change": "Own mean change"}


def grouped(scheme, targets, models, fname, title):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [v[(v.target == t) & (v.scheme == scheme) & (v.model == m)].R2.mean() for t in targets]
        ax.bar(np.arange(len(targets)) + i * w, vals, w, color=COL[m], label=NAME[m])
    ax.set_xticks(np.arange(len(targets)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([t.replace("_", " ") for t in targets])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Cross-validated R$^2$ (pooled)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(models))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)


grouped("LOSO", ["level", "dynamic_only", "intensity", "yoy_change"],
        ["RF", "XGB", "base_scale_only", "base_other_states_mean"], "fig1_spatial_vs_baselines.png",
        "Spatial generalisation (leave-one-state-out)")
t = v[(v.scheme == "Expanding") & v.model.isin(["RF", "XGB", "base_own_history_mean", "base_own_lag1"])]
grouped("Expanding", ["level", "dynamic_only", "intensity"],
        ["RF", "XGB", "base_own_history_mean", "base_own_lag1"], "fig2_temporal_vs_baselines.png",
        "Temporal generalisation (expanding window)")

vp = pd.read_csv(os.path.join(RES, "validation_predictions.csv"))
vp = vp[(vp.target == "level") & (vp.scheme == "LOSO")]
vp["ape"] = (vp.XGB - vp.actual).abs() / vp.actual * 100
ps = vp.groupby("State").agg(mean_actual=("actual", "mean"), MAPE_pct=("ape", "mean")).reset_index()
ps.to_csv(os.path.join(RES, "per_state_error_level_xgb.csv"), index=False)
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.scatter(ps.mean_actual, ps.MAPE_pct, color=COL["XGB"], s=22)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("State mean emissions (kt CO$_2$e)")
ax.set_ylabel("Mean absolute % error, XGBoost LOSO")
ax.set_title("Prediction error rises sharply for small states", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_per_state_error.png"), dpi=200)
print("figures written to", OUT)
