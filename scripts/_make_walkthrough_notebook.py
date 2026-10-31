"""
Generates notebooks/01_state_panel_walkthrough.ipynb (nbformat 4) without needing nbformat.
Re-run this script to regenerate the notebook; edit the cells here, not the .ipynb.
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "notebooks", "01_state_panel_walkthrough.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# State-level agricultural GHG emissions in India: walkthrough

This notebook reads the panel and the saved results and reproduces the tables and figures.
It does **not** retrain models by default (the full validation takes roughly 10-15 minutes); set `RUN_SUITE = True`
in the first code cell to regenerate every result. Method and caveats: see `README.md`.
"""),
    code("""
import os, sys, subprocess
import numpy as np
import pandas as pd
try:
    from IPython.display import display, Image
except ImportError:          # allows running this notebook's code as a plain script
    display, Image = print, None

RUN_SUITE = False            # True: rebuild target + rerun validation (slow)

ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
RES = os.path.join(ROOT, "results", "state_panel")
FIG = os.path.join(ROOT, "results", "figures")
PY = sys.executable
if RUN_SUITE:
    for s in ["build_tier1_ghg_target", "reconcile_with_bur", "validation_suite", "bootstrap_ci", "shap_dynamic", "make_figures"]:
        subprocess.run([PY, os.path.join(ROOT, "scripts", s + ".py")], check=True, cwd=ROOT)
pd.set_option("display.width", 200)
"""),
    md("""
## 1. The panel
37 states and union territories, 2004-2024, built from the RBI Handbook of Statistics on Indian States, the 2012 and 2019
Livestock Censuses, the CEA General Review and NASA POWER.
"""),
    code("""
panel = pd.read_csv(os.path.join(ROOT, "data/processed/state_panel/state_year_panel_with_target.csv"))
print(f"{panel.shape[0]} state-years, {panel.State.nunique()} states, "
      f"{panel.Year_Standardized.min()}-{panel.Year_Standardized.max()}, no duplicate keys: "
      f"{not panel.duplicated(['State', 'Year_Standardized']).any()}")
cover = panel[["Rice_Area", "Gross_Sown_Area", "Fertiliser_Consumption_NPK", "Annual_Rainfall_mm",
               "Temperature_C", "Stage_of_Extraction_pct"]].notna().mean().round(3)
display(cover.rename("share of state-years populated").to_frame())
"""),
    md("""
## 2. The target and its reconciliation with India's inventory
The target is a bottom-up IPCC Tier-1 estimate (rice CH4, residue burning, livestock, fertiliser soil N2O). Grid
electricity is **not** included (Energy sector in national inventories). The 2020 national sum is compared with BUR4.
"""),
    code("""
comps = [c for c in panel.columns if c.startswith("GHG_")]
share = (panel[comps].sum() / panel[comps].sum().sum()).round(3).rename("share of target, all years")
display(share.to_frame())
display(pd.read_csv(os.path.join(RES, "bur_reconciliation.csv")).round(2))
"""),
    md("""
## 3. Spatial generalisation (leave-one-state-out)
Feature selection is fitted inside each training fold; the target's own lags are never features. **Scale only** is a
regression of log emissions on log gross sown area.
"""),
    code("""
v = pd.read_csv(os.path.join(RES, "validation_suite.csv"))
spatial = v[v.scheme == "LOSO"].pivot_table(index="target", columns="model", values=["R2", "R2_log"])
display(spatial.round(3))
"""),
    code("""
b = pd.read_csv(os.path.join(RES, "bootstrap_r2_differences.csv"))
display(b[b.scheme == "LOSO"].round(3))
if Image is not None:
    display(Image(os.path.join(FIG, "fig1_spatial_vs_baselines.png")))
"""),
    md("""
## 4. Where the models fail
Mean absolute percentage error of XGBoost for each held-out state against state size.
"""),
    code("""
ps = pd.read_csv(os.path.join(RES, "per_state_error_level_xgb.csv")).sort_values("mean_actual", ascending=False)
display(ps.round(0).head(5)); display(ps.round(0).tail(5))
if Image is not None:
    display(Image(os.path.join(FIG, "fig3_per_state_error.png")))
"""),
    md("""
## 5. Temporal generalisation (expanding window)
"""),
    code("""
temporal = v[v.scheme == "Expanding"].pivot_table(index="target", columns="model", values="R2")
display(temporal.round(3))
display(b[b.scheme == "Expanding"].round(3))
if Image is not None:
    display(Image(os.path.join(FIG, "fig2_temporal_vs_baselines.png")))
"""),
    md("""
## 6. What the dynamic-only model uses (SHAP, interpretation only)
"""),
    code("""
display(pd.read_csv(os.path.join(RES, "shap_dynamic_v2.csv")).head(8).round(1))
"""),
    md("""
## Reading the results
* For the emission level of an unseen state, a one-variable size regression matches the machine-learning models
  (XGBoost minus scale-only R2 = -0.04, 95% CI -0.30 to 0.52); in log terms it is clearly better.
* On the dynamic-only target the models lead in raw R2 but the interval is far too wide to claim an advantage.
* No model beats persistence over time, and year-over-year change is not predictable.
* Errors for the smallest states are large (41% to over 1,200%).
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT)
