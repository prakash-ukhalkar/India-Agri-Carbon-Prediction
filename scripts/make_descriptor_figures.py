"""
Dataset figures, from data_release/ and results/state_panel/ only.
  descriptor_fig1_components.png   national emissions by component, with the data-quality periods marked
  descriptor_fig2_vs_ghgpi.png     dataset vs GHG Platform India, state-years (log-log) and national series
  descriptor_fig3_flags.png        state x year data-quality flags for livestock and fertiliser
  descriptor_fig4_uncertainty.png  national total with the Monte Carlo 90% interval, GHGPI series and BUR4 2020
Colours are colour-blind safe (Okabe-Ito).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results/state_panel")
OUT = os.path.join(BASE, "results/figures")
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(os.path.join(BASE, "data_release/agri_ghg_state_year.csv"))
g = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/ghgpi_state_agri_2005_2018.csv"))
g["ref"] = g.rice_ktCO2e + g.enteric_ktCO2e + g.manure_ktCO2e + g.soils_ktCO2e + g.cropland_burning_ktCO2e
mc = pd.read_csv(os.path.join(RES, "uncertainty_mc_national.csv"))
C = {"rice": "#009E73", "livestock": "#D55E00", "fert": "#0072B2", "resid": "#CC79A7", "grey": "#999999"}

# ---- Fig 1: national components ----
comp = pd.DataFrame({
    "Livestock (CH4 + N2O)": d.GHG_Livestock_CH4_ktCO2e.fillna(0) + d.GHG_Livestock_N2O_ktCO2e.fillna(0),
    "Fertiliser soil N2O": d.GHG_FertilizerSoil_N2O_ktCO2e.fillna(0),
    "Rice cultivation CH4": d.GHG_Rice_CH4_ktCO2e.fillna(0),
    "Residue burning": d.GHG_ResidueBurning_CH4_ktCO2e.fillna(0) + d.GHG_ResidueBurning_N2O_ktCO2e.fillna(0)})
comp["Year"] = d.Year_Standardized
nat = comp.groupby("Year").sum() / 1000
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.stackplot(nat.index, nat.T.values, labels=nat.columns, colors=[C["livestock"], C["fert"], C["rice"], C["resid"]])
for y in (2007, 2012, 2019):
    ax.axvline(y, color="k", lw=0.8, ls=":")
ax.text(2007, nat.sum(axis=1).max() * 1.02, "census 2007", ha="center", fontsize=7)
ax.text(2012, nat.sum(axis=1).max() * 1.02, "2012", ha="center", fontsize=7)
ax.text(2019, nat.sum(axis=1).max() * 1.02, "2019", ha="center", fontsize=7)
ax.axvspan(2019, 2024.4, color="k", alpha=0.07)
ax.text(2021.7, nat.sum(axis=1).max() * 0.45, "livestock held\nconstant", ha="center", fontsize=8)
ax.set_xlim(2004, 2024.4)
ax.set_xticks([2004, 2008, 2012, 2016, 2020, 2024])
ax.set_ylabel("Mt CO$_2$e (SAR GWPs)")
ax.set_title("India agricultural emissions by component (national sum of states)", fontsize=10)
ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.0))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "descriptor_fig1_components.png"), dpi=200)
plt.close(fig)

# ---- Fig 2: vs GHGPI (Andhra Pradesh + Telangana compared as one unit; see _compare_utils.py) ----
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _compare_utils import load_merged
m = load_merged()
m = m[m.Target_Agri_GHG_ktCO2e.notna() & (m.ref_total > 0)].rename(columns={"ref_total": "ref"})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
for lab, col in (("complete", C["fert"]), ("partial", C["livestock"])):
    s_ = m[m.Data_Completeness == lab]
    a.scatter(s_.ref, s_.Target_Agri_GHG_ktCO2e, s=10, alpha=0.7, color=col, label=f"{lab} (n={len(s_)})")
lim = [m.ref.min() * 0.8, m.ref.max() * 1.2]
a.plot(lim, lim, color="k", lw=0.8)
a.plot(lim, [x * 1.2 for x in lim], color="k", lw=0.5, ls="--")
a.plot(lim, [x * 0.8 for x in lim], color="k", lw=0.5, ls="--")
a.set_xscale("log"), a.set_yscale("log")
a.set_xlabel("GHG Platform India (kt CO$_2$e)"), a.set_ylabel("This dataset (kt CO$_2$e)")
a.set_title("Unit-years, 2005-2018 (dashed: +/-20%)", fontsize=10)
a.legend(frameon=False, fontsize=8)
n_years = m.Year_Standardized.nunique()
keep = m.groupby("State").Year_Standardized.nunique()
bal = m[m.State.isin(keep[keep == n_years].index)]
natm = bal.groupby("Year_Standardized")[["Target_Agri_GHG_ktCO2e", "ref"]].sum() / 1000
b.plot(natm.index, natm.ref, color=C["grey"], marker="o", ms=3, label="GHG Platform India")
b.plot(natm.index, natm.Target_Agri_GHG_ktCO2e, color=C["fert"], marker="s", ms=3, label="This dataset")
b.set_ylabel("Mt CO$_2$e"), b.set_title(f"National sum, {bal.State.nunique()} units present in every year", fontsize=10)
b.legend(frameon=False, fontsize=8)
for ax_ in (a, b):
    ax_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "descriptor_fig2_vs_ghgpi.png"), dpi=200)
plt.close(fig)

# ---- Fig 3: flag heatmaps ----
order = d.groupby("State").Target_Agri_GHG_ktCO2e.mean().sort_values(ascending=False).index
years = sorted(d.Year_Standardized.unique())
codes_l = {"observed": 0, "interpolated": 1, "held_constant": 2, "missing": 3}
codes_f = {"observed": 0, "national_downscaled": 1, "held_constant": 2, "missing": 3}
cmap = ListedColormap(["#0072B2", "#56B4E9", "#E69F00", "#DDDDDD", "#F0F0F0"])
cmap_f = ListedColormap(["#0072B2", "#009E73", "#E69F00", "#DDDDDD"])
fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
for ax_, col, codes, title, labels in (
        (axes[0], "Flag_Livestock", codes_l, "Livestock activity data", ["observed", "interpolated", "held constant", "missing"]),
        (axes[1], "Flag_Fertiliser", codes_f, "Fertiliser activity data", ["observed", "national downscaled", "held constant", "missing"])):
    grid = d.pivot(index="State", columns="Year_Standardized", values=col).reindex(order).reindex(columns=years)
    use = cmap_f if col == "Flag_Fertiliser" else cmap
    ax_.imshow(grid.apply(lambda c: c.map(codes)).values.astype(float), cmap=use, vmin=0, vmax=3, aspect="auto")
    ax_.set_xticks(range(0, len(years), 4)), ax_.set_xticklabels(years[::4], fontsize=7)
    ax_.set_title(title, fontsize=10)
    ax_.legend(handles=[Patch(color=use(i), label=l) for i, l in enumerate(labels)], fontsize=7, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=len(labels))
axes[0].set_yticks(range(len(order))), axes[0].set_yticklabels(order, fontsize=6)
fig.suptitle("Data-quality flags by state (largest emitters at the top)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "descriptor_fig3_flags.png"), dpi=200)
plt.close(fig)

# ---- Fig 4: uncertainty ----
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.fill_between(mc.Year, mc.p05_Mt, mc.p95_Mt, color=C["fert"], alpha=0.2, label="Monte Carlo 90% interval")
ax.plot(mc.Year, mc.central_Mt, color=C["fert"], label="This dataset (central)")
gn = g.groupby("Year_Standardized").ref.sum() / 1000
ax.plot(gn.index, gn.values, color=C["grey"], marker="o", ms=3, label="GHG Platform India")
ax.scatter([2020], [406], color=C["livestock"], zorder=5, label="BUR4 (2020)")
ax.set_ylabel("Mt CO$_2$e"), ax.set_xlim(2004, 2024.4)
ax.legend(frameon=False, fontsize=8)
ax.set_title("National total with uncertainty and reference series", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "descriptor_fig4_uncertainty.png"), dpi=200)
print("descriptor figures written to", OUT)
