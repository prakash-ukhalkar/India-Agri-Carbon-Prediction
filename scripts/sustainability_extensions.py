"""
Two extensions of scripts/sustainability_analysis.py, same sample and definitions (2011-2023, Andhra Pradesh + Telangana combined):

1. Component-level decoupling: growth of rice, livestock, fertiliser and residue-burning emissions against agricultural GSVA,
   nationally (balanced units) and by unit (Tapio classes).
2. Window sensitivity: the state classification (total emissions) re-estimated for 2011-2019, 2013-2023 and 2014-2023.

Outputs: results/state_panel/sustainability_components_national.csv, sustainability_component_classes.csv,
sustainability_window_units.csv, sustainability_window_summary.csv, and results/figures/sust_fig4_components.png
Run after scripts/sustainability_analysis.py.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results/state_panel")
FIG = os.path.join(BASE, "results/figures")
Y0, Y1 = 2011, 2023

p = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv")).rename(columns={"Year_Standardized": "Year"})
comp = {"Rice": ["GHG_Rice_CH4_ktCO2e"],
        "Livestock": ["GHG_Livestock_CH4_ktCO2e", "GHG_Livestock_N2O_ktCO2e"],
        "Fertiliser": ["GHG_FertilizerSoil_N2O_ktCO2e"],
        "Residue burning": ["GHG_ResidueBurning_CH4_ktCO2e", "GHG_ResidueBurning_N2O_ktCO2e"]}
for k, c in comp.items():
    p[k] = p[c].sum(axis=1, min_count=1)
p["E"] = p["Target_Agri_GHG_ktCO2e"]
p["Eobs"] = p[["Rice", "Fertiliser", "Residue burning"]].sum(axis=1, min_count=3)
p["V"] = p["GSVA_Agriculture_Constant"]
ap = p.State.isin(["Andhra Pradesh", "Telangana"])
cols = ["E", "Eobs", "V"] + list(comp)
comb = p[ap].groupby("Year")[cols].sum(min_count=2).reset_index()
comb["State"] = "Andhra Pradesh + Telangana"
q = pd.concat([p[~ap][["State", "Year"] + cols], comb[["State", "Year"] + cols]], ignore_index=True)

summ = pd.read_csv(os.path.join(OUT, "sustainability_state_summary.csv"))
units = summ.State.tolist()          # the 32 analysed units
q = q[q.State.isin(units)]


def trend(y, x):
    ok = np.isfinite(y) & (y > 0) & np.isfinite(x)
    if ok.sum() < 8:
        return np.nan, np.nan
    r = stats.linregress(x[ok], np.log(y[ok]))
    return r.slope, r.stderr


def tapio(g_e, g_v, e):
    if g_v > 0:
        if g_e < 0:
            return "strong decoupling"
        if e < 0.8:
            return "weak decoupling"
        return "expansive coupling" if e <= 1.2 else "expansive negative decoupling"
    if g_e > 0:
        return "strong negative decoupling"
    return "weak negative decoupling" if e < 0.8 else "recessive"


def classify(d, ycol, y0, y1):
    d = d[(d.Year >= y0) & (d.Year <= y1)].sort_values("Year")
    x = d.Year.values.astype(float)
    gE, _ = trend(d[ycol].values, x)
    gV, _ = trend(d.V.values, x)
    if not np.isfinite(gE) or not np.isfinite(gV):
        return np.nan, np.nan, np.nan, "n/a"
    if abs(gV) < 0.005:
        return gE, gV, np.nan, "no clear GSVA trend"
    e = gE / gV
    return gE, gV, e, tapio(gE, gV, e)


# ---- 1. component-level decoupling
nat_rows, cls_rows = [], []
for c in ["E"] + list(comp):
    label = "All components" if c == "E" else c
    pv = q.pivot(index="Year", columns="State", values=c)
    pv = pv.loc[Y0:Y1].dropna(axis=1)
    tot = pv.sum(axis=1)
    pvv = q.pivot(index="Year", columns="State", values="V").loc[Y0:Y1].dropna(axis=1).sum(axis=1)
    x = np.arange(Y0, Y1 + 1, dtype=float)
    gC, _ = trend(tot.values, x)
    gV, _ = trend(pvv.values, x)
    nat_rows.append(dict(component=label, units=pv.shape[1], change_2011_2023_pct=100 * (tot.loc[Y1] / tot.loc[Y0] - 1),
                         g_pct_yr=100 * gC, g_GSVA_pct_yr=100 * gV, elasticity=gC / gV,
                         share_2021_23_pct=np.nan if c == "E" else 100 * tot.loc[Y1 - 2:Y1].sum() / q.pivot(index="Year", columns="State", values="E").loc[Y1 - 2:Y1].sum().sum()))
    for s, d in q.groupby("State"):
        gE, gV_, e, cl = classify(d, c, Y0, Y1)
        cls_rows.append(dict(State=s, component=label, g_pct_yr=100 * gE if np.isfinite(gE) else np.nan, elasticity=e, **{"class": cl}))
N = pd.DataFrame(nat_rows)
C = pd.DataFrame(cls_rows)
N.round(3).to_csv(os.path.join(OUT, "sustainability_components_national.csv"), index=False)
counts = C[~C["class"].isin(["n/a"])].pivot_table(index="class", columns="component", values="State", aggfunc="count", fill_value=0)
counts.to_csv(os.path.join(OUT, "sustainability_component_classes.csv"))

# ---- 2. window sensitivity (total emissions)
windows = [(2011, 2023), (2011, 2019), (2013, 2023), (2014, 2023)]
wr = []
for s, d in q.groupby("State"):
    row = {"State": s}
    for a, b in windows:
        gE, gV, e, cl = classify(d, "E", a, b)
        row[f"class_{a}_{b}"] = cl
        row[f"g_E_{a}_{b}"] = 100 * gE if np.isfinite(gE) else np.nan
        row[f"g_V_{a}_{b}"] = 100 * gV if np.isfinite(gV) else np.nan
    wr.append(row)
W = pd.DataFrame(wr).merge(summ[["State", "share_of_national_pct"]], on="State").sort_values("share_of_national_pct", ascending=False)
W.round(3).to_csv(os.path.join(OUT, "sustainability_window_units.csv"), index=False)
base = "class_2011_2023"
srows = []
for a, b in windows[1:]:
    k = f"class_{a}_{b}"
    both = (W[base] != "no clear GSVA trend") & (W[k] != "no clear GSVA trend")
    same = (W[base] == W[k]) & both
    # coarse direction: is emission growth below GSVA growth (relative decoupling) in both windows?
    rel_base = W["g_E_2011_2023"] < W["g_V_2011_2023"]
    rel_k = W[f"g_E_{a}_{b}"] < W[f"g_V_{a}_{b}"]
    pos = (W["g_V_2011_2023"] > 0) & (W[f"g_V_{a}_{b}"] > 0)
    srows.append(dict(window=f"{a}-{b}", units_compared=int(both.sum()), same_class=int(same.sum()),
                      share_national_same_class_pct=W.loc[same, "share_of_national_pct"].sum(),
                      relative_decoupling_units_in_window=int((rel_k & (W[f"g_V_{a}_{b}"] > 0)).sum()),
                      units_with_positive_GSVA_trend=int((W[f"g_V_{a}_{b}"] > 0).sum()),
                      same_relative_status=int((rel_base == rel_k)[pos].sum()), n_pos=int(pos.sum())))
# national growth by window
for a, b in windows:
    pv = q.pivot(index="Year", columns="State", values="E").loc[a:b].dropna(axis=1).sum(axis=1)
    pvv = q.pivot(index="Year", columns="State", values="V").loc[a:b].dropna(axis=1).sum(axis=1)
    x = np.arange(a, b + 1, dtype=float)
    gE, _ = trend(pv.values, x)
    gV, _ = trend(pvv.values, x)
    print(f"national {a}-{b}: emissions {100*gE:.2f} %/yr, GSVA {100*gV:.2f} %/yr, e={gE/gV:.2f}")
S = pd.DataFrame(srows)
S.round(2).to_csv(os.path.join(OUT, "sustainability_window_summary.csv"), index=False)

# ---- figure: component indices against GSVA
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.4, 3.9))
styles = {"Rice": ("-", "o"), "Livestock": ("--", "s"), "Fertiliser": ("-.", "^"), "Residue burning": (":", "D")}
for c, (ls, mk) in styles.items():
    pv = q.pivot(index="Year", columns="State", values=c).loc[Y0:Y1].dropna(axis=1).sum(axis=1)
    ax.plot(pv.index, 100 * pv / pv.loc[Y0], ls=ls, marker=mk, ms=3.5, lw=1.5, label=c)
pvv = q.pivot(index="Year", columns="State", values="V").loc[Y0:Y1].dropna(axis=1).sum(axis=1)
ax.plot(pvv.index, 100 * pvv / pvv.loc[Y0], color="black", lw=2.2, label="Agricultural GSVA")
ax.set_ylabel(f"Index ({Y0} = 100)")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sust_fig4_components.png"), dpi=600)

# ---- figure: emission intensity per sown hectare, all units with area data
D = summ.dropna(subset=["tCO2e_per_ha"]).sort_values("tCO2e_per_ha")
fig, ax = plt.subplots(figsize=(6.4, 6.2))
marks = {"Livestock": "o", "Rice": "s"}
ypos = pd.Series(np.arange(len(D)), index=D.index)
for src, mk in marks.items():
    m = D.dominant_source == src
    ax.scatter(D.tCO2e_per_ha[m], ypos[m],
               s=14 + 3.2 * D.share_of_national_pct[m], marker=mk, facecolors="0.25" if src == "Livestock" else "none",
               edgecolors="0.25", label=f"Largest source: {src.lower()}")
ax.axvline(D.tCO2e_per_ha.median(), color="0.6", lw=0.8, ls="--")
ax.set_yticks(ypos.values)
ax.set_yticklabels(D.State.str.replace("Andhra Pradesh + Telangana", "AP + Telangana"))
ax.set_xlabel("Emissions per sown hectare, 2021-2023 mean (t CO2e per ha)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.tick_params(axis="y", labelsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sust_fig5_intensity.png"), dpi=600)

print(N.round(2).to_string(index=False))
print(counts.to_string())
print(S.round(1).to_string(index=False))
print(W[["State", "share_of_national_pct"] + [f"class_{a}_{b}" for a, b in windows]].head(12).to_string(index=False))
