"""
Emission intensity, decoupling and source profiles by state, 2011-2023, from the state-year panel.

Window: 2011-2023 (constant-price agricultural GSVA starts in 2011-12; gross sown area ends 2023).
Unit of analysis: states/UTs, with Andhra Pradesh and Telangana combined (the two are one unit before 2014 in the
sources). GSVA is the RBI Handbook table 30 (constant 2011-12 prices; assumed to be in Rs lakh, consistent with
Punjab ~ Rs 50,000 crore in 2011-12), so intensities per rupee are given as kg CO2e per Rs and are indicative;
indices and growth rates do not depend on the unit.

Decoupling follows the Tapio elasticity classes: e = (growth of emissions) / (growth of agricultural GSVA), growth
= log-linear trend over the window. Trends are plain OLS slopes with a t-interval; with 13 autocorrelated annual
points the interval is optimistic, so a class is only reported as firm when the interval for e excludes a class border.

Outputs: results/state_panel/sustainability_state_summary.csv, sustainability_national_indices.csv,
results/figures/sust_fig*.png
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv")
OUT = os.path.join(BASE, "results/state_panel")
FIG = os.path.join(BASE, "results/figures")
os.makedirs(FIG, exist_ok=True)
Y0, Y1 = 2011, 2023

p = pd.read_csv(PANEL)
p = p.rename(columns={"Year_Standardized": "Year"})
comp = {
    "Rice": ["GHG_Rice_CH4_ktCO2e"],
    "Livestock": ["GHG_Livestock_CH4_ktCO2e", "GHG_Livestock_N2O_ktCO2e"],
    "Fertiliser": ["GHG_FertilizerSoil_N2O_ktCO2e"],
    "Residue burning": ["GHG_ResidueBurning_CH4_ktCO2e", "GHG_ResidueBurning_N2O_ktCO2e"],
}
for k, cols in comp.items():
    p[k] = p[cols].sum(axis=1, min_count=1)
p["E"] = p["Target_Agri_GHG_ktCO2e"]
p["V"] = p["GSVA_Agriculture_Constant"]
p["A"] = p["Gross_Sown_Area"]            # '000 ha
p["F"] = p["Total_Foodgrains_Production"]  # '000 t
p["Eobs"] = p[["Rice", "Fertiliser", "Residue burning"]].sum(axis=1, min_count=3)  # annually observed components

# combine Andhra Pradesh + Telangana
ap = p["State"].isin(["Andhra Pradesh", "Telangana"])
cols = ["E", "Eobs", "V", "A", "F"] + list(comp)
sub = p[ap].groupby("Year")
comb = sub[cols].sum(min_count=2).reset_index()
comb["State"] = "Andhra Pradesh + Telangana"
q = pd.concat([p[~ap][["State", "Year"] + cols], comb[["State", "Year"] + cols]], ignore_index=True)
q = q[(q.Year >= Y0) & (q.Year <= Y1)]


def slope(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    r = stats.linregress(x[ok], np.log(y[ok]))
    return r.slope, r.stderr, ok.sum()


rows = []
for s, d in q.groupby("State"):
    d = d.sort_values("Year")
    x = d.Year.values.astype(float)
    gE, seE, nE = slope(d.E.values, x)
    gV, seV, nV = slope(d.V.values, x)
    gA, _, _ = slope(d.A.values, x)
    gO, seO, _ = slope(d.Eobs.values, x)
    if not np.isfinite(gE) or not np.isfinite(gV) or nV < 10:
        continue
    last = d[d.Year >= Y1 - 2]
    flat = abs(gV) < 0.005   # GSVA trend below 0.5 %/yr: elasticity unstable, class not assigned
    e = gE / gV if not flat else np.nan
    # delta-method standard error for e
    se_e = np.sqrt((seE / gV) ** 2 + (gE * seV / gV ** 2) ** 2) if not flat else np.nan
    rows.append(dict(
        State=s, n_years=int(nV),
        E_ktCO2e_2021_23=last.E.mean(),
        g_E_pct_yr=100 * gE, g_V_pct_yr=100 * gV, g_A_pct_yr=100 * gA if np.isfinite(gA) else np.nan,
        elasticity=e, elasticity_se=se_e,
        g_Eobs_pct_yr=100 * gO, elasticity_obs=gO / gV if not flat else np.nan,
        kgCO2e_per_Rs=10 * last.E.sum() / last.V.sum() if last.V.notna().all() else np.nan,
        tCO2e_per_ha=last.E.sum() / last.A.sum() if last.A.notna().all() else np.nan,
        tCO2e_per_t_foodgrain=last.E.sum() / last.F.sum() if last.F.notna().all() else np.nan,
        **{f"share_{k}": last[k].sum() / last[list(comp)].sum().sum() for k in comp},
    ))
S = pd.DataFrame(rows)
nat = S.E_ktCO2e_2021_23.sum()
S["share_of_national_pct"] = 100 * S.E_ktCO2e_2021_23 / nat


def tapio(g_e, g_v, e):
    if g_v > 0:
        if g_e < 0: return "strong decoupling"
        if e < 0.8: return "weak decoupling"
        if e <= 1.2: return "expansive coupling"
        return "expansive negative decoupling"
    if g_e > 0: return "strong negative decoupling"
    if e < 0.8: return "weak negative decoupling"   # both falling, emissions falling slower
    return "recessive"                               # value added falling, emissions falling


S["class"] = [tapio(a, b, c) if np.isfinite(c) else "no clear GSVA trend" for a, b, c in zip(S.g_E_pct_yr, S.g_V_pct_yr, S.elasticity)]
S["class_observed_only"] = [tapio(a, b, c) if np.isfinite(c) else "no clear GSVA trend"
                            for a, b, c in zip(S.g_Eobs_pct_yr, S.g_V_pct_yr, S.elasticity_obs)]
S["class_robust"] = S["class"] == S["class_observed_only"]
borders = [0, 0.8, 1.2]
S["class_firm"] = [bool(np.isfinite(e) and np.isfinite(se) and all(abs(e - b) > 1.645 * se for b in borders))
                   for e, se in zip(S.elasticity, S.elasticity_se)]
shares = S[[c for c in S if c.startswith("share_") and c != "share_of_national_pct"]]
S["dominant_source"] = shares.idxmax(axis=1).str.replace("share_", "")
S["dominant_share_pct"] = 100 * shares.max(axis=1)
S = S.sort_values("E_ktCO2e_2021_23", ascending=False)
S.round(4).to_csv(os.path.join(OUT, "sustainability_state_summary.csv"), index=False)

# national indices on the balanced set of units
bal = S.State.tolist()
qq = q[q.State.isin(bal)]
qb = pd.DataFrame({c: qq.pivot(index="Year", columns="State", values=c).dropna(axis=1).sum(axis=1)
                   for c in ["E", "Eobs", "V", "A", "F"]})  # each series on units observed in every year
idx = (qb / qb.loc[Y0] * 100)
idx.columns = ["Emissions (all components)", "Emissions (rice, fertiliser, residue)", "Agricultural GSVA (constant prices)",
               "Gross sown area", "Foodgrain production"]
idx.round(2).to_csv(os.path.join(OUT, "sustainability_national_indices.csv"))

# ---- figures
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for c, st in zip(idx.columns, ["-", "--", ":", "-.", (0, (1, 1))]):
    ax.plot(idx.index, idx[c], ls=st, lw=1.8, label=c)
ax.set_ylabel(f"Index ({Y0} = 100)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sust_fig1_indices.png"), dpi=600)

fig, ax = plt.subplots(figsize=(6.4, 5.2))
big = S.E_ktCO2e_2021_23 / S.E_ktCO2e_2021_23.max()
rob = (S["class"] == S["class_observed_only"]) & (S["class"] != "no clear GSVA trend")
for flag, fc, lab in [(True, "0.25", "class unchanged without livestock"), (False, "none", "class changes without livestock")]:
    m = rob == flag
    ax.scatter(S.g_V_pct_yr[m], S.g_E_pct_yr[m], s=20 + 160 * big[m], facecolors=fc, edgecolors="0.25", alpha=0.75, label=lab)
xl, yl = (-5, 7), (-4, 3)
ax.plot(xl, xl, color="0.6", lw=0.8)
ax.plot(xl, [0.8 * v for v in xl], color="0.8", lw=0.6, ls="--")
ax.axhline(0, color="0.6", lw=0.8)
ax.axvline(0, color="0.6", lw=0.8)
ax.set_xlim(*xl)
ax.set_ylim(*yl)
for _, r in S.head(8).iterrows():
    ax.annotate(r.State.replace("Andhra Pradesh + Telangana", "AP+TG"), (r.g_V_pct_yr, r.g_E_pct_yr), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel(f"Growth of agricultural GSVA, % per year ({Y0}-{Y1})")
ax.set_ylabel("Growth of emissions, % per year")
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.text(0.02, 0.97, "Delhi (GSVA -8.7 %/yr) lies outside the axes", transform=ax.transAxes, fontsize=6.5, va="top")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sust_fig2_decoupling.png"), dpi=600)

top = S.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
left = np.zeros(len(top))
for k, c, h in zip(comp, ["#4c78a8", "#e45756", "#72b7b2", "#b79f4a"], ["//", "", "xx", ".."]):
    v = 100 * top[f"share_{k}"].values
    ax.barh(top.State.str.replace("Andhra Pradesh + Telangana", "AP + Telangana"), v, left=left, label=k, color=c,
            hatch=h, edgecolor="white", linewidth=0.4)
    left += v
ax.set_xlabel("Share of state agricultural emissions, 2021-2023 (%)")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sust_fig3_sources.png"), dpi=600)

print("units:", len(S), "| national 2021-23 mean (balanced units): %.0f kt" % nat)
print(S["class"].value_counts().to_string())
print("firm classes:", int(S.class_firm.sum()), "| class unchanged when livestock removed:", int(S.class_robust.sum()))
print(S["class_observed_only"].value_counts().to_string())
print(idx.loc[[Y0, Y1]].round(1).to_string())
print(S[["State", "share_of_national_pct", "g_E_pct_yr", "g_V_pct_yr", "elasticity", "class", "class_observed_only", "class_firm", "tCO2e_per_ha",
         "dominant_source"]].round(2).head(20).to_string(index=False))
