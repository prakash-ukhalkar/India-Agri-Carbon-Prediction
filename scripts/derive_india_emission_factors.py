"""
Derive India-specific emission factors for the bottom-up target (v3) and write them as DATA with provenance.

Why: v2 used IPCC Tier-1 defaults and overstated livestock (x1.36) and fertiliser soil N2O (x2.08) against the GHG
Platform India (GHGPI) state inventory, which uses India-specific (NATCOM-II-based, partly IPCC 2006) factors. v3 adopts
the same published factor VALUES but derives only national-average effective factors, using the 2019 Livestock Census
national composition (dairy/non-dairy, indigenous/cross-bred, age class), so state differences in composition are
NOT reproduced. This keeps the panel simple and leaves a measurable gap to the Tier-2 inventory.

IMPORTANT: because the factor values come from the same compilation used as the benchmark, agreement
with GHGPI validates implementation and activity-data handling, NOT the factors themselves. Factors should be cited
to the primary sources named by GHGPI (NATCOM-II; Gupta et al. 2009; Bhatia et al. 2004/2013; IPCC 2006) after the
author verifies them there.

Input : data/raw/benchmark_ghgpi/GHGPI-AFOLU-Estimates-2005-to-2018.xlsx (see fetch_ghgpi_benchmark.py)
Output: data/processed/state_panel/india_specific_emission_factors.csv
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "benchmark_ghgpi", "GHGPI-AFOLU-Estimates-2005-to-2018.xlsx")
OUT = os.path.join(BASE, "data", "processed", "state_panel", "india_specific_emission_factors.csv")
SRC = "GHGPI AFOLU workbook v4.0 (sheet: {}); factor origin per GHGPI: NATCOM-II / IPCC 2006"

x = pd.ExcelFile(RAW)

# ---- livestock: per-head factors by category (sheet 'Livestock Emission Factors ') ----
ef = x.parse("Livestock Emission Factors ", header=None)
# rows 4-15 = indigenous (dairy, 0-1, 1-3, adult), cross-bred (same), buffalo (same); cols 4,5,6 = enteric, manure CH4, manure N2O
assert str(ef.iloc[4, 1]).strip().startswith("Indigenous") and str(ef.iloc[8, 1]).strip().startswith("Crossbred") and str(ef.iloc[12, 1]).strip().startswith("Buffalo"), "factor-sheet layout changed"
blocks = {"indigenous": ef.iloc[4:8, [4, 5, 6]].astype(float).values,
          "crossbred": ef.iloc[8:12, [4, 5, 6]].astype(float).values,
          "buffalo": ef.iloc[12:16, [4, 5, 6]].astype(float).values}
small_rows = {"sheep": 16, "goat": 17, "horse": 18, "donkey": 19, "camel": 20, "pig": 21}
small = {k: ef.iloc[i, [4, 5, 6]].astype(float).values for k, i in small_rows.items()}

# ---- 2019 census national composition ('Livestock Population 2019', India row; thousands) ----
pop = x.parse("Livestock Population 2019", header=None)
india = pop.iloc[4:, :].copy()
row = [r for _, r in pop.iterrows() if str(r.iloc[1]).strip() == "India"][0].tolist()
v = {i: float(row[i]) for i in range(2, len(row)) if pd.notna(row[i])}
# column map (verified against printed header): cross-bred dairy, 0-1, 1-3, non-dairy adult total(7);
# indigenous dairy(9), 0-1(10), 1-3(11), non-dairy adult total(14); buffalo dairy(16), 0-1(17), 1-3(18), adult non-dairy(21)
comp = {"crossbred": [v[2], v[3], v[4], v[7]], "indigenous": [v[9], v[10], v[11], v[14]],
        "buffalo": [v[16], v[17], v[18], v[21]]}
checks = {"cattle_total_2019 (thousand)": sum(sum(comp[k]) for k in ("crossbred", "indigenous")),
          "buffalo_total_2019 (thousand)": sum(comp["buffalo"])}


def effective(group_names):
    w = np.concatenate([comp[g] for g in group_names])
    f = np.vstack([blocks[g] for g in group_names])
    return (w[:, None] * f).sum(axis=0) / w.sum()


cattle = effective(["crossbred", "indigenous"])
buffalo = effective(["buffalo"])

rows = []


def add(name, value, unit, note, sheet):
    rows.append({"parameter": name, "value": value, "unit": unit, "note": note, "source": SRC.format(sheet)})


add("cattle_enteric", cattle[0], "kg CH4/head/yr", "census-composition-weighted (2019, national)", "Livestock Emission Factors; Livestock Population 2019")
add("cattle_manure_ch4", cattle[1], "kg CH4/head/yr", "weighted", "same")
add("cattle_manure_n2o", cattle[2], "kg N2O/head/yr", "weighted", "same")
add("buffalo_enteric", buffalo[0], "kg CH4/head/yr", "weighted", "same")
add("buffalo_manure_ch4", buffalo[1], "kg CH4/head/yr", "weighted", "same")
add("buffalo_manure_n2o", buffalo[2], "kg N2O/head/yr", "weighted", "same")
for k, vals in small.items():
    add(f"{k}_enteric", vals[0], "kg CH4/head/yr", "category factor", "Livestock Emission Factors")
    add(f"{k}_manure_ch4", vals[1], "kg CH4/head/yr", "category factor", "Livestock Emission Factors")
    add(f"{k}_manure_n2o", vals[2], "kg N2O/head/yr", "category factor", "Livestock Emission Factors")

# ---- rice: regime emission factors and national area shares ('Rice emissions') ----
rice = x.parse("Rice emissions", header=None)
regime_ef = {str(rice.iloc[i, 1]).strip(): float(rice.iloc[i, 2]) for i in range(1, 8)}
hdr = [i for i in range(len(rice)) if str(rice.iloc[i, 1]).startswith("Rice Cultivation Area")][0]
year_col = {int(float(rice.iloc[hdr, j])): j for j in range(2, rice.shape[1]) if pd.notna(rice.iloc[hdr, j])}
col18 = year_col[2018]
# the area blocks use longer labels than the factor table
ALIAS = {"Continuous Flooding": "Continuous Flooding", "Intermittent Single Aeration": "Single Aeration",
         "Intermittent Mutiple Aeration": "Multiple Aeration", "Rainfed Flood Prone": "Flood Prone",
         "Rainfed Drought Prone": "Drought Prone", "Deep Water": "Deep Water", "Upland": "Upland"}
areas, current = {}, None
for i in range(hdr + 1, len(rice)):
    lab = str(rice.iloc[i, 1]).strip()
    if lab in ALIAS:
        current = ALIAS[lab]
    elif lab == "India" and current and current not in areas:
        areas[current] = float(rice.iloc[i, col18])
assert set(areas) == set(regime_ef), f"rice regimes not all matched: {set(regime_ef) ^ set(areas)}"
assert 40000 < sum(areas.values()) < 46000, "regime areas should sum to ~43 Mha"
tot = sum(areas.values())
rice_eff = sum(regime_ef[r] * a for r, a in areas.items()) / tot
add("rice_ch4_effective", rice_eff, "kg CH4/ha/season", f"area-weighted over water regimes, 2018; regimes: " +
    ", ".join(f"{r} {a/tot:.1%} x {regime_ef[r]:g}" for r, a in areas.items()), "Rice emissions")

# ---- fertiliser N2O (Fertilizer Consumption Emission, parameter block) ----
add("fert_direct_ef", 0.0058, "kg N2O-N/kg N", "direct emission factor (GHGPI sheet, row 'Direct Emissions')", "Fertilizer Consumption Emission")
add("fert_ef_deposition", 0.005, "kg N2O-N/kg N", "indirect, atmospheric deposition", "Fertilizer Consumption Emission")
add("fert_ef_leaching", 0.005, "kg N2O-N/kg N", "indirect, leaching", "Fertilizer Consumption Emission")
add("fert_frac_volatilised", 0.15, "fraction", "volatilised N (urea and other)", "Fertilizer Consumption Emission")
add("fert_frac_leached", 0.10, "fraction", "leached N", "Fertilizer Consumption Emission")
add("gwp_ch4_ar2", 21.0, "-", "IPCC SAR GWP-100, used by India", "Rice emissions")
add("gwp_n2o_ar2", 310.0, "-", "IPCC SAR GWP-100, used by India", "Fertilizer Consumption Emission")

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"census checks: {checks}")
print(f"regime EFs: {regime_ef}")
print(f"rice effective EF = {rice_eff:.1f} kg CH4/ha (regime areas 2018 total {tot:.0f} thousand ha)")
print(out[["parameter", "value", "unit"]].round(4).to_string(index=False))
