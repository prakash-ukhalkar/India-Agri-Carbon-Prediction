"""
Independent national cross-checks against the FAOSTAT files in data/raw (all are national, so they cannot validate state
values). Output: results/state_panel/faostat_crosschecks.csv  (long format: check, year, value columns)

 1. Fertiliser nitrogen: FAOSTAT India N (official questionnaire data) vs the panel's national N applied.
 2. Livestock trend after the last census: FAOSTAT 'amount excreted in manure (N)' by species, used as a proxy for stock
    changes (excretion = stocks x a fixed rate), indexed to 2019 = 100. Tests how wrong 'held constant after 2019' could be.
 3. Rice CH4: FAOSTAT's IPCC Tier-1 estimate vs the panel (India-specific factor) vs BUR4 (3,225 Gg CH4 in 2020), showing what
    generic Tier-1 defaults do to the rice component.
"""
import io
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data/raw")
panel = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv"))


def load(path):
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_html(io.StringIO(open(path, encoding="utf-8", errors="ignore").read()))[0]


rows = []
# 1 fertiliser N
fao = load(os.path.join(RAW, "Farm_Activity_Data/Fertilizer_use/Fertilizer Use_FAOSTAT.xls"))
n_fao = fao[fao.Item.str.contains("nitrogen", case=False)].set_index("Year").Value / 1e3        # kt N
n_panel = panel.groupby("Year_Standardized").Fertiliser_N_applied_ktN.sum()
for y in range(2004, 2024):
    rows.append({"check": "fertiliser_N_panel_over_FAOSTAT", "year": y, "value": n_panel.get(y, np.nan) / n_fao.get(y, np.nan)})

# 2 livestock trend
man = load(os.path.join(RAW, "Farm_Activity_Data/Live_stock/Live_Stock_Manuer_FAOSTAT.xls"))
piv = man.pivot_table(index="Year", columns="Item", values="Value", aggfunc="sum")
for item in ["Cattle, dairy", "Cattle, non-dairy", "Buffalo", "Goats", "Sheep"]:
    for y in range(2019, 2024):
        rows.append({"check": f"faostat_stock_index_2019=100: {item}", "year": y, "value": piv.loc[y, item] / piv.loc[2019, item] * 100})
tot = piv[["Cattle, dairy", "Cattle, non-dairy"]].sum(axis=1)
for y in range(2019, 2024):
    rows.append({"check": "faostat_stock_index_2019=100: Cattle (dairy + non-dairy N excretion)", "year": y, "value": tot.loc[y] / tot.loc[2019] * 100})

# 3 rice CH4
em = load(os.path.join(RAW, "Emission_Data/Emissions_from_Crops_FAOSTAT.xls"))
rice = em[(em.Element == "Rice cultivation (Emissions CH4)") & (em.Item == "Rice")].set_index("Year").Value   # kt CH4
for y in (2011, 2015, 2018, 2020):
    ours = panel[panel.Year_Standardized == y].GHG_Rice_CH4_ktCO2e.sum() / 21.0
    rows.append({"check": "rice_CH4_kt: FAOSTAT Tier-1", "year": y, "value": rice.get(y, np.nan)})
    rows.append({"check": "rice_CH4_kt: panel (India-specific factor)", "year": y, "value": ours})
rows.append({"check": "rice_CH4_kt: BUR4", "year": 2020, "value": 3225.0})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, "results/state_panel/faostat_crosschecks.csv"), index=False)
pd.set_option("display.width", 200)
print(out.pivot_table(index="check", columns="year", values="value").round(2).to_string())
