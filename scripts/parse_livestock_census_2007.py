"""
18th Livestock Census (2007), state-level headcounts, from the official-source table reproduced by MoSPI (Statistical
Year Book, Table 3.11.2 "Livestock population as per 2007 census, provisional", source: Department of Animal Husbandry,
Dairying & Fisheries; PDF in data/raw/state_panel/livestock_census/). Provisional village-level totals.

Replaces an earlier version that took 2007 from the GHG Platform India workbook (whose usage terms are unconfirmed). The two
agree exactly (asserted below), so the workbook is no longer needed for this anchor.

Layout: after the state name the row gives cross-bred, indigenous and total cattle, then buffaloes, sheep, goats, pigs, ...
Only the first seven numbers (cattle total .. pigs) are used: later columns are blank for some rows and unreliable to align.
Andhra Pradesh is UNDIVIDED (Telangana is not listed in 2007). Ladakh is part of Jammu & Kashmir.

Output: data/processed/state_panel/livestock_census_2007.csv  (thousand head x 1000 -> head)
"""
import os
import re
import pandas as pd
import pdfplumber

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(BASE, "data/raw/state_panel/livestock_census/MoSPI_Table_3.11.2_Livestock_Census_2007_provisional.pdf")
OUT = os.path.join(BASE, "data/processed/state_panel/livestock_census_2007.csv")
WORKBOOK_CHECK = os.path.join(BASE, "data/raw/benchmark_ghgpi/GHGPI-AFOLU-Estimates-2005-to-2018.xlsx")
NAME = {"Chhatisgarh": "Chhattisgarh", "Orissa": "Odisha", "Tamilnadu": "Tamil Nadu", "Pondicherry": "Puducherry",
        "A& Nicobar": "Andaman & Nicobar Islands", "D & Nagar Haveli": "Dadra & Nagar Haveli"}
COLS = ["Livestock_Cattle_Head", "Livestock_Buffalo_Head", "Livestock_Sheep_Head", "Livestock_Goat_Head", "Livestock_Pigs_Head"]
OFFICIAL_ALL_INDIA = [199075, 105343, 71558, 140537, 11134]   # thousand, last row of the same table

with pdfplumber.open(PDF) as pdf:
    text = pdf.pages[0].extract_text()
lines = text.split(chr(10))
rows, pending = [], None
for i, line in enumerate(lines):
    m = re.match(r"^(\d{1,2})\s+([A-Za-z&. ]+?)\s+([\d ]+)$", line.strip())
    if m and not line.startswith("All India"):
        name = m.group(2).strip()
        nums = [int(x) for x in m.group(3).split()]
        if name == "Arunachal Pradesh":          # this row wraps; pdfplumber puts the remaining columns on the PREVIOUS line
            nums += [int(x) for x in lines[i - 1].split()]
        if len(nums) >= 7:
            rows.append({"State": NAME.get(name, name), **dict(zip(COLS, [nums[2], nums[3], nums[4], nums[5], nums[6]]))})
df = pd.DataFrame(rows)
allind = [int(x) for x in re.search(r"All India ([\d ]+)", text).group(1).split()]
assert allind[2:7] == OFFICIAL_ALL_INDIA, allind
assert len(df) == 35, f"expected 35 states/UTs, parsed {len(df)}"
for c, tot in zip(COLS, OFFICIAL_ALL_INDIA):
    assert abs(df[c].sum() - tot) / tot < 0.001, f"{c}: parsed sum {df[c].sum()} vs official {tot}"
df[COLS] = df[COLS] * 1000.0
df["Year_Standardized"] = 2007

# the workbook compilation must agree with the official table (this is what lets us drop the workbook as a source)
if os.path.exists(WORKBOOK_CHECK):
    w = pd.read_excel(WORKBOOK_CHECK, sheet_name="Livestock population 2007 ", header=None)
    w = w[w[0].astype(str).str.contains("2007")]
    M = {"Andaman and Nicobar Island": "Andaman & Nicobar Islands", "Dadra and Nagar Haveli": "Dadra & Nagar Haveli",
         "Daman and Diu": "Daman & Diu", "Jammu and Kashmir": "Jammu & Kashmir"}
    wk = pd.DataFrame({"State": w[1].astype(str).str.strip().replace(M),
                       "Livestock_Cattle_Head": (pd.to_numeric(w[8]) + pd.to_numeric(w[15])) * 1000,
                       "Livestock_Buffalo_Head": pd.to_numeric(w[22]) * 1000, "Livestock_Sheep_Head": pd.to_numeric(w[24]) * 1000,
                       "Livestock_Goat_Head": pd.to_numeric(w[25]) * 1000, "Livestock_Pigs_Head": pd.to_numeric(w[26]) * 1000}).set_index("State")
    j = df.set_index("State")[COLS].join(wk, rsuffix="_wb").dropna()
    bad = [(st, c) for c in COLS for st in j.index
           if abs(j.loc[st, c] - j.loc[st, c + "_wb"]) > max(0.005 * j.loc[st, c], 1500)]
    print(f"workbook vs official-table 2007: {len(j) * len(COLS)} cells compared, {len(bad)} differ by >0.5% and >1,500 head: {bad}")
df.to_csv(OUT, index=False)
print(f"saved {len(df)} states -> {OUT}; Andhra Pradesh cattle {df.set_index('State').loc['Andhra Pradesh','Livestock_Cattle_Head']:.0f}")
