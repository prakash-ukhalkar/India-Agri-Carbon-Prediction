"""
Extract the state-wise total nitrogen consumption series (thousand t N, 2004-05 to 2018-19) compiled in the GHG Platform
India workbook (sheet 'Fertilizer Consumption Emission'; underlying statistics from the Department of Fertilizers /
Fertiliser Association of India, per GHGPI). Used ONLY to (i) back-fill 2004-2012, when the RBI Handbook reports no
fertiliser data, and (ii) calibrate a state-specific N share of N+P+K on the 2014-2018 overlap.

This is a secondary compilation: back-filled years are flagged `compiled_secondary`, and agreement with GHGPI for those
years is not independent evidence. Usage terms of the workbook are unconfirmed.

Output: data/processed/state_panel/fertiliser_N_compiled_2004_2018.csv  (State, Year_Standardized, N_compiled_ktN)
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "benchmark_ghgpi", "GHGPI-AFOLU-Estimates-2005-to-2018.xlsx")
OUT = os.path.join(BASE, "data", "processed", "state_panel", "fertiliser_N_compiled_2004_2018.csv")
STATE_MAP = {"Andaman and Nicobar Island": "Andaman & Nicobar Islands", "Dadra and Nagar Haveli": "Dadra & Nagar Haveli",
             "Daman and Diu": "Daman & Diu", "Jammu and Kashmir": "Jammu & Kashmir"}

f = pd.read_excel(RAW, sheet_name="Fertilizer Consumption Emission", header=None)
HDR = 4  # row holding the fiscal-year labels ("2004-05", ...)
cols = {j: int(str(f.iloc[HDR, j])[:4]) for j in range(2, f.shape[1])
        if str(f.iloc[HDR, j])[:2] == "20" and "-" in str(f.iloc[HDR, j])}
assert min(cols.values()) == 2004 and max(cols.values()) == 2018, f"unexpected year header: {sorted(cols.values())}"
rows = []
for i in range(6, 45):
    st = str(f.iloc[i, 1]).strip()
    if st == "nan" or st.startswith(("Quantity", "Total N", "Emission")):
        continue
    for j, y in cols.items():
        v = f.iloc[i, j]
        if pd.notna(v):
            rows.append({"State": STATE_MAP.get(st, st), "Year_Standardized": y, "N_compiled_ktN": float(v)})
out = pd.DataFrame(rows)
nat = out.groupby("Year_Standardized").N_compiled_ktN.sum()
# the workbook states the national total in the row 'Total N Quantity Consumed'; our state sum must agree closely
total_row = [float(f.iloc[6, j]) for j in sorted(cols)]
assert all(abs(a - b) / b < 0.01 for a, b in zip(nat.values, total_row)), "state sum != national total row"
out.to_csv(OUT, index=False)
print(f"saved {len(out)} rows, {out.State.nunique()} states -> {OUT}")
print("national N (Mt):", (nat / 1000).round(1).to_dict())
