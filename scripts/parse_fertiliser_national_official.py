"""
Parse the official All-India fertiliser consumption table (N, P2O5, K2O; 1950-51 to 2023-24, thousand t) from
data/raw/Farm_Activity_Data/Fertilizer_use/Fertilizer consumption_1951-2024.pdf (table 6.01(a); the Fertiliser Association of India, Fertiliser Statistics
should be confirmed and cited), and cross-check it against FAOSTAT's India series in the same folder.

Used as NATIONAL CONTROL TOTALS for fertiliser nitrogen: state values are rescaled each year so that their sum equals the
official All-India N (see build_tier1_ghg_target.py). Fiscal years are labelled by their starting calendar year.
Found while auditing data/raw: the RBI Handbook's N+P+K state sums already equal this table to 0.1% for 2013-2021, and
the secondary compiled N series overstated 2004-2006 by 4-19%.

Output: data/processed/state_panel/fertiliser_national_official.csv  (Year_Standardized, N_ktN, P2O5_kt, K2O_kt, NPK_kt)
"""
import io
import os
import re
import numpy as np
import pandas as pd
import pdfplumber

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(BASE, "data/raw/Farm_Activity_Data/Fertilizer_use/Fertilizer consumption_1951-2024.pdf")
FAO = os.path.join(BASE, "data/raw/Farm_Activity_Data/Fertilizer_use/Fertilizer Use_FAOSTAT.xls")
OUT = os.path.join(BASE, "data/processed/state_panel/fertiliser_national_official.csv")

with pdfplumber.open(PDF) as pdf:
    text = "\n".join((p.extract_text() or "") for p in pdf.pages)
rows = []
for line in text.splitlines():
    m = re.match(r"^(\d{4})-(\d{2,4})\b(.*)$", line.strip())
    if not m:
        continue
    start = int(m.group(1))
    if start < 2000:
        continue
    rest = re.sub(r"(?<=\s)(\d) (\d,\d{3}\.\d)", r"\1\2", " " + m.group(3))   # repair '1 9,101.3' -> '19,101.3'
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.\d", rest)]
    if len(nums) >= 4:
        n, p, k, tot = nums[-4:]
        assert abs(n + p + k - tot) < 1.5, f"{start}: N+P+K {n + p + k:.1f} != total {tot:.1f}"
        rows.append({"Year_Standardized": start, "N_ktN": n, "P2O5_kt": p, "K2O_kt": k, "NPK_kt": tot})
df = pd.DataFrame(rows).drop_duplicates("Year_Standardized").sort_values("Year_Standardized")
assert df.Year_Standardized.min() <= 2004 and df.Year_Standardized.max() >= 2023, "expected 2004-2023"

fao = pd.read_excel(FAO) if False else None
try:
    fao = pd.read_excel(FAO)
except Exception:
    fao = pd.read_html(io.StringIO(open(FAO, encoding="utf-8", errors="ignore").read()))[0]
fn = fao[fao.Item.str.contains("nitrogen", case=False)].set_index("Year").Value / 1000.0   # t -> kt
cmp_ = df.set_index("Year_Standardized").N_ktN.to_frame().join(fn.rename("FAO_N_kt"), how="inner")
cmp_ = cmp_.loc[2007:2021]
dev = (cmp_.N_ktN / cmp_.FAO_N_kt - 1).abs()
assert dev.max() < 0.03, f"official table and FAOSTAT disagree by {dev.max():.1%} (2007-2021)"
df.to_csv(OUT, index=False)
print(f"saved {len(df)} years ({df.Year_Standardized.min()}-{df.Year_Standardized.max()}) -> {OUT}")
print(f"FAOSTAT nitrogen agrees with the official table within {dev.max():.1%} over 2007-2021 (max)")
print(df[df.Year_Standardized.isin([2004, 2013, 2020, 2023])].round(1).to_string(index=False))
