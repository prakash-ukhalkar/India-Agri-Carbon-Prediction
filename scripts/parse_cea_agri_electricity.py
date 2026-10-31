"""
Parse state-wise agricultural electricity consumption (Table 9.3, "Electrical
Energy Sales to Ultimate Consumers, Category-wise/State-wise") from the CEA
General Review 2021 report, covering FY 2019-20, and merge into the panel.

Single-year snapshot for now (only this one report edition was obtained).
Getting additional editions (each year's General Review has the same Table
9.3 structure for that year) would extend this to multiple years - flagged
as a follow-up, not fabricated by interpolation.
"""
import os
import re
import pdfplumber
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _panel_io import upsert_into_panel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "raw", "state_panel", "cea_agri_electricity", "cea_general_review.pdf")
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "cea_agri_electricity_2019.csv")

TABLE_93_PAGE = 157  # 0-indexed page with the Traction/Agriculture/PWW/Misc/Total columns
YEAR = 2019  # FY 2019-20 -> Year_Standardized convention used elsewhere (starting year)

# Report spelling -> canonical name used in state_year_panel.csv
NAME_MAP = {
    "Jammu & Kashmir & Ladakh": "Jammu & Kashmir",
    "A & N Islands": "Andaman & Nicobar Islands",
    "D & N Haveli": "Dadra & Nagar Haveli",
}
SKIP_ROWS = {"Sub-Total (NR)", "Sub-Total (WR)", "Sub-Total (SR)", "Sub-Total (ER)", "Sub-Total (NER)", "Total (All India)"}

# State names as they appear in the report, longest-first for matching
REPORT_STATE_NAMES = sorted([
    "Chandigarh", "Delhi", "Haryana", "Himachal Pradesh", "Jammu & Kashmir & Ladakh",
    "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand",
    "Chhattisgarh", "Gujarat", "Madhya Pradesh", "Maharashtra", "Daman & Diu", "D & N Haveli", "Goa",
    "Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu", "Puducherry", "Lakshadweep",
    "Bihar", "Jharkhand $", "Jharkhand", "Odisha", "West Bengal $", "West Bengal", "Sikkim", "A & N Islands",
    "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura",
], key=len, reverse=True)


def parse_number(tok):
    tok = tok.strip().rstrip("$")
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        text = pdf.pages[TABLE_93_PAGE].extract_text()

    rows = []
    for line in text.split("\n"):
        line = line.strip()
        matched = None
        for name in REPORT_STATE_NAMES:
            if line.startswith(name + " "):
                matched = name.rstrip(" $")
                rest = line[len(name):].strip()
                break
        if matched is None or matched in SKIP_ROWS:
            continue
        values = [parse_number(t) for t in rest.split()]
        # Columns on this page: Traction, Agriculture, PWW&Sewage, Misc, Total
        if len(values) >= 5:
            agriculture_gwh = values[1]
            rows.append({"State": NAME_MAP.get(matched, matched), "Year_Standardized": YEAR, "Agri_Electricity_Consumption_GWh": agriculture_gwh})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Parsed {len(df)} states -> {OUT_PATH}")
    print(df.to_string(index=False))

    panel = pd.read_csv(PANEL_PATH)
    merged = upsert_into_panel(panel, df)
    merged.to_csv(PANEL_PATH, index=False)
    n_matched = merged.loc[merged["Year_Standardized"] == YEAR, "Agri_Electricity_Consumption_GWh"].notna().sum()
    print(f"\nMerged into panel: {n_matched} of {YEAR} rows populated with agri electricity consumption")


if __name__ == "__main__":
    main()
