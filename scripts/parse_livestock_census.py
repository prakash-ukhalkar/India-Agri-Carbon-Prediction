"""
Parse the 20th Livestock Census (2019) state-wise headcount file into long
format and merge into the state_year_panel as Year_Standardized=2019 rows.

Single-year snapshot for now (only the 2019 edition was obtained). Earlier
editions (1997/2003/2007/2012) would need to be added the same way to give
the panel more than one time point for livestock - flagged as a follow-up,
not fabricated by interpolating a single year backward/forward.
"""
import os
import xlrd
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _panel_io import upsert_into_panel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLS_PATH = os.path.join(BASE_DIR, "data", "raw", "state_panel", "livestock_census", "datafile.xls")
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "livestock_census_2019.csv")

# Handbook/RBI canonical state names used elsewhere in the panel
NAME_MAP = {
    "A & N Islands": "Andaman & Nicobar Islands",
    "D &N Haveli": "Dadra & Nagar Haveli",
    "Daman And Diu": "Daman & Diu",
}

SPECIES_COLS = [
    "Cattle", "Buffalo", "Sheep", "Goat", "Pigs", "Horse &Ponies",
    "Mules", "Donkeys", "Camel", "Yaks", "Mithun", "Total Livestock",
]


def main():
    wb = xlrd.open_workbook(XLS_PATH, ignore_workbook_corruption=True)
    sh = wb.sheet_by_index(0)
    header = sh.row_values(0)
    rows = [sh.row_values(r) for r in range(1, sh.nrows)]
    df = pd.DataFrame(rows, columns=header)
    df = df[df["State/UT"] != "All India - Total"].copy()
    df["State"] = df["State/UT"].replace(NAME_MAP)
    df["Year_Standardized"] = 2019

    keep_cols = ["State", "Year_Standardized"] + SPECIES_COLS
    df = df[keep_cols].rename(columns={c: f"Livestock_{c.strip().replace(' ', '_').replace('&', 'and')}_Head" for c in SPECIES_COLS})
    df.to_csv(OUT_PATH, index=False)
    print(f"Parsed {len(df)} states -> {OUT_PATH}")

    panel = pd.read_csv(PANEL_PATH)
    merged = upsert_into_panel(panel, df)
    merged.to_csv(PANEL_PATH, index=False)
    n_matched = merged.loc[merged["Year_Standardized"] == 2019, "Livestock_Cattle_Head"].notna().sum()
    print(f"Merged into panel: {n_matched} of 2019 rows populated with livestock headcounts")

    # Report any 2019-panel states that didn't find a livestock match (name mismatch)
    panel_2019_states = set(panel.loc[panel["Year_Standardized"] == 2019, "State"])
    livestock_states = set(df["State"])
    missing = panel_2019_states - livestock_states
    if missing:
        print(f"States in panel's 2019 rows with NO livestock match: {sorted(missing)}")


if __name__ == "__main__":
    main()
